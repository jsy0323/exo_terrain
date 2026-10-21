import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

import atexit
import yaml
import gymnasium as gym
from omegaconf import OmegaConf
from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support
from multiprocessing import shared_memory
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from env_wrapper import SB3RslVecEnv
from RL.runners import OnPolicyRunner
from hydra import main

from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager
from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11
from Env.MFG_Musculoskelet_V11.mfgenv.terrain import apply_terrain_config_overrides

gym.register(
    id='MFG_MS_V11',
    entry_point=MFG_Musculoskeletal_V11,
    max_episode_steps=200
)

class TMProxyManager(BaseManager): ...
TMProxyManager.register('TrajectoryManager', TrajectoryManager)

_SHM_HANDLES = [] 

def _to_shm(arr: np.ndarray):
    arr = np.asarray(arr, order="C")
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[...] = arr
    _SHM_HANDLES.append(shm)
    return {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}

def build_traj_shm_meta(tm_proxy: TrajectoryManager):
    n = int(tm_proxy.length() if hasattr(tm_proxy, "length") else len(tm_proxy))
    sf = float(tm_proxy.get_sample_frequency())
    meta = []
    for i in range(n):
        qpos, qvel = tm_proxy.get(i) 
        item = {
            "qpos":  _to_shm(qpos),
            "qvel":  _to_shm(qvel),
            "speed": float(tm_proxy.get_speed(i)),
            "foot":  int(tm_proxy.get_foot_start(i)),
            "periodic": bool(tm_proxy.is_periodic(i)),
            "repeat": int(tm_proxy.get_repeat_count(i)),
            "cycle_frames": int(tm_proxy.get_cycle_frames(i)),
            "confidence": float(tm_proxy.get_periodicity_confidence(i)),
        }
        meta.append(item)
    return {"sample_frequency": sf, "meta": meta}

def _cleanup_shared_memory():
    for shm in _SHM_HANDLES:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

atexit.register(_cleanup_shared_memory)


def _checkpoint_state_dict(checkpoint_path, device):
    checkpoint_path = os.fspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint has no model_state_dict: {checkpoint_path}"
        )
    clean_state = {
        str(key).replace("module.", ""): value
        for key, value in state_dict.items()
    }
    return checkpoint, clean_state


def _find_flat_exo_branch(state_dict, exo_action_dim):
    """Find the multi-agent actor branch whose final layer emits exo actions."""
    branch_ids = set()
    for key in state_dict:
        parts = key.split(".")
        if len(parts) >= 5 and parts[0] == "actor_branches" and parts[1].isdigit():
            branch_ids.add(int(parts[1]))

    candidates = []
    for branch_id in sorted(branch_ids):
        prefix = f"actor_branches.{branch_id}.net."
        output_layers = []
        for key, tensor in state_dict.items():
            if not key.startswith(prefix) or not key.endswith(".weight"):
                continue
            layer_token = key[len(prefix) :].split(".", 1)[0]
            if layer_token.isdigit() and getattr(tensor, "ndim", 0) == 2:
                output_layers.append((int(layer_token), tensor))
        if output_layers:
            _, final_weight = max(output_layers, key=lambda item: item[0])
            if int(final_weight.shape[0]) == int(exo_action_dim):
                candidates.append(branch_id)

    if len(candidates) != 1:
        raise ValueError(
            "Could not identify one flat-policy exo branch with output dimension "
            f"{exo_action_dim}; candidates={candidates}, branches={sorted(branch_ids)}"
        )
    return candidates[0]


def load_exo_actor_checkpoint(
    policy,
    checkpoint_path,
    device,
    load_noise_std=True,
):
    """Load only an exoskeleton actor from a flat or terrain-exo checkpoint."""
    policy = getattr(policy, "module", policy)
    if not hasattr(policy, "exo_actor") or not hasattr(policy, "exo_action_dim"):
        raise TypeError(
            "exo_checkpoint_path requires a policy exposing exo_actor and "
            "exo_action_dim (for example ActorCriticTerrainExo)."
        )

    _, state_dict = _checkpoint_state_dict(checkpoint_path, device)
    native_prefix = "exo_actor."
    if any(key.startswith(native_prefix) for key in state_dict):
        source_prefix = native_prefix
        source_description = "native terrain-exo actor"
    else:
        branch_id = _find_flat_exo_branch(state_dict, policy.exo_action_dim)
        source_prefix = f"actor_branches.{branch_id}."
        source_description = f"flat multi-agent branch {branch_id}"

    source_actor_state = {
        key[len(source_prefix) :]: tensor
        for key, tensor in state_dict.items()
        if key.startswith(source_prefix)
    }
    target_actor_state = policy.exo_actor.state_dict()
    missing = sorted(set(target_actor_state) - set(source_actor_state))
    unexpected = sorted(set(source_actor_state) - set(target_actor_state))
    shape_mismatches = [
        (
            key,
            tuple(source_actor_state[key].shape),
            tuple(target_actor_state[key].shape),
        )
        for key in sorted(set(target_actor_state) & set(source_actor_state))
        if source_actor_state[key].shape != target_actor_state[key].shape
    ]
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "Exo actor architecture mismatch while loading "
            f"{checkpoint_path}: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )
    policy.exo_actor.load_state_dict(source_actor_state, strict=True)

    loaded_noise_std = False
    if load_noise_std and hasattr(policy, "exo_log_std"):
        source_log_std = state_dict.get("exo_log_std")
        if source_log_std is None:
            source_log_std = state_dict.get("log_std")
            if source_log_std is not None:
                source_log_std = source_log_std.reshape(-1)[-int(policy.exo_action_dim) :]
        if source_log_std is not None:
            source_log_std = source_log_std.reshape_as(policy.exo_log_std)
            with torch.no_grad():
                policy.exo_log_std.copy_(
                    source_log_std.to(
                        device=policy.exo_log_std.device,
                        dtype=policy.exo_log_std.dtype,
                    )
                )
            loaded_noise_std = True

    print(
        f"Loaded {source_description} from {checkpoint_path} into "
        f"ActorCriticTerrainExo ({len(source_actor_state)} tensors, "
        f"noise_std={'loaded' if loaded_noise_std else 'kept'})."
    )

@main(version_base=None, config_path="configs", config_name="default_V11")
def hydra_main(cfg):
    cfg = yaml.safe_load(OmegaConf.to_yaml(cfg))
    env_id   = cfg["env"]["id"]
    num_envs = int(cfg["env"].get("num_envs", 4))

    with open("Env/MFG_Musculoskelet_V11/cfg/mujocoenv_default.yaml", "r", encoding="utf8") as f:
        base_cfg = yaml.safe_load(f)
    apply_terrain_config_overrides(base_cfg, cfg.get("terrain"))
    policy_cfg = cfg["train"].get("policy", {})
    terrain_cnn_enabled = bool(policy_cfg.get("terrain_cnn_enabled", False))
    terrain_obs_enabled = bool(policy_cfg.get("terrain_obs_enabled", terrain_cnn_enabled))
    base_cfg["terrain_obs_enabled"] = terrain_obs_enabled
    terrain_policy_classes = {"ActorCritic", "ActorCriticTerrainExo"}
    policy_class_name = policy_cfg.get("class_name")
    if terrain_cnn_enabled and policy_class_name not in terrain_policy_classes:
        raise ValueError(
            "terrain_cnn_enabled=True requires train.policy.class_name to be one of "
            f"{sorted(terrain_policy_classes)}, got {policy_class_name!r}"
        )

    manager = TMProxyManager()
    manager.start()

    tm_args = base_cfg["traj_manager"]
    shared_tm = manager.TrajectoryManager(
        data_path=tm_args["data_path"],
        sample_frequency=float(tm_args["sample_frequency"]),
        speed_range=(float(tm_args["speed_range"][0]), float(tm_args["speed_range"][1])),
        verbose=bool(tm_args["verbose"]),
    )
    
    shm_pack = build_traj_shm_meta(shared_tm)           
    traj_shm_meta = shm_pack["meta"]
    traj_sample_frequency = shm_pack["sample_frequency"]
    
    base_cfg = dict(base_cfg)
    base_cfg.setdefault("default_frame_skip", 1)
    base_cfg["traj_manager"] = None

    envs = make_vec_env(
        env_id=env_id,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
        env_kwargs={
            "config": base_cfg,
            "traj_shm_meta": traj_shm_meta,
            "traj_sample_frequency": traj_sample_frequency,
        },
    )

    env = SB3RslVecEnv(
        envs,
        clip_actions=cfg["env"].get("clip_actions", None),
        device=cfg.get("device", "cpu"),
        use_amp=bool(cfg["train"].get("discriminator", {}).get("use", False)),
    )

    resume_path = cfg.get("resume_path", None)
    exo_checkpoint_path = cfg.get("exo_checkpoint_path", None)
    
    runner = OnPolicyRunner(
        env,
        cfg["train"],
        log_dir=cfg.get("log_dir", "./logs"),
        device=cfg.get("device", "cpu"),
    )
    if resume_path:
        load_optimizer = bool(cfg.get("resume_load_optimizer", False))
        runner.load(resume_path, load_optimizer=load_optimizer)
        if bool(cfg.get("resume_restore_iteration", False)):
            resume_checkpoint, _ = _checkpoint_state_dict(resume_path, "cpu")
            runner.current_learning_iteration = int(resume_checkpoint.get("iter", 0))
            print(
                "Restored learning iteration to "
                f"{runner.current_learning_iteration} from {resume_path}."
            )

    if exo_checkpoint_path:
        load_exo_actor_checkpoint(
            runner.alg.policy,
            exo_checkpoint_path,
            device=cfg.get("device", "cpu"),
            load_noise_std=bool(cfg.get("exo_load_noise_std", True)),
        )
    runner.learn(num_learning_iterations=cfg["train"]["num_learning_iterations"])


if __name__ == "__main__":
    freeze_support()
    hydra_main()
