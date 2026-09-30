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
    
    runner = OnPolicyRunner(
        env,
        cfg["train"],
        log_dir=cfg.get("log_dir", "./logs"),
        device=cfg.get("device", "cpu"),
    )
    runner.load(resume_path, load_optimizer=False)
    runner.learn(num_learning_iterations=cfg["train"]["num_learning_iterations"])


if __name__ == "__main__":
    freeze_support()
    hydra_main()
