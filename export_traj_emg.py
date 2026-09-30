#!/usr/bin/env python3
"""
Export muscle activations (EMG), position-motor controls, and exo-agent outputs
for each reference trajectory.

For every traj_id:
  - reset at frame=0
  - roll out with the trained policy for a fixed number of steps
  - record phase (%), env.data.act (EMG), and applied motor ctrl per step

Saved to: <checkpoint_dir>/traj_emg (default), as traj_<id>.npz with keys:
  - phase        : (T,) float32
  - emg          : (T, 80) float32
  - motor_ctrl   : (T, M) float32
  - motor_indices: (M,) int64
  - motor_names  : (M,) str
  - exo_ctrl     : (T, E) float32
  - exo_indices  : (E,) int64
  - exo_names    : (E,) str
  - terrain_seed : scalar int64 used for this rollout
  - terrain_hash : SHA-256 of the exact terrain heightmap
  - terrain_shape: (2,) int64 hfield shape
  - traj_id, speed
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from RL.modules.actor_critic_latent_test import ActorCritic as ActorCriticLegacy
from RL.modules.actor_critic_latent_exo import ActorCritic as ActorCriticExo
from RL.modules.actor_critic_multiagent import ActorCriticMultiAgent
from RL.modules.actor_critic_terrainexo import ActorCriticTerrainExo
from RL.modules import ActorCritic as ActorCriticCurrent
from RL.modules import EmpiricalNormalization
from RL.utils.utils import AMPNormalizer
from env_wrapper_eval import GymMujocoWrapper
from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager
from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11
from Env.MFG_Musculoskelet_V11.mfgenv.terrain import apply_terrain_config_overrides
from Env.MFG_Musculoskelet_V11.mfgenv.terrain import get_hfield_heightmap_m

torch.serialization.add_safe_globals([AMPNormalizer])


def load_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, device: torch.device):
    with open("Env/MFG_Musculoskelet_V11/cfg/mujocoenv_default.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg = dict(base_cfg)
    apply_terrain_config_overrides(base_cfg, cfg.get("terrain"))
    base_cfg["terrain_obs_enabled"] = bool(
        cfg.get("policy", {}).get("terrain_cnn_enabled", False)
    )
    # EMG export is headless; do not inherit the default human viewer settings.
    base_cfg["render_mode"] = None
    base_cfg["auto_render"] = False
    # Keep older config files working with the newer MuJoCo env constructor.
    base_cfg.setdefault("default_frame_skip", 1)

    tm_args = base_cfg["traj_manager"]
    shared_tm = TrajectoryManager(
        data_path=tm_args["data_path"],
        sample_frequency=float(tm_args["sample_frequency"]),
        speed_range=(float(tm_args["speed_range"][0]), float(tm_args["speed_range"][1])),
        verbose=bool(tm_args["verbose"]),
    )

    base_cfg["traj_manager"] = shared_tm

    env = MFG_Musculoskeletal_V11(
        config=base_cfg,
        render_mode=None,
    )
    env = GymMujocoWrapper(env, auto_reset=False, is_finite_horizon=False, device=device)
    return env


def _infer_sequential_dims(state_dict, prefix):
    linear_layers = []
    for key, tensor in state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        suffix = key[len(prefix) :]
        layer_id = suffix.split(".")[0]
        if not layer_id.isdigit():
            continue
        linear_layers.append((int(layer_id), int(tensor.shape[1]), int(tensor.shape[0])))

    if not linear_layers:
        raise ValueError(f"Cannot infer sequential dims for prefix '{prefix}'.")

    linear_layers.sort(key=lambda x: x[0])
    return {
        "input_dim": linear_layers[0][1],
        "hidden_dims": [out_dim for _, _, out_dim in linear_layers[:-1]],
        "output_dim": linear_layers[-1][2],
    }


def _infer_backbone_head_dims(state_dict, backbone_prefix, head_weight_key):
    backbone_layers = _infer_sequential_dims(state_dict, backbone_prefix)
    if head_weight_key not in state_dict:
        raise ValueError(f"Missing head weight '{head_weight_key}' in checkpoint.")
    return {
        "input_dim": backbone_layers["input_dim"],
        "hidden_dims": backbone_layers["hidden_dims"] + [backbone_layers["output_dim"]],
        "output_dim": int(state_dict[head_weight_key].shape[0]),
    }


def _infer_multi_agent_spec(state_dict, norm_dim, env_num_actions, policy_cfg):
    ma_cfg = dict(policy_cfg or {})
    nested_ma_cfg = ma_cfg.get("multi_agent", {})
    if isinstance(nested_ma_cfg, dict):
        ma_cfg.update(nested_ma_cfg)

    branch_specs = {}
    prefix = "actor_branches."
    for key, tensor in state_dict.items():
        if not key.startswith(prefix):
            continue
        parts = key.split(".")
        if len(parts) < 5 or parts[2] != "net" or parts[4] != "weight":
            continue
        branch_id = int(parts[1])
        layer_id = int(parts[3])
        layer_map = branch_specs.setdefault(branch_id, {})
        layer_map[layer_id] = (int(tensor.shape[1]), int(tensor.shape[0]))

    if len(branch_specs) != 2:
        raise ValueError(
            f"ActorCriticMultiAgent expects 2 actor branches, got {sorted(branch_specs.keys())}"
        )

    agent_obs_dims = []
    agent_action_dims = []
    actor_hidden_dims = None
    for branch_id in sorted(branch_specs.keys()):
        layers = sorted(branch_specs[branch_id].items())
        agent_obs_dims.append(layers[0][1][0])
        agent_action_dims.append(layers[-1][1][1])
        branch_hidden_dims = [shape[1] for _, shape in layers[:-1]]
        if actor_hidden_dims is None:
            actor_hidden_dims = branch_hidden_dims

    num_actions = int(sum(agent_action_dims))
    if num_actions != int(env_num_actions):
        raise ValueError(
            f"Checkpoint action dim {num_actions} does not match env action dim {env_num_actions}."
        )

    if any(k.startswith("critic.") for k in state_dict.keys()):
        critic_spec = _infer_sequential_dims(state_dict, "critic.")
    elif any(k.startswith("critic_backbone.") for k in state_dict.keys()):
        critic_spec = _infer_backbone_head_dims(state_dict, "critic_backbone.", "critic_head.weight")
    else:
        raise ValueError("Cannot infer critic dimensions for multi-agent checkpoint.")

    sum_agent_obs = int(sum(agent_obs_dims))
    shared_actor_obs = bool(ma_cfg.get("shared_actor_obs", sum_agent_obs != int(norm_dim)))
    num_actor_obs = int(norm_dim)
    if not shared_actor_obs and sum_agent_obs != int(norm_dim):
        raise ValueError(
            f"Split multi-agent obs dims {agent_obs_dims} do not sum to obs normalizer dim {norm_dim}."
        )

    human_branch_idx = ma_cfg.get("human_branch_idx")
    if human_branch_idx is None:
        human_branch_idx = int(np.argmax(agent_action_dims))
    human_branch_idx = int(human_branch_idx)
    exo_branch_idx = 1 - human_branch_idx
    exo_action_dim = int(agent_action_dims[exo_branch_idx])

    first256 = False
    exo_obs = False
    if shared_actor_obs:
        exo_input_dim = int(agent_obs_dims[exo_branch_idx])
        first256 = exo_input_dim == 256
        exo_obs = (not first256) and (exo_input_dim != num_actor_obs)

    return {
        "shared_actor_obs": shared_actor_obs,
        "human_branch_idx": human_branch_idx,
        "agent_obs_dims": [int(v) for v in agent_obs_dims],
        "agent_action_dims": [int(v) for v in agent_action_dims],
        "num_actor_obs": num_actor_obs,
        "num_critic_obs": int(critic_spec["input_dim"]),
        "num_actions": num_actions,
        "actor_hidden_dims": actor_hidden_dims
        or ma_cfg.get("actor_hidden_dims", ma_cfg.get("hidden_dims", [256, 256, 256])),
        "critic_hidden_dims": critic_spec["hidden_dims"],
        "exo_action_dim": exo_action_dim,
        "first256": first256,
        "exo_obs": exo_obs,
    }


def _infer_terrain_exo_spec(state_dict, norm_dim, env_num_actions, policy_cfg):
    policy_cfg = dict(policy_cfg or {})
    human_spec = _infer_backbone_head_dims(
        state_dict,
        "human_actor.backbone.",
        "human_actor.action_head.weight",
    )
    exo_spec = _infer_sequential_dims(state_dict, "exo_actor.net.")
    critic_spec = _infer_sequential_dims(state_dict, "critic.")

    human_action_dim = int(human_spec["output_dim"])
    exo_action_dim = int(exo_spec["output_dim"])
    num_actions = human_action_dim + exo_action_dim
    if num_actions != int(env_num_actions):
        raise ValueError(
            f"Checkpoint action dim {num_actions} does not match env action dim "
            f"{env_num_actions}."
        )
    if int(critic_spec["input_dim"]) != int(norm_dim):
        raise ValueError(
            "Terrain-exo critic obs dim does not match obs normalizer dim: "
            f"{critic_spec['input_dim']} vs {norm_dim}"
        )

    terrain_patch_shape = tuple(policy_cfg.get("terrain_patch_shape", [20, 30]))
    terrain_feature_dim = int(policy_cfg.get("terrain_feature_dim", 81))
    terrain_cnn_enabled = any(
        key.startswith("human_actor.terrain_encoder.") for key in state_dict
    )
    terrain_attention_enabled = any(
        key.startswith("human_actor.terrain_encoder.attention_logits.")
        for key in state_dict
    )
    expected_human_mlp_input = int(norm_dim)
    if terrain_cnn_enabled:
        expected_human_mlp_input += (
            terrain_feature_dim - int(np.prod(terrain_patch_shape))
        )
    if int(human_spec["input_dim"]) != expected_human_mlp_input:
        raise ValueError(
            "Terrain-exo human actor input inferred from checkpoint/config does not "
            f"match: {human_spec['input_dim']} vs {expected_human_mlp_input}. "
            "Check terrain_patch_shape and terrain_feature_dim."
        )

    exo_input_dim = int(exo_spec["input_dim"])
    first256 = exo_input_dim == 256
    exo_obs = exo_input_dim not in (int(norm_dim), 256)
    human_branch_idx = int(policy_cfg.get("human_branch_idx", 1))
    if human_branch_idx not in (0, 1):
        raise ValueError(f"human_branch_idx must be 0 or 1, got {human_branch_idx}")
    agent_action_dims = [0, 0]
    agent_action_dims[human_branch_idx] = human_action_dim
    agent_action_dims[1 - human_branch_idx] = exo_action_dim

    return {
        "human_branch_idx": human_branch_idx,
        "agent_action_dims": agent_action_dims,
        "num_actions": num_actions,
        "num_critic_obs": int(critic_spec["input_dim"]),
        "actor_hidden_dims": human_spec["hidden_dims"],
        "critic_hidden_dims": critic_spec["hidden_dims"],
        "exo_actor_hidden_dims": exo_spec["hidden_dims"],
        "exo_action_dim": exo_action_dim,
        "first256": first256,
        "exo_obs": exo_obs,
        "terrain_cnn_enabled": terrain_cnn_enabled,
        "terrain_patch_shape": terrain_patch_shape,
        "terrain_feature_dim": terrain_feature_dim,
        "terrain_attention_enabled": terrain_attention_enabled,
    }


def load_policy(cfg: dict, env, device: torch.device):
    checkpoint_path = cfg["policy"]["checkpoint"]
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model_state = {
        str(k).replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()
    }
    obs_norm_state = checkpoint.get("obs_norm_state_dict")
    if isinstance(obs_norm_state, dict) and "_mean" in obs_norm_state:
        norm_dim = int(obs_norm_state["_mean"].shape[-1])
    else:
        norm_dim = int(env.num_obs)

    activation = cfg["policy"].get("activation", "elu")
    is_terrain_exo_policy = (
        any(k.startswith("human_actor.") for k in model_state)
        and any(k.startswith("exo_actor.") for k in model_state)
    )
    is_multi_agent_policy = any(
        k.startswith("actor_branches.") for k in model_state
    )
    policy_meta = {
        "is_multi_agent_policy": False,
        "shared_actor_obs": False,
        "agent_obs_dims": [],
        "num_actor_obs": norm_dim,
        "exo_action_dim": 0,
    }

    if is_terrain_exo_policy:
        spec = _infer_terrain_exo_spec(
            model_state,
            norm_dim,
            env.num_actions,
            cfg.get("policy", {}),
        )
        policy_meta.update(
            {
                "is_multi_agent_policy": True,
                "shared_actor_obs": True,
                "agent_obs_dims": [norm_dim, norm_dim],
                "num_actor_obs": norm_dim,
                "exo_action_dim": spec["exo_action_dim"],
            }
        )
        policy = ActorCriticTerrainExo(
            num_actor_obs=norm_dim,
            num_critic_obs=spec["num_critic_obs"],
            num_actions=spec["num_actions"],
            actor_hidden_dims=spec["actor_hidden_dims"],
            critic_hidden_dims=spec["critic_hidden_dims"],
            exo_actor_hidden_dims=spec["exo_actor_hidden_dims"],
            activation=activation,
            init_noise_std=float(cfg["policy"].get("init_noise_std", 0.1)),
            noise_std_type=str(cfg["policy"].get("noise_std_type", "scalar")),
            exo_action_dim=spec["exo_action_dim"],
            first256=spec["first256"],
            exo_obs=spec["exo_obs"],
            exo_obs_noise_std=float(cfg["policy"].get("exo_obs_noise_std", 0.0)),
            muscle_action_dim=int(cfg["policy"].get("muscle_action_dim", 80)),
            agent_action_dims=spec["agent_action_dims"],
            shared_actor_obs=True,
            human_branch_idx=spec["human_branch_idx"],
            terrain_cnn_enabled=spec["terrain_cnn_enabled"],
            terrain_patch_shape=spec["terrain_patch_shape"],
            terrain_feature_dim=spec["terrain_feature_dim"],
            terrain_attention_enabled=spec["terrain_attention_enabled"],
            latent_cov_scale=float(cfg["policy"].get("latent_cov_scale", 1.0)),
        ).to(device)
        policy_action_dim = int(spec["num_actions"])
    elif is_multi_agent_policy:
        spec = _infer_multi_agent_spec(model_state, norm_dim, env.num_actions, cfg.get("policy", {}))
        policy_meta.update(
            {
                "is_multi_agent_policy": True,
                "shared_actor_obs": spec["shared_actor_obs"],
                "agent_obs_dims": spec["agent_obs_dims"],
                "num_actor_obs": spec["num_actor_obs"],
                "exo_action_dim": spec["exo_action_dim"],
            }
        )
        policy = ActorCriticMultiAgent(
            num_actor_obs=spec["num_actor_obs"],
            num_critic_obs=spec["num_critic_obs"],
            num_actions=spec["num_actions"],
            actor_hidden_dims=spec["actor_hidden_dims"],
            critic_hidden_dims=spec["critic_hidden_dims"],
            activation=activation,
            init_noise_std=float(cfg["policy"].get("init_noise_std", 0.1)),
            noise_std_type=str(cfg["policy"].get("noise_std_type", "scalar")),
            exo_action_dim=spec["exo_action_dim"],
            first256=spec["first256"],
            exo_obs=spec["exo_obs"],
            exo_obs_noise_std=float(cfg["policy"].get("exo_obs_noise_std", 0.0)),
            muscle_action_dim=int(cfg["policy"].get("muscle_action_dim", 80)),
            agent_obs_dims=spec["agent_obs_dims"] if not spec["shared_actor_obs"] else None,
            agent_action_dims=spec["agent_action_dims"],
            shared_actor_obs=spec["shared_actor_obs"],
            human_branch_idx=spec["human_branch_idx"],
        ).to(device)
        policy_action_dim = int(spec["num_actions"])
    elif ("action_head_main.weight" in model_state) or ("action_head_exo.weight" in model_state):
        actor_spec = _infer_backbone_head_dims(model_state, "backbone.", "action_head_main.weight")
        critic_spec = _infer_sequential_dims(model_state, "critic.")
        exo_dim = None
        if "action_head_exo.weight" in model_state:
            exo_dim = int(model_state["action_head_exo.weight"].shape[0])
        elif "action_head_exo.bias" in model_state:
            exo_dim = int(model_state["action_head_exo.bias"].shape[0])
        if exo_dim is None:
            exo_dim = int(cfg["policy"].get("extra_action_dims", 2))

        policy_meta["num_actor_obs"] = int(actor_spec["input_dim"])
        policy_meta["exo_action_dim"] = int(exo_dim)
        policy = ActorCriticExo(
            num_actions=actor_spec["output_dim"] + exo_dim,
            num_actor_obs=actor_spec["input_dim"],
            num_critic_obs=critic_spec["input_dim"],
            actor_hidden_dims=actor_spec["hidden_dims"],
            critic_hidden_dims=critic_spec["hidden_dims"],
            activation=activation,
            init_noise_std=float(cfg["policy"].get("init_noise_std", 0.1)),
            noise_std_type=str(cfg["policy"].get("noise_std_type", "scalar")),
            extra_action_dims=exo_dim,
        ).to(device)
        policy_action_dim = int(actor_spec["output_dim"] + exo_dim)
    elif any(k.startswith("actor.") for k in model_state.keys()):
        actor_spec = _infer_sequential_dims(model_state, "actor.")
        critic_spec = _infer_sequential_dims(model_state, "critic.")
        policy_meta["num_actor_obs"] = int(actor_spec["input_dim"])
        policy = ActorCriticCurrent(
            num_actions=actor_spec["output_dim"],
            num_actor_obs=actor_spec["input_dim"],
            num_critic_obs=critic_spec["input_dim"],
            actor_hidden_dims=actor_spec["hidden_dims"],
            critic_hidden_dims=critic_spec["hidden_dims"],
            activation=activation,
            init_noise_std=float(cfg["policy"].get("init_noise_std", 0.1)),
            noise_std_type=str(cfg["policy"].get("noise_std_type", "scalar")),
        ).to(device)
        policy_action_dim = int(actor_spec["output_dim"])
    else:
        actor_spec = _infer_backbone_head_dims(model_state, "backbone.", "action_head.weight")
        critic_spec = _infer_sequential_dims(model_state, "critic.")
        terrain_cnn_enabled = bool(cfg["policy"].get("terrain_cnn_enabled", False))
        terrain_patch_shape = tuple(cfg["policy"].get("terrain_patch_shape", [9, 9]))
        terrain_feature_dim = int(cfg["policy"].get("terrain_feature_dim", 64))
        terrain_attention_enabled = bool(
            cfg["policy"].get("terrain_attention_enabled", False)
        )
        terrain_obs_dim = int(terrain_patch_shape[0] * terrain_patch_shape[1])
        if terrain_cnn_enabled:
            policy_meta["num_actor_obs"] = (
                int(actor_spec["input_dim"]) - terrain_feature_dim + terrain_obs_dim
            )
        else:
            policy_meta["num_actor_obs"] = int(actor_spec["input_dim"])
        policy = ActorCriticLegacy(
            num_actions=actor_spec["output_dim"],
            num_actor_obs=policy_meta["num_actor_obs"],
            num_critic_obs=critic_spec["input_dim"],
            actor_hidden_dims=actor_spec["hidden_dims"],
            critic_hidden_dims=critic_spec["hidden_dims"],
            activation=activation,
            init_noise_std=float(cfg["policy"].get("init_noise_std", 0.1)),
            noise_std_type=str(cfg["policy"].get("noise_std_type", "scalar")),
            terrain_cnn_enabled=terrain_cnn_enabled,
            terrain_patch_shape=terrain_patch_shape,
            terrain_feature_dim=terrain_feature_dim,
            terrain_attention_enabled=terrain_attention_enabled,
        ).to(device)
        policy_action_dim = int(actor_spec["output_dim"])

    env_action_dim = int(env.num_actions)
    action_padding_dim = env_action_dim - policy_action_dim
    if action_padding_dim < 0:
        raise ValueError(
            f"Checkpoint action dim {policy_action_dim} exceeds env action dim {env_action_dim}."
        )
    if action_padding_dim > 0:
        export_cfg = dict(cfg.get("export", {}) or {})
        allow_padding = bool(export_cfg.get("pad_missing_exo_actions", True))
        base_env = env.envs[0].unwrapped
        padded_names = list(getattr(base_env, "actuator_names", []))[-action_padding_dim:]
        looks_like_exo = all(
            ("exo" in str(name).lower()) or ("ankle_motor" in str(name).lower())
            for name in padded_names
        )
        if not allow_padding or not looks_like_exo:
            raise ValueError(
                f"Checkpoint has {policy_action_dim} actions but env has {env_action_dim}; "
                f"cannot safely pad trailing actuators {padded_names}."
            )
        if int(policy_meta["exo_action_dim"]) == 0:
            policy_meta["exo_action_dim"] = action_padding_dim
        print(
            f"Checkpoint has no controls for trailing exo actuators {padded_names}; "
            "export will append zero actions."
        )
    policy_meta["policy_action_dim"] = policy_action_dim
    policy_meta["env_action_dim"] = env_action_dim
    policy_meta["action_padding_dim"] = action_padding_dim

    if policy_meta["num_actor_obs"] != norm_dim:
        if isinstance(obs_norm_state, dict) and "_mean" in obs_norm_state:
            raise ValueError(
                "Checkpoint actor obs dim does not match obs normalizer dim: "
                f"{policy_meta['num_actor_obs']} vs {norm_dim}"
            )
        obs_norm = EmpiricalNormalization(shape=[policy_meta["num_actor_obs"]], until=1.0e8).to(device)
    else:
        obs_norm = EmpiricalNormalization(shape=[norm_dim], until=1.0e8).to(device)

    if isinstance(obs_norm_state, dict):
        obs_norm.load_state_dict(obs_norm_state)
    else:
        print("Checkpoint has no obs_norm_state_dict; using fresh normalizer.")
    obs_norm.eval()

    policy.load_state_dict(model_state)
    policy.eval()
    print(f"Loaded policy from {checkpoint_path}")
    return policy, obs_norm, policy_meta


def _compose_policy_obs(obs_tensor, info, policy_meta, base_env):
    if not policy_meta["is_multi_agent_policy"]:
        return obs_tensor

    if int(obs_tensor.shape[-1]) == int(policy_meta["num_actor_obs"]):
        return obs_tensor
    if policy_meta["shared_actor_obs"]:
        return obs_tensor

    obs_np = obs_tensor.detach().cpu().numpy().reshape(obs_tensor.shape[0], -1)
    if obs_np.shape[0] != 1:
        raise ValueError("EMG export currently supports single-env rollout only.")
    global_obs = obs_np[0]

    obs_by_agent = None
    if isinstance(info, dict):
        raw = info.get("obs_by_agent")
        if isinstance(raw, dict):
            obs_by_agent = raw
    if obs_by_agent is None:
        builder = getattr(base_env, "_build_multi_agent_obs", None)
        if callable(builder):
            obs_by_agent = builder(global_obs)
    if not isinstance(obs_by_agent, dict):
        raise ValueError(
            "Cannot obtain per-agent observations for multi-agent policy inference "
            f"(obs dim={obs_tensor.shape[-1]}, expected={policy_meta['num_actor_obs']})."
        )

    chunks = []
    keys = list(obs_by_agent.keys())
    if len(keys) != len(policy_meta["agent_obs_dims"]):
        raise ValueError(
            f"obs_by_agent count {len(keys)} does not match checkpoint branches "
            f"{len(policy_meta['agent_obs_dims'])}."
        )

    for i, key in enumerate(keys):
        arr = np.asarray(obs_by_agent[key], dtype=np.float32).reshape(-1)
        expected = int(policy_meta["agent_obs_dims"][i])
        if arr.size != expected:
            raise ValueError(
                f"obs_by_agent['{key}'] dim mismatch: got {arr.size}, expected {expected}"
            )
        chunks.append(arr)

    cat_obs = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(cat_obs).to(obs_tensor.device).unsqueeze(0)


def _get_motor_metadata(base_env):
    motor_indices = np.asarray(
        getattr(base_env, "position_indices", np.empty(0, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    actuator_names = list(getattr(base_env, "actuator_names", []))
    motor_names = []
    for idx in motor_indices:
        if 0 <= int(idx) < len(actuator_names):
            motor_names.append(str(actuator_names[int(idx)]))
        else:
            motor_names.append(f"actuator_{int(idx)}")
    return motor_indices, np.asarray(motor_names, dtype=str)


def _compute_applied_ctrl(actions: torch.Tensor, env, base_env) -> np.ndarray:
    ctrl = np.asarray(actions.detach().cpu().numpy(), dtype=np.float32).reshape(-1)

    clip_actions = getattr(env, "clip_actions", None)
    if clip_actions is not None:
        np.clip(ctrl, -float(clip_actions), float(clip_actions), out=ctrl)

    if getattr(base_env, "_do_clip_action", False):
        ctrl, _ = base_env._clip_action(ctrl)

    return np.asarray(ctrl, dtype=np.float32)


def _get_exo_metadata(base_env, policy_meta):
    exo_action_dim = int(policy_meta.get("exo_action_dim", 0))
    if exo_action_dim <= 0:
        return np.empty(0, dtype=np.int64), np.asarray([], dtype=str)

    actuator_names = list(getattr(base_env, "actuator_names", []))
    num_actuators = len(actuator_names)
    if exo_action_dim > num_actuators:
        raise ValueError(
            f"exo_action_dim {exo_action_dim} exceeds actuator count {num_actuators}."
        )

    exo_indices = np.arange(num_actuators - exo_action_dim, num_actuators, dtype=np.int64)
    exo_names = np.asarray([str(actuator_names[int(idx)]) for idx in exo_indices], dtype=str)
    return exo_indices, exo_names


def _align_actions_to_env(actions: torch.Tensor, policy_meta: dict) -> torch.Tensor:
    padding_dim = int(policy_meta.get("action_padding_dim", 0))
    if padding_dim <= 0:
        return actions
    padding = torch.zeros(
        *actions.shape[:-1],
        padding_dim,
        dtype=actions.dtype,
        device=actions.device,
    )
    return torch.cat((actions, padding), dim=-1)


def _terrain_seed_for_traj(export_cfg: dict, traj_id: int) -> int | None:
    base_seed = export_cfg.get("terrain_seed")
    if base_seed is None:
        return None

    base_seed = int(base_seed)
    mode = str(export_cfg.get("terrain_seed_mode", "per_trajectory")).strip().lower()
    if mode == "shared":
        return base_seed
    if mode == "per_trajectory":
        # SeedSequence avoids accidental overflow and gives stable seeds across processes.
        return int(np.random.SeedSequence([base_seed, int(traj_id)]).generate_state(1)[0])
    raise ValueError(
        f"Unsupported export.terrain_seed_mode={mode!r}; "
        "expected 'shared' or 'per_trajectory'."
    )


def _terrain_metadata(base_env, terrain_seed: int | None) -> tuple[dict, np.ndarray]:
    if int(getattr(base_env.model, "nhfield", 0)) <= 0:
        return (
            {
                "terrain_seed": np.int64(-1 if terrain_seed is None else terrain_seed),
                "terrain_hash": np.asarray("", dtype=str),
                "terrain_shape": np.asarray([0, 0], dtype=np.int64),
            },
            np.empty((0, 0), dtype=np.float32),
        )

    heightmap_m = get_hfield_heightmap_m(
        base_env.model,
        hfield_name=base_env.terrain_hfield_name,
    )
    canonical_heightmap = np.ascontiguousarray(heightmap_m, dtype="<f4")
    digest = hashlib.sha256(canonical_heightmap.tobytes()).hexdigest()
    metadata = {
        "terrain_seed": np.int64(-1 if terrain_seed is None else terrain_seed),
        "terrain_hash": np.asarray(digest, dtype=str),
        "terrain_shape": np.asarray(canonical_heightmap.shape, dtype=np.int64),
    }
    return metadata, canonical_heightmap


def rollout_emg(
    env,
    policy,
    obs_norm,
    policy_meta,
    traj_id: int,
    terrain_seed: int | None = None,
    save_terrain_heightmap: bool = False,
):
    obs_raw, info = env.reset(id=traj_id, frame=0, terrain_seed=terrain_seed)

    base_env = env.envs[0].unwrapped
    terrain_metadata, terrain_heightmap_m = _terrain_metadata(base_env, terrain_seed)
    obs = obs_norm(_compose_policy_obs(obs_raw, info, policy_meta, base_env))
    motor_indices, motor_names = _get_motor_metadata(base_env)
    exo_indices, exo_names = _get_exo_metadata(base_env, policy_meta)

    phases, emgs, motor_ctrls, exo_ctrls = [], [], [], []
    done = False
    while not done:
        phases.append(float(base_env.ref_traj.phase))
        emgs.append(np.asarray(base_env.data.act, dtype=np.float32).copy())

        with torch.no_grad():
            actions = policy.act_inference(obs)
            actions = _align_actions_to_env(actions, policy_meta)
        applied_ctrl = _compute_applied_ctrl(actions[0], env, base_env)
        motor_ctrls.append(applied_ctrl[motor_indices].copy())
        exo_ctrls.append(applied_ctrl[exo_indices].copy())
        obs_raw, _, done, info = env.step(actions)
        done = bool(done.item()) if torch.is_tensor(done) else bool(done)
        obs = obs_norm(_compose_policy_obs(obs_raw, info, policy_meta, base_env))

    result = {
        "phase": np.asarray(phases, dtype=np.float32),
        "emg": np.stack(emgs).astype(np.float32),
        "motor_ctrl": np.stack(motor_ctrls).astype(np.float32),
        "motor_indices": motor_indices,
        "motor_names": motor_names,
        "exo_ctrl": np.stack(exo_ctrls).astype(np.float32),
        "exo_indices": exo_indices,
        "exo_names": exo_names,
        "traj_id": traj_id,
        "speed": float(base_env.ref_traj.speed),
        **terrain_metadata,
    }
    if save_terrain_heightmap:
        result["terrain_heightmap_m"] = terrain_heightmap_m
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Export EMG (muscle activations) and motor controls per trajectory."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Path to YAML config (used for checkpoint/env paths).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to store traj_<id>.npz. Defaults to <checkpoint_dir>/traj_emg.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override policy.checkpoint while keeping the same terrain/export config.",
    )
    parser.add_argument(
        "--terrain_seed",
        type=int,
        default=None,
        help="Override export.terrain_seed for reproducible terrain comparison.",
    )
    parser.add_argument(
        "--terrain_seed_mode",
        choices=("shared", "per_trajectory"),
        default=None,
        help="Use one terrain for all trajectories, or a stable terrain per traj_id.",
    )
    args = parser.parse_args()

    cfg = load_cfg(Path(args.config))
    if args.checkpoint is not None:
        cfg["policy"] = dict(cfg.get("policy", {}) or {})
        cfg["policy"]["checkpoint"] = args.checkpoint
    export_cfg = dict(cfg.get("export", {}) or {})
    if args.terrain_seed is not None:
        export_cfg["terrain_seed"] = int(args.terrain_seed)
    if args.terrain_seed_mode is not None:
        export_cfg["terrain_seed_mode"] = args.terrain_seed_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = build_env(cfg, device)
    policy, obs_norm, policy_meta = load_policy(cfg, env, device)

    checkpoint_dir = os.path.dirname(cfg["policy"]["checkpoint"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(checkpoint_dir) / "traj_emg_best"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_total = len(env.envs[0].unwrapped.traj_manager)
    print(f"Found {traj_total} trajectories. Saving EMG and motor controls to {out_dir}")

    for traj_id in range(traj_total):
        terrain_seed = _terrain_seed_for_traj(export_cfg, traj_id)
        data = rollout_emg(
            env,
            policy,
            obs_norm,
            policy_meta,
            traj_id,
            terrain_seed=terrain_seed,
            save_terrain_heightmap=bool(export_cfg.get("save_terrain_heightmap", False)),
        )
        out_path = out_dir / f"traj_{traj_id}.npz"
        np.savez_compressed(out_path, **data)
        print(
            f"Saved {out_path} "
            f"(terrain_seed={int(data['terrain_seed'])}, "
            f"terrain_hash={str(data['terrain_hash'])[:12]})"
        )

    env.close()


if __name__ == "__main__":
    main()
