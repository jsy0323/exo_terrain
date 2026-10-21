#!/usr/bin/env python3
import os
import time
import pandas as pd
import wandb
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from RL.modules.actor_critic_latent_test import ActorCritic as ActorCriticLegacy
from RL.modules.actor_critic_latent_exo import ActorCritic as ActorCriticExo
from RL.modules.actor_critic_multiagent import ActorCriticMultiAgent
from RL.modules.actor_critic_terrainexo import ActorCriticTerrainExo
from RL.modules import ActorCritic as ActorCriticCurrent
from RL.modules import EmpiricalNormalization
from env_wrapper_eval import GymMujocoWrapper
from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager
from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11
from Env.MFG_Musculoskelet_V11.mfgenv.reward import get_imitation_reward
from Env.MFG_Musculoskelet_V11.mfgenv.terrain import apply_terrain_config_overrides
from dataclasses import dataclass
from RL.utils.utils import AMPNormalizer
torch.serialization.add_safe_globals([AMPNormalizer])

class PolicyVisualizer:
    IMITATION_ERROR_KEYS = ("site_pos_err", "site_vel_err", "joint_angle_err")
    ENERGY_ERROR_KEYS = ("energy_cost", "activation_l1", "activation_l2_sq", "metab_cost")

    def __init__(self, cfg_path):
        self.load_config(cfg_path)
        self.setup_environment()
        self.num_obs = self.env.num_obs
        self.num_envs = int(self.env.num_envs)
        self.is_multi_agent_policy = False
        self.shared_actor_obs = False
        self.human_branch_idx = None
        self.exo_branch_idx = None
        self.agent_obs_dims = []
        self.agent_action_dims = []
        self.num_actor_obs = self.num_obs
        self.obs_normalizer = None
        self.base_dir = os.path.dirname(self.cfg["policy"]["checkpoint"])
        self.txt_path = os.path.join(self.base_dir, self.cfg["visualization"]["save_name"])
        self.load_policy()

    def load_config(self, cfg_path):
        """Load visualization configuration"""
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self):
        with open("Env/MFG_Musculoskelet_V11/cfg/mujocoenv_default.yaml", "r",encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg = dict(base_cfg)
        if self.cfg.get("env", {}).get("model_path"):
            base_cfg["model_path"] = self.cfg["env"]["model_path"]
        apply_terrain_config_overrides(base_cfg, self.cfg.get("terrain"))
        terrain_cnn_enabled = bool(self.cfg.get("policy", {}).get("terrain_cnn_enabled", False))
        base_cfg["terrain_obs_enabled"] = terrain_cnn_enabled
        # New MuJoCoEnv requires this key at construction time. The MS env
        # recomputes the effective frame_skip from the trajectory after reset,
        # so a safe placeholder keeps older config files working.
        base_cfg.setdefault("default_frame_skip", 1)
            
        tm_args = base_cfg["traj_manager"]    
        shared_tm = TrajectoryManager(
            data_path=tm_args["data_path"],
            sample_frequency=float(tm_args["sample_frequency"]),
            speed_range=(float(tm_args["speed_range"][0]), float(tm_args["speed_range"][1])),
            verbose=bool(tm_args["verbose"])
        )
        self.num_traj = shared_tm.length()

        base_cfg["traj_manager"] = shared_tm
        
        self.env = MFG_Musculoskeletal_V11(
            config=base_cfg,
            render_mode="human" if self.cfg["env"]["render_mode"] == "human" else None
        )
        # import ipdb;ipdb.set_trace()
        self.env = GymMujocoWrapper(
            self.env,
            device=self.device,
            is_finite_horizon=False
        )
        self.base_env = self.env.envs[0]

    def load_policy(self):
        """Load trained policy network"""
        if torch.cuda.is_available():
            checkpoint = torch.load(self.cfg["policy"]["checkpoint"], weights_only=False)
        else:
            checkpoint = torch.load(self.cfg["policy"]["checkpoint"],map_location=torch.device('cpu'))

        model_state = {
            str(k).replace("module.", ""): v
            for k, v in checkpoint["model_state_dict"].items()
        }
        obs_norm_state = checkpoint.get("obs_norm_state_dict", None)
        if isinstance(obs_norm_state, dict) and "_mean" in obs_norm_state:
            norm_dim = int(obs_norm_state["_mean"].shape[-1])
        else:
            norm_dim = int(self.env.num_obs)
        self.num_actor_obs = norm_dim
        self.obs_normalizer = EmpiricalNormalization(shape=[norm_dim], until=1.0e8).to(self.device)

        activation = self.cfg["policy"].get("activation", "elu")
        is_terrain_exo_policy = (
            any(k.startswith("human_actor.") for k in model_state)
            and any(k.startswith("exo_actor.") for k in model_state)
        )
        self.is_multi_agent_policy = (
            is_terrain_exo_policy
            or any(k.startswith("actor_branches.") for k in model_state)
        )

        if is_terrain_exo_policy:
            spec = self._infer_terrain_exo_spec(model_state, norm_dim)
            self.shared_actor_obs = True
            self.human_branch_idx = spec["human_branch_idx"]
            self.exo_branch_idx = 1 - self.human_branch_idx
            self.agent_obs_dims = [norm_dim, norm_dim]
            self.agent_action_dims = spec["agent_action_dims"]
            self.num_actor_obs = norm_dim

            self.policy = ActorCriticTerrainExo(
                num_actor_obs=norm_dim,
                num_critic_obs=spec["num_critic_obs"],
                num_actions=spec["num_actions"],
                actor_hidden_dims=spec["actor_hidden_dims"],
                critic_hidden_dims=spec["critic_hidden_dims"],
                exo_actor_hidden_dims=spec["exo_actor_hidden_dims"],
                activation=activation,
                init_noise_std=float(self.cfg["policy"].get("init_noise_std", 0.1)),
                noise_std_type=str(self.cfg["policy"].get("noise_std_type", "scalar")),
                exo_action_dim=spec["exo_action_dim"],
                first256=spec["first256"],
                exo_obs=spec["exo_obs"],
                exo_obs_noise_std=float(self.cfg["policy"].get("exo_obs_noise_std", 0.0)),
                muscle_action_dim=int(self.cfg["policy"].get("muscle_action_dim", 80)),
                agent_action_dims=spec["agent_action_dims"],
                shared_actor_obs=True,
                human_branch_idx=spec["human_branch_idx"],
                terrain_cnn_enabled=spec["terrain_cnn_enabled"],
                terrain_patch_shape=spec["terrain_patch_shape"],
                terrain_feature_dim=spec["terrain_feature_dim"],
                terrain_attention_enabled=spec["terrain_attention_enabled"],
                latent_cov_scale=float(self.cfg["policy"].get("latent_cov_scale", 1.0)),
            ).to(self.device)
        elif self.is_multi_agent_policy:
            spec = self._infer_multi_agent_spec(model_state, norm_dim)
            self.shared_actor_obs = spec["shared_actor_obs"]
            self.human_branch_idx = spec["human_branch_idx"]
            self.exo_branch_idx = 1 - self.human_branch_idx
            self.agent_obs_dims = spec["agent_obs_dims"]
            self.agent_action_dims = spec["agent_action_dims"]
            self.num_actor_obs = spec["num_actor_obs"]

            self.policy = ActorCriticMultiAgent(
                num_actor_obs=spec["num_actor_obs"],
                num_critic_obs=spec["num_critic_obs"],
                num_actions=spec["num_actions"],
                actor_hidden_dims=spec["actor_hidden_dims"],
                critic_hidden_dims=spec["critic_hidden_dims"],
                activation=activation,
                init_noise_std=float(self.cfg["policy"].get("init_noise_std", 0.1)),
                noise_std_type=str(self.cfg["policy"].get("noise_std_type", "scalar")),
                exo_action_dim=spec["exo_action_dim"],
                first256=spec["first256"],
                exo_obs=spec["exo_obs"],
                exo_obs_noise_std=float(self.cfg["policy"].get("exo_obs_noise_std", 0.0)),
                agent_obs_dims=spec["agent_obs_dims"] if not spec["shared_actor_obs"] else None,
                agent_action_dims=spec["agent_action_dims"],
                shared_actor_obs=spec["shared_actor_obs"],
                human_branch_idx=spec["human_branch_idx"],
            ).to(self.device)
        elif ("action_head_main.weight" in model_state) or ("action_head_exo.weight" in model_state):
            actor_spec = self._infer_backbone_head_dims(model_state, "backbone.", "action_head_main.weight")
            critic_spec = self._infer_sequential_dims(model_state, "critic.")
            self.num_actor_obs = actor_spec["input_dim"]
            exo_dim = None
            if "action_head_exo.weight" in model_state:
                exo_dim = int(model_state["action_head_exo.weight"].shape[0])
            elif "action_head_exo.bias" in model_state:
                exo_dim = int(model_state["action_head_exo.bias"].shape[0])
            if exo_dim is None:
                exo_dim = int(self.cfg["policy"].get("extra_action_dims", 2))

            self.policy = ActorCriticExo(
                num_actions=actor_spec["output_dim"] + exo_dim,
                num_actor_obs=actor_spec["input_dim"],
                num_critic_obs=critic_spec["input_dim"],
                actor_hidden_dims=actor_spec["hidden_dims"],
                critic_hidden_dims=critic_spec["hidden_dims"],
                activation=activation,
                init_noise_std=float(self.cfg["policy"].get("init_noise_std", 0.1)),
                noise_std_type=str(self.cfg["policy"].get("noise_std_type", "scalar")),
                extra_action_dims=exo_dim,
            ).to(self.device)
        elif any(k.startswith("actor.") for k in model_state.keys()):
            actor_spec = self._infer_sequential_dims(model_state, "actor.")
            critic_spec = self._infer_sequential_dims(model_state, "critic.")
            self.num_actor_obs = actor_spec["input_dim"]
            self.policy = ActorCriticCurrent(
                num_actions=actor_spec["output_dim"],
                num_actor_obs=actor_spec["input_dim"],
                num_critic_obs=critic_spec["input_dim"],
                actor_hidden_dims=actor_spec["hidden_dims"],
                critic_hidden_dims=critic_spec["hidden_dims"],
                activation=activation,
                init_noise_std=float(self.cfg["policy"].get("init_noise_std", 0.1)),
                noise_std_type=str(self.cfg["policy"].get("noise_std_type", "scalar")),
            ).to(self.device)
        else:
            actor_spec = self._infer_backbone_head_dims(model_state, "backbone.", "action_head.weight")
            critic_spec = self._infer_sequential_dims(model_state, "critic.")
            terrain_cnn_enabled = bool(self.cfg["policy"].get("terrain_cnn_enabled", False))
            terrain_patch_shape = tuple(self.cfg["policy"].get("terrain_patch_shape", [9, 9]))
            terrain_feature_dim = int(self.cfg["policy"].get("terrain_feature_dim", 64))
            terrain_attention_enabled = bool(self.cfg["policy"].get("terrain_attention_enabled", False))
            terrain_obs_dim = int(terrain_patch_shape[0] * terrain_patch_shape[1])
            if terrain_cnn_enabled:
                self.num_actor_obs = actor_spec["input_dim"] - terrain_feature_dim + terrain_obs_dim
            else:
                self.num_actor_obs = actor_spec["input_dim"]
            self.policy = ActorCriticLegacy(
                num_actions=actor_spec["output_dim"],
                num_actor_obs=self.num_actor_obs,
                num_critic_obs=critic_spec["input_dim"],
                actor_hidden_dims=actor_spec["hidden_dims"],
                critic_hidden_dims=critic_spec["hidden_dims"],
                activation=activation,
                init_noise_std=float(self.cfg["policy"].get("init_noise_std", 0.1)),
                noise_std_type=str(self.cfg["policy"].get("noise_std_type", "scalar")),
                terrain_cnn_enabled=terrain_cnn_enabled,
                terrain_patch_shape=terrain_patch_shape,
                terrain_feature_dim=terrain_feature_dim,
                terrain_attention_enabled=terrain_attention_enabled,
            ).to(self.device)

        if self.num_actor_obs != norm_dim:
            if isinstance(obs_norm_state, dict) and "_mean" in obs_norm_state:
                raise ValueError(
                    "Checkpoint actor obs dim does not match obs normalizer dim: "
                    f"{self.num_actor_obs} vs {norm_dim}"
                )
            self.obs_normalizer = EmpiricalNormalization(shape=[self.num_actor_obs], until=1.0e8).to(self.device)

        if isinstance(obs_norm_state, dict):
            self.obs_normalizer.load_state_dict(obs_norm_state)
        else:
            print("Checkpoint has no obs_norm_state_dict; using fresh normalizer.")
        self.obs_normalizer.eval()
        self.policy.load_state_dict(model_state)
        self.policy.eval()
        print(f"Loaded policy from {self.cfg['policy']['checkpoint']}")

    def _infer_sequential_dims(self, state_dict, prefix):
        linear_layers = []
        for key, tensor in state_dict.items():
            if not key.startswith(prefix) or not key.endswith(".weight"):
                continue
            suffix = key[len(prefix):]
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

    def _infer_backbone_head_dims(self, state_dict, backbone_prefix, head_weight_key):
        backbone_layers = self._infer_sequential_dims(state_dict, backbone_prefix)
        if head_weight_key not in state_dict:
            raise ValueError(f"Missing head weight '{head_weight_key}' in checkpoint.")
        return {
            "input_dim": backbone_layers["input_dim"],
            "hidden_dims": backbone_layers["hidden_dims"] + [backbone_layers["output_dim"]],
            "output_dim": int(state_dict[head_weight_key].shape[0]),
        }

    def _infer_terrain_exo_spec(self, state_dict, norm_dim):
        policy_cfg = self.cfg.get("policy", {})
        human_spec = self._infer_backbone_head_dims(
            state_dict,
            "human_actor.backbone.",
            "human_actor.action_head.weight",
        )
        exo_spec = self._infer_sequential_dims(state_dict, "exo_actor.net.")
        critic_spec = self._infer_sequential_dims(state_dict, "critic.")

        human_action_dim = int(human_spec["output_dim"])
        exo_action_dim = int(exo_spec["output_dim"])
        num_actions = human_action_dim + exo_action_dim
        if num_actions != int(self.env.num_actions):
            raise ValueError(
                f"Checkpoint action dim {num_actions} does not match env action dim "
                f"{self.env.num_actions}."
            )
        # Actor normalization describes policy obs only. A privileged critic
        # may have additional inputs, which are unused during inference.
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

    def _infer_multi_agent_spec(self, state_dict, norm_dim):
        ma_cfg = self.cfg.get("policy", {})
        if not isinstance(ma_cfg, dict):
            ma_cfg = {}
        else:
            ma_cfg = dict(ma_cfg)
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
        if num_actions != int(self.env.num_actions):
            raise ValueError(
                f"Checkpoint action dim {num_actions} does not match env action dim {self.env.num_actions}."
            )

        if any(k.startswith("critic.") for k in state_dict.keys()):
            critic_spec = self._infer_sequential_dims(state_dict, "critic.")
        elif any(k.startswith("critic_backbone.") for k in state_dict.keys()):
            critic_spec = self._infer_backbone_head_dims(
                state_dict, "critic_backbone.", "critic_head.weight"
            )
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
            muscle_obs_dim = int(ma_cfg.get("muscle_action_dim", 80))
            base_obs_dim = int(norm_dim) - ActorCriticMultiAgent.EXO_PHASE_DIM - muscle_obs_dim
            joint_dim = (
                base_obs_dim
                - ActorCriticMultiAgent.EXO_BASE_JOINT_START
                - ActorCriticMultiAgent.EXO_FUTURE_REF_DIM
            )
            inferred_exo_obs_dim = (
                ActorCriticMultiAgent.EXO_SELECTED_SITE_DIM + joint_dim
                if joint_dim > 0
                else None
            )
            exo_obs = (exo_input_dim == 140) or (exo_input_dim == inferred_exo_obs_dim)

        return {
            "shared_actor_obs": shared_actor_obs,
            "human_branch_idx": human_branch_idx,
            "agent_obs_dims": [int(v) for v in agent_obs_dims],
            "agent_action_dims": [int(v) for v in agent_action_dims],
            "num_actor_obs": num_actor_obs,
            "num_critic_obs": int(critic_spec["input_dim"]),
            "num_actions": num_actions,
            "actor_hidden_dims": actor_hidden_dims or ma_cfg.get(
                "actor_hidden_dims",
                ma_cfg.get("hidden_dims", [256, 256, 256]),
            ),
            "critic_hidden_dims": critic_spec["hidden_dims"],
            "exo_action_dim": exo_action_dim,
            "first256": first256,
            "exo_obs": exo_obs,
        }

    def _compose_policy_obs(self, obs_tensor, info=None):
        if not self.is_multi_agent_policy:
            return obs_tensor

        if int(obs_tensor.shape[-1]) == int(self.num_actor_obs):
            return obs_tensor
        if self.shared_actor_obs:
            return obs_tensor

        obs_np = obs_tensor.detach().cpu().numpy().reshape(self.num_envs, -1)
        if self.num_envs != 1:
            raise ValueError("Visualizer currently supports single-env playback only.")
        global_obs = obs_np[0]

        obs_by_agent = None
        if isinstance(info, dict):
            raw = info.get("obs_by_agent", None)
            if isinstance(raw, dict):
                obs_by_agent = raw
        if obs_by_agent is None:
            builder = getattr(self.base_env, "_build_multi_agent_obs", None)
            if callable(builder):
                obs_by_agent = builder(global_obs)
        if not isinstance(obs_by_agent, dict):
            raise ValueError(
                "Cannot obtain per-agent observations for multi-agent policy inference "
                f"(obs dim={obs_tensor.shape[-1]}, expected={self.num_actor_obs})."
            )

        chunks = []
        keys = list(obs_by_agent.keys())
        if len(keys) != len(self.agent_obs_dims):
            raise ValueError(
                f"obs_by_agent count {len(keys)} does not match checkpoint branches {len(self.agent_obs_dims)}."
            )

        for i, key in enumerate(keys):
            arr = np.asarray(obs_by_agent[key], dtype=np.float32).reshape(-1)
            expected = int(self.agent_obs_dims[i])
            if arr.size != expected:
                raise ValueError(
                    f"obs_by_agent['{key}'] dim mismatch: got {arr.size}, expected {expected}"
                )
            chunks.append(arr)

        cat_obs = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        return torch.from_numpy(cat_obs).to(obs_tensor.device).unsqueeze(0)

    @staticmethod
    def _is_optional_exo_actuator(name):
        name = str(name).lower()
        return name.startswith("exo_") or name.startswith("ankle_motor_")

    def _align_actions_to_env(self, actions):
        """Pad omitted trailing exoskeleton controls with neutral zeros."""
        expected = int(self.env.num_actions)
        got = int(actions.shape[-1])
        if got == expected:
            return actions

        actuator_names = list(getattr(self.base_env, "actuator_names", ()))
        missing_names = actuator_names[got:] if len(actuator_names) == expected else []
        can_pad_optional_exo = (
            got < expected
            and len(missing_names) == expected - got
            and all(self._is_optional_exo_actuator(name) for name in missing_names)
        )
        if not can_pad_optional_exo:
            raise ValueError(
                f"Policy action size mismatch: checkpoint produced {got}, environment expects "
                f"{expected}. Environment actuator tail={actuator_names[max(got, 0):]}."
            )

        if torch.is_tensor(actions):
            padding = actions.new_zeros((*actions.shape[:-1], expected - got))
            aligned = torch.cat((actions, padding), dim=-1)
        else:
            actions_np = np.asarray(actions)
            padding = np.zeros((*actions_np.shape[:-1], expected - got), dtype=actions_np.dtype)
            aligned = np.concatenate((actions_np, padding), axis=-1)

        if not getattr(self, "_printed_action_padding_warning", False):
            print(
                f"Policy outputs {got} actions; padding neutral controls for trailing "
                f"exoskeleton actuators: {missing_names}"
            )
            self._printed_action_padding_warning = True
        return aligned

    def _to_scalar(self, v):
        if torch.is_tensor(v):
            if v.numel() == 1:
                return float(v.item())
            return float(v.mean().item())
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return float(v.item())
            return float(v.mean())
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _compute_current_energy_metrics(self):
        """Compute energy diagnostics directly when reward breakdown is unavailable."""
        if not bool(getattr(self.base_env, "_has_muscle_actuators", False)):
            return {
                "energy_cost": 0.0,
                "activation_l1": 0.0,
                "activation_l2_sq": 0.0,
                "metab_cost": 0.0,
            }

        data = getattr(self.base_env, "data", None)
        act = getattr(data, "act", None)
        if act is None:
            return {}
        act = np.asarray(act, dtype=np.float64).reshape(-1)
        if act.size == 0:
            return {
                "energy_cost": 0.0,
                "activation_l1": 0.0,
                "activation_l2_sq": 0.0,
                "metab_cost": 0.0,
            }

        activation_l1 = float(np.sum(np.abs(act)))
        activation_l2_sq = float(np.dot(act, act))
        energy_cost = 1.0e-3 * (0.5 * activation_l1 + activation_l2_sq)

        metab_cost = 0.0
        volumes = getattr(self.base_env, "muscle_volumes", None)
        if isinstance(volumes, np.ndarray) and volumes.size >= act.size:
            metab_cost = float(np.dot(volumes[:act.size], act * act))

        return {
            "energy_cost": energy_cost,
            "activation_l1": activation_l1,
            "activation_l2_sq": activation_l2_sq,
            "metab_cost": metab_cost,
        }

    def _extract_error_metrics(self, info):
        """Extract raw imitation MSEs and energy metrics for reporting."""
        info = info if isinstance(info, dict) else {}
        reward_info = info.get("reward_info", {})
        reward_info = reward_info if isinstance(reward_info, dict) else {}

        imitation = reward_info.get("imitation", {})
        imitation = imitation if isinstance(imitation, dict) else {}
        site_info = imitation.get("site_info", {})
        joint_info = imitation.get("joint_info", {})
        site_info = site_info if isinstance(site_info, dict) else {}
        joint_info = joint_info if isinstance(joint_info, dict) else {}

        metrics = {
            "site_pos_err": self._to_scalar(site_info.get("pos_err")),
            "site_vel_err": self._to_scalar(site_info.get("vel_err")),
            "joint_angle_err": self._to_scalar(joint_info.get("ang_err")),
        }

        # Recompute only missing imitation diagnostics, for configurations that
        # omit reward_info while still returning observation components.
        if any(metrics[key] is None for key in self.IMITATION_ERROR_KEYS):
            obs_components = info.get("obs_components")
            if obs_components is not None:
                try:
                    _, breakdown = get_imitation_reward(self.base_env, obs_components)
                except Exception:
                    breakdown = {}
                fallback_site = breakdown.get("site_info", {}) if isinstance(breakdown, dict) else {}
                fallback_joint = breakdown.get("joint_info", {}) if isinstance(breakdown, dict) else {}
                if isinstance(fallback_site, dict):
                    if metrics["site_pos_err"] is None:
                        metrics["site_pos_err"] = self._to_scalar(fallback_site.get("pos_err"))
                    if metrics["site_vel_err"] is None:
                        metrics["site_vel_err"] = self._to_scalar(fallback_site.get("vel_err"))
                if isinstance(fallback_joint, dict) and metrics["joint_angle_err"] is None:
                    metrics["joint_angle_err"] = self._to_scalar(fallback_joint.get("ang_err"))

        smooth = reward_info.get("smooth", {})
        smooth = smooth if isinstance(smooth, dict) else {}
        energy = smooth.get("energy", {})
        energy = energy if isinstance(energy, dict) else {}
        energy_details = energy.get("metrics", {})
        energy_details = energy_details if isinstance(energy_details, dict) else {}
        metrics.update(
            {
                "energy_cost": self._to_scalar(energy.get("cost")),
                "activation_l1": self._to_scalar(energy_details.get("l1")),
                "activation_l2_sq": self._to_scalar(energy_details.get("l2_sq")),
                "metab_cost": self._to_scalar(energy_details.get("metab_cost")),
            }
        )

        if any(metrics[key] is None for key in self.ENERGY_ERROR_KEYS):
            fallback_energy = self._compute_current_energy_metrics()
            for key in self.ENERGY_ERROR_KEYS:
                if metrics[key] is None:
                    metrics[key] = fallback_energy.get(key)

        return {key: value for key, value in metrics.items() if value is not None}

    @staticmethod
    def _mean_metrics(sums, counts, keys):
        return {
            key: sums[key] / counts[key]
            for key in keys
            if counts.get(key, 0) > 0
        }

    def _extract_reward_terms(self, reward_info):
        terms = {}
        if not isinstance(reward_info, dict):
            return terms
        for top_k, top_v in reward_info.items():
            if not isinstance(top_v, dict):
                scalar = self._to_scalar(top_v)
                if scalar is not None:
                    terms[top_k] = scalar
                continue

            if top_v.get("skipped"):
                continue

            if top_k == "imitation":
                for k, v in top_v.items():
                    if k.endswith("_info") or k == "skipped":
                        continue
                    scalar = self._to_scalar(v)
                    if scalar is not None:
                        terms[f"imitation.{k}"] = scalar
            elif top_k == "smooth":
                for k, v in top_v.items():
                    if isinstance(v, dict):
                        if v.get("skipped"):
                            continue
                        cost = self._to_scalar(v.get("cost"))
                        weight = self._to_scalar(v.get("weight"))
                        if cost is not None:
                            terms[f"smooth.{k}.cost"] = cost
                            if weight is not None:
                                terms[f"smooth.{k}.weighted"] = cost * weight
                    else:
                        scalar = self._to_scalar(v)
                        if scalar is not None:
                            terms[f"smooth.{k}"] = scalar
            elif top_k == "goal":
                for k, v in top_v.items():
                    if k.endswith("_info") or k == "skipped":
                        continue
                    scalar = self._to_scalar(v)
                    if scalar is not None:
                        terms[f"goal.{k}"] = scalar
            else:
                for k, v in top_v.items():
                    if k.endswith("_info") or k == "skipped":
                        continue
                    scalar = self._to_scalar(v)
                    if scalar is not None:
                        terms[f"{top_k}.{k}"] = scalar
        return terms

    def _extract_imitation_total(self, reward_info):
        imm = reward_info.get("imitation", None) if isinstance(reward_info, dict) else None
        if not isinstance(imm, dict) or imm.get("skipped"):
            return None
        weights = getattr(self.env, "imitation_weights", {}) or {}
        total = 0.0
        found = False
        for key, w in weights.items():
            if key not in imm:
                continue
            scalar = self._to_scalar(imm.get(key))
            if scalar is None:
                continue
            total += float(w) * scalar
            found = True
        return total if found else None

    def _compute_imitation_from_components(self, obs_components):
        if obs_components is None:
            return None
        try:
            r_imm, _ = get_imitation_reward(self.base_env, obs_components)
        except Exception:
            return None
        return float(r_imm)

    def _extract_smooth_component_weighted(self, reward_info, name):
        smooth = reward_info.get("smooth", None) if isinstance(reward_info, dict) else None
        if not isinstance(smooth, dict):
            return None
        comp = smooth.get(name, None)
        if not isinstance(comp, dict) or comp.get("skipped"):
            return None
        cost = self._to_scalar(comp.get("cost"))
        if cost is None:
            return None
        weight = self._to_scalar(comp.get("weight"))
        if weight is None:
            return cost
        return cost * weight

    def run(self):
        """Run visualization loop"""
        num_episodes = int(self.cfg["visualization"].get("num_episodes", self.num_traj))
        # num_episodes = 3
        print(f"Starting visualization for {num_episodes} episodes (all trajectories)...")
        if self.cfg['visualization']["wandb"]:
            wandb.init(project="Test_Env",name="baseline")
            
        overall_reward_sums = {}
        overall_steps = 0
        error_keys = self.IMITATION_ERROR_KEYS + self.ENERGY_ERROR_KEYS
        overall_error_sums = {key: 0.0 for key in error_keys}
        overall_error_counts = {key: 0 for key in error_keys}
        first_k = min(3, num_episodes)
        first_k_means = {"imitation": [], "energy": [], "synergy": []}

        for episode in range(num_episodes):
            obs_raw, info = self.env.reset(id=episode % self.num_traj,frame=0)
            obs = self.obs_normalizer(self._compose_policy_obs(obs_raw, info))
            episode_reward = 0
            done = False
            speed,id = self.env.speed()
            print(f"Episode {episode + 1} (traj_id={id}) starting with speed: {speed:.2f}")

            count = 0
            episode_reward_sums = {}
            episode_error_sums = {key: 0.0 for key in error_keys}
            episode_error_counts = {key: 0 for key in error_keys}
            term_sums = {"imitation": 0.0, "energy": 0.0, "synergy": 0.0}
            term_counts = {"imitation": 0, "energy": 0, "synergy": 0}
            while not done:
                count +=1
                with torch.no_grad():
                    actions = self.policy.act_inference(obs)
                    actions = self._align_actions_to_env(actions)
                    # actions[:,-2:]=actions[:,-2:]*0.0  #disable exo
                obs_raw, reward, done, info = self.env.step(actions)
                # if count>=50:
                #     import ipdb;ipdb.set_trace()
                #     print(reward,info['reward_info']["imitation"],info['reward_info']["smooth"])
                obs = self.obs_normalizer(self._compose_policy_obs(obs_raw, info))
                episode_reward += reward.item()
                reward_info = info.get("reward_info", {})
                terms = self._extract_reward_terms(reward_info)
                for k, v in terms.items():
                    episode_reward_sums[k] = episode_reward_sums.get(k, 0.0) + v
                    overall_reward_sums[k] = overall_reward_sums.get(k, 0.0) + v

                error_metrics = self._extract_error_metrics(info)
                for key, value in error_metrics.items():
                    if key not in episode_error_sums:
                        continue
                    episode_error_sums[key] += value
                    episode_error_counts[key] += 1
                    overall_error_sums[key] += value
                    overall_error_counts[key] += 1
                overall_steps += 1

                imm_total = self._extract_imitation_total(reward_info)
                if imm_total is None:
                    imm_total = self._compute_imitation_from_components(info.get("obs_components"))
                if imm_total is not None:
                    term_sums["imitation"] += imm_total
                    term_counts["imitation"] += 1
                energy_weighted = self._extract_smooth_component_weighted(reward_info, "energy")
                if energy_weighted is not None:
                    term_sums["energy"] += energy_weighted
                    term_counts["energy"] += 1
                synergy_weighted = self._extract_smooth_component_weighted(reward_info, "synergy")
                if synergy_weighted is not None:
                    term_sums["synergy"] += synergy_weighted
                    term_counts["synergy"] += 1
                # Control playback speed
                time.sleep(1.0 / (self.env.max_episode_length * self.cfg["visualization"]["speedup"]))
            
            print("episode reward =",episode_reward, ", steps =", count) 
            if count > 0 and episode_reward_sums:
                episode_reward_means = {k: v / count for k, v in episode_reward_sums.items()}
                print("episode sub-reward mean =", episode_reward_means)
            episode_imitation_errors = self._mean_metrics(
                episode_error_sums, episode_error_counts, self.IMITATION_ERROR_KEYS
            )
            episode_energy_errors = self._mean_metrics(
                episode_error_sums, episode_error_counts, self.ENERGY_ERROR_KEYS
            )
            if episode_imitation_errors:
                print("episode mean imitation error =", episode_imitation_errors)
            if episode_energy_errors:
                print("episode mean energy metrics =", episode_energy_errors)
            print(info["terminated_info"])

            if episode < first_k:
                for k in ("imitation", "energy", "synergy"):
                    if term_counts[k] > 0:
                        first_k_means[k].append(term_sums[k] / term_counts[k])

        if overall_steps > 0 and overall_reward_sums:
            overall_reward_means = {k: v / overall_steps for k, v in overall_reward_sums.items()}
            print("overall sub-reward mean =", overall_reward_means)
        overall_imitation_errors = self._mean_metrics(
            overall_error_sums, overall_error_counts, self.IMITATION_ERROR_KEYS
        )
        overall_energy_errors = self._mean_metrics(
            overall_error_sums, overall_error_counts, self.ENERGY_ERROR_KEYS
        )
        if overall_imitation_errors:
            print("overall mean imitation error =", overall_imitation_errors)
        if overall_energy_errors:
            print("overall mean energy metrics =", overall_energy_errors)
        if first_k > 0:
            avg_first_k = {}
            for k, vals in first_k_means.items():
                if vals:
                    avg_first_k[k] = float(sum(vals) / len(vals))
            if avg_first_k:
                print(f"first {first_k} episode mean (imitation/energy/synergy) =", avg_first_k)
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description="RSL-RL Policy Visualizer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    visualizer = PolicyVisualizer(args.config)
    visualizer.run()
    

if __name__ == "__main__":
    main()
