#!/usr/bin/env python3
"""Visualize a terrain humanoid policy with an exo actor from a flat policy."""

import argparse
from pathlib import Path

import torch
from torch import nn

from RL.modules import ActorCriticMultiAgent, EmpiricalNormalization
from visualize import PolicyVisualizer


class _RecordingNormalizer:
    """Preserve the raw observation while keeping the normal visualizer API."""

    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.last_raw = None

    def __call__(self, observations):
        self.last_raw = observations
        return self.normalizer(observations)


class _HybridTerrainExoPolicy(nn.Module):
    """Compose humanoid actions from one policy and exo actions from another."""

    def __init__(
        self,
        humanoid_policy,
        flat_exo_policy,
        flat_obs_normalizer,
        observation_recorder,
        humanoid_action_dim,
        flat_obs_dim,
    ):
        super().__init__()
        self.humanoid_policy = humanoid_policy
        self.flat_exo_policy = flat_exo_policy
        self.flat_obs_normalizer = flat_obs_normalizer
        self.observation_recorder = observation_recorder
        self.humanoid_action_dim = int(humanoid_action_dim)
        self.flat_obs_dim = int(flat_obs_dim)

    def act_inference(self, terrain_observations):
        raw_observations = self.observation_recorder.last_raw
        if raw_observations is None:
            raise RuntimeError("Raw observations were not recorded before policy inference.")
        if int(raw_observations.shape[-1]) < self.flat_obs_dim:
            raise ValueError(
                f"Terrain observation has {raw_observations.shape[-1]} values, but the flat "
                f"exo checkpoint requires {self.flat_obs_dim}."
            )

        humanoid_actions = self.humanoid_policy.act_inference(terrain_observations)
        if int(humanoid_actions.shape[-1]) < self.humanoid_action_dim:
            raise ValueError(
                f"Terrain humanoid policy produced {humanoid_actions.shape[-1]} actions, "
                f"expected at least {self.humanoid_action_dim}."
            )
        humanoid_actions = humanoid_actions[..., : self.humanoid_action_dim]

        # Terrain observations append the height patch at the end, so their first
        # flat_obs_dim values exactly match the flat-policy observation layout.
        flat_raw_observations = raw_observations[..., : self.flat_obs_dim]
        flat_observations = self.flat_obs_normalizer(flat_raw_observations)
        flat_branch_means = self.flat_exo_policy._compute_actor_means(flat_observations)
        exo_actions = flat_branch_means[self.flat_exo_policy.exo_branch_idx]
        return torch.cat((humanoid_actions, exo_actions), dim=-1)


class HybridTerrainExoVisualizer(PolicyVisualizer):
    def __init__(self, cfg_path, exo_checkpoint, terrain_checkpoint=None):
        self.exo_checkpoint_path = str(exo_checkpoint)
        self.terrain_checkpoint_override = (
            None if terrain_checkpoint is None else str(terrain_checkpoint)
        )
        super().__init__(cfg_path)
        self._install_flat_exo_policy()

    def load_config(self, cfg_path):
        super().load_config(cfg_path)
        if self.terrain_checkpoint_override is not None:
            self.cfg["policy"]["checkpoint"] = self.terrain_checkpoint_override

    def _install_flat_exo_policy(self):
        checkpoint_path = Path(self.exo_checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Flat exo checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        state_dict = {
            str(key).replace("module.", ""): value
            for key, value in checkpoint["model_state_dict"].items()
        }
        if not any(key.startswith("actor_branches.") for key in state_dict):
            raise ValueError(
                "Flat exo checkpoint is not an ActorCriticMultiAgent checkpoint: "
                f"{checkpoint_path}"
            )

        norm_state = checkpoint.get("obs_norm_state_dict")
        if not isinstance(norm_state, dict) or "_mean" not in norm_state:
            raise ValueError(f"Flat exo checkpoint has no observation normalizer: {checkpoint_path}")
        flat_obs_dim = int(norm_state["_mean"].shape[-1])
        spec = self._infer_multi_agent_spec(state_dict, flat_obs_dim)
        if spec["exo_action_dim"] != 2:
            raise ValueError(
                f"Flat exo policy must output 2 exo actions, got {spec['exo_action_dim']}."
            )

        flat_policy = ActorCriticMultiAgent(
            num_actor_obs=spec["num_actor_obs"],
            num_critic_obs=spec["num_critic_obs"],
            num_actions=spec["num_actions"],
            actor_hidden_dims=spec["actor_hidden_dims"],
            critic_hidden_dims=spec["critic_hidden_dims"],
            activation=self.cfg["policy"].get("activation", "elu"),
            init_noise_std=float(self.cfg["policy"].get("init_noise_std", 0.1)),
            noise_std_type=str(self.cfg["policy"].get("noise_std_type", "scalar")),
            exo_action_dim=spec["exo_action_dim"],
            first256=spec["first256"],
            exo_obs=spec["exo_obs"],
            exo_obs_noise_std=0.0,
            muscle_action_dim=int(self.cfg["policy"].get("muscle_action_dim", 80)),
            agent_obs_dims=spec["agent_obs_dims"] if not spec["shared_actor_obs"] else None,
            agent_action_dims=spec["agent_action_dims"],
            shared_actor_obs=spec["shared_actor_obs"],
            human_branch_idx=spec["human_branch_idx"],
        ).to(self.device)
        flat_policy.load_state_dict(state_dict)
        flat_policy.eval()

        flat_normalizer = EmpiricalNormalization(shape=[flat_obs_dim], until=1.0e8).to(
            self.device
        )
        flat_normalizer.load_state_dict(norm_state)
        flat_normalizer.eval()

        terrain_obs_dim = int(self.num_actor_obs)
        terrain_patch_dim = int(getattr(self.base_env, "_obs_terrain_dim", 0))
        if terrain_obs_dim - flat_obs_dim != terrain_patch_dim:
            raise ValueError(
                "Flat and terrain observation layouts are incompatible: "
                f"terrain={terrain_obs_dim}, flat={flat_obs_dim}, "
                f"terrain_patch={terrain_patch_dim}. Expected terrain-flat == terrain_patch."
            )

        exo_action_dim = int(spec["exo_action_dim"])
        humanoid_action_dim = int(self.env.num_actions) - exo_action_dim
        actuator_tail = list(getattr(self.base_env, "actuator_names", ()))[
            humanoid_action_dim:
        ]
        if len(actuator_tail) != exo_action_dim or not all(
            self._is_optional_exo_actuator(name) for name in actuator_tail
        ):
            raise ValueError(
                "Environment actions do not end with the expected exoskeleton actuators: "
                f"{actuator_tail}"
            )

        recorder = _RecordingNormalizer(self.obs_normalizer)
        humanoid_policy = self.policy
        self.obs_normalizer = recorder
        self.policy = _HybridTerrainExoPolicy(
            humanoid_policy=humanoid_policy,
            flat_exo_policy=flat_policy,
            flat_obs_normalizer=flat_normalizer,
            observation_recorder=recorder,
            humanoid_action_dim=humanoid_action_dim,
            flat_obs_dim=flat_obs_dim,
        ).to(self.device)
        self.policy.eval()

        print(
            "Hybrid policy ready: "
            f"terrain humanoid={self.cfg['policy']['checkpoint']} "
            f"({humanoid_action_dim} actions), flat exo={checkpoint_path} "
            f"({exo_action_dim} actions, {flat_obs_dim}D normalizer, "
            f"exo input={flat_policy.actor_input_dims[flat_policy.exo_branch_idx]}D)."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a terrain humanoid policy combined with a flat-trained exo actor."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Visualization configuration containing the terrain humanoid checkpoint.",
    )
    parser.add_argument(
        "--exo-checkpoint",
        type=str,
        default=str(
            Path(__file__).resolve().parent
            / "models"
            / "exo_flat_terrain"
            / "model_best.pt"
        ),
        help="Flat-trained checkpoint that supplies the exoskeleton actor.",
    )
    parser.add_argument(
        "--terrain-checkpoint",
        type=str,
        default=None,
        help="Optional terrain humanoid checkpoint override.",
    )
    args = parser.parse_args()

    visualizer = HybridTerrainExoVisualizer(
        cfg_path=args.config,
        exo_checkpoint=args.exo_checkpoint,
        terrain_checkpoint=args.terrain_checkpoint,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
