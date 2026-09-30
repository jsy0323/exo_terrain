# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-aware humanoid policy with an independent active-exoskeleton actor."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

from ..utils import resolve_nn_activation
from .actor_critic_latent_test import TerrainAttentionEncoder


def _validate_hidden_dims(name: str, hidden_dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in hidden_dims)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"{name} must contain positive dimensions, got {list(dims)}")
    return dims


def _make_backbone(input_dim: int, hidden_dims: Sequence[int], activation: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous_dim, int(hidden_dim)),
                copy.deepcopy(activation),
            ]
        )
        previous_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class _TerrainHumanActor(nn.Module):
    """Humanoid actor that encodes the trailing terrain height patch."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: nn.Module,
        terrain_cnn_enabled: bool,
        terrain_patch_shape: Sequence[int],
        terrain_feature_dim: int,
        terrain_attention_enabled: bool,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.terrain_cnn_enabled = bool(terrain_cnn_enabled)
        self.terrain_patch_shape = tuple(int(dim) for dim in terrain_patch_shape)
        if len(self.terrain_patch_shape) != 2 or any(dim <= 0 for dim in self.terrain_patch_shape):
            raise ValueError(
                "terrain_patch_shape must contain two positive dimensions, got "
                f"{self.terrain_patch_shape}"
            )

        self.terrain_obs_dim = int(self.terrain_patch_shape[0] * self.terrain_patch_shape[1])
        self.terrain_feature_dim = int(terrain_feature_dim)
        self.terrain_attention_enabled = bool(terrain_attention_enabled)
        if self.terrain_feature_dim <= 0:
            raise ValueError(f"terrain_feature_dim must be positive, got {self.terrain_feature_dim}")

        if self.terrain_cnn_enabled:
            if self.input_dim <= self.terrain_obs_dim:
                raise ValueError(
                    f"human input_dim={self.input_dim} must exceed terrain_obs_dim={self.terrain_obs_dim}"
                )
            self.base_actor_obs_dim = self.input_dim - self.terrain_obs_dim
            height, width = self.terrain_patch_shape
            if self.terrain_attention_enabled:
                self.terrain_encoder = TerrainAttentionEncoder(
                    patch_shape=self.terrain_patch_shape,
                    feature_dim=self.terrain_feature_dim,
                    base_obs_dim=self.base_actor_obs_dim,
                    activation=activation,
                )
            else:
                self.terrain_encoder = nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=3, padding=1),
                    copy.deepcopy(activation),
                    nn.Conv2d(8, 16, kernel_size=3, padding=1),
                    copy.deepcopy(activation),
                    nn.Flatten(),
                    nn.Linear(16 * height * width, self.terrain_feature_dim),
                    copy.deepcopy(activation),
                )
            mlp_input_dim = self.base_actor_obs_dim + self.terrain_feature_dim
        else:
            self.base_actor_obs_dim = self.input_dim
            self.terrain_encoder = None
            mlp_input_dim = self.input_dim

        self.backbone = _make_backbone(mlp_input_dim, hidden_dims, activation)
        self.action_head = nn.Linear(int(hidden_dims[-1]), self.output_dim)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.zero_()

    def _actor_features(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"Humanoid observation dim mismatch: got {observations.shape[-1]}, expected {self.input_dim}"
            )
        if not self.terrain_cnn_enabled:
            return observations

        base_obs = observations[..., : self.base_actor_obs_dim]
        terrain_obs = observations[
            ..., self.base_actor_obs_dim : self.base_actor_obs_dim + self.terrain_obs_dim
        ]
        height, width = self.terrain_patch_shape
        terrain_image = terrain_obs.reshape(-1, 1, height, width)
        flat_base_obs = base_obs.reshape(-1, self.base_actor_obs_dim)
        if self.terrain_attention_enabled:
            terrain_features = self.terrain_encoder(terrain_image, flat_base_obs)
        else:
            terrain_features = self.terrain_encoder(terrain_image)
        features = torch.cat((flat_base_obs, terrain_features), dim=-1)
        return features.reshape(*observations.shape[:-1], features.shape[-1])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.backbone(self._actor_features(observations)))


class _ExoskeletonActor(nn.Module):
    """Independent exoskeleton MLP, matching the multi-agent actor branch."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: nn.Module,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(int(input_dim), int(hidden_dims[0])),
            copy.deepcopy(activation),
        ]
        for hidden_in, hidden_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.extend(
                [
                    nn.Linear(int(hidden_in), int(hidden_out)),
                    copy.deepcopy(activation),
                ]
            )
        layers.append(nn.Linear(int(hidden_dims[-1]), int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ActorCriticTerrainExo(nn.Module):
    """Two-agent actor critic for terrain locomotion with active ankle assistance.

    The humanoid and exoskeleton have independent actors and a shared critic. The
    humanoid actor retains the terrain CNN/attention path used by the existing
    terrain policy. Environment actions are always emitted in actuator order:
    humanoid controls first, exoskeleton controls last.

    A legacy single-agent terrain checkpoint can be loaded directly. Its terrain
    encoder, humanoid backbone/action head, critic, and humanoid action noise are
    restored; the new exoskeleton actor remains freshly initialized.
    """

    is_recurrent = False
    num_agents = 2

    EXO_PHASE_DIM = 4
    EXO_BASE_JOINT_START = 340
    EXO_FUTURE_REF_DIM = 313
    EXO_SELECTED_SITE_DIM = 89
    _EXO_SELECTED_RANGES = (
        (0, 14),
        (53, 56),
        (152, 158),
        (233, 236),
        (293, 296),
        (62, 65),
        (170, 176),
        (242, 245),
        (302, 305),
        (71, 74),
        (188, 194),
        (251, 254),
        (311, 314),
        (20, 23),
        (86, 92),
        (200, 203),
        (260, 263),
        (35, 38),
        (116, 122),
        (215, 218),
        (275, 278),
    )

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        exo_action_dim: int = 2,
        exo_actor_hidden_dims: Sequence[int] | None = None,
        first256: bool = False,
        exo_obs: bool = False,
        exo_obs_noise_std: float = 0.0,
        muscle_action_dim: int = 80,
        agent_obs_dims: Sequence[int] | None = None,
        agent_action_dims: Sequence[int] | None = None,
        shared_actor_obs: bool = True,
        human_branch_idx: int = 1,
        terrain_cnn_enabled: bool = True,
        terrain_patch_shape: Sequence[int] = (20, 30),
        terrain_feature_dim: int = 81,
        terrain_attention_enabled: bool = True,
        terrain_obs_enabled: bool | None = None,
        latent_cov_scale: float | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTerrainExo.__init__ got unexpected arguments, which will be ignored: "
                + str(sorted(kwargs))
            )
        del terrain_obs_enabled
        super().__init__()

        actor_hidden_dims = _validate_hidden_dims("actor_hidden_dims", actor_hidden_dims)
        critic_hidden_dims = _validate_hidden_dims("critic_hidden_dims", critic_hidden_dims)
        if exo_actor_hidden_dims is None:
            exo_actor_hidden_dims = actor_hidden_dims
        exo_actor_hidden_dims = _validate_hidden_dims(
            "exo_actor_hidden_dims", exo_actor_hidden_dims
        )

        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.shared_actor_obs = bool(shared_actor_obs)
        self.human_branch_idx = int(human_branch_idx)
        if self.human_branch_idx not in (0, 1):
            raise ValueError(f"human_branch_idx must be 0 or 1, got {self.human_branch_idx}")
        self.exo_branch_idx = 1 - self.human_branch_idx
        self.exo_action_dim = int(exo_action_dim)
        if self.exo_action_dim <= 0:
            raise ValueError(f"exo_action_dim must be positive, got {self.exo_action_dim}")

        if agent_action_dims is None:
            human_action_dim = self.num_actions - self.exo_action_dim
            agent_action_dims = [0, 0]
            agent_action_dims[self.human_branch_idx] = human_action_dim
            agent_action_dims[self.exo_branch_idx] = self.exo_action_dim
        self.agent_action_dims = [int(dim) for dim in agent_action_dims]
        if len(self.agent_action_dims) != self.num_agents:
            raise ValueError(
                "ActorCriticTerrainExo expects two action branches, got "
                f"{self.agent_action_dims}"
            )
        if sum(self.agent_action_dims) != self.num_actions:
            raise ValueError(
                f"sum(agent_action_dims)={sum(self.agent_action_dims)} must equal "
                f"num_actions={self.num_actions}"
            )
        if self.agent_action_dims[self.exo_branch_idx] != self.exo_action_dim:
            raise ValueError(
                f"Exoskeleton branch must output {self.exo_action_dim} actions, got "
                f"{self.agent_action_dims[self.exo_branch_idx]}"
            )
        self.human_action_dim = self.agent_action_dims[self.human_branch_idx]
        if self.human_action_dim <= 0:
            raise ValueError("Humanoid action dimension must be positive")

        self.terrain_cnn_enabled = bool(terrain_cnn_enabled)
        self.terrain_patch_shape = tuple(int(dim) for dim in terrain_patch_shape)
        self.terrain_obs_dim = int(self.terrain_patch_shape[0] * self.terrain_patch_shape[1])
        self.terrain_feature_dim = int(terrain_feature_dim)
        self.terrain_attention_enabled = bool(terrain_attention_enabled)
        self.muscle_action_dim = int(muscle_action_dim)
        self.muscle_obs_dim = self.muscle_action_dim
        self.first256 = bool(first256)
        self.exo_obs = bool(exo_obs)
        if not self.shared_actor_obs and (self.first256 or self.exo_obs):
            raise ValueError("first256 and exo_obs require shared_actor_obs=True")
        if self.first256 and self.exo_obs:
            raise ValueError("first256 and exo_obs are mutually exclusive")

        self.exo_obs_noise_std = float(exo_obs_noise_std)
        if self.exo_obs_noise_std < 0.0:
            raise ValueError(
                f"exo_obs_noise_std must be non-negative, got {self.exo_obs_noise_std}"
            )

        if self.shared_actor_obs:
            self.agent_obs_dims = [self.num_actor_obs, self.num_actor_obs]
            human_obs_dim = self.num_actor_obs
            if self.first256:
                if self.num_actor_obs < 256:
                    raise ValueError(
                        f"first256 requires num_actor_obs >= 256, got {self.num_actor_obs}"
                    )
                exo_actor_obs_dim = 256
            elif self.exo_obs:
                exo_actor_obs_dim = self._infer_exo_obs_dim(self.num_actor_obs)
            else:
                exo_actor_obs_dim = self.num_actor_obs
        else:
            if agent_obs_dims is None:
                raise ValueError("agent_obs_dims is required when shared_actor_obs=False")
            self.agent_obs_dims = [int(dim) for dim in agent_obs_dims]
            if len(self.agent_obs_dims) != self.num_agents:
                raise ValueError(
                    f"ActorCriticTerrainExo expects two observation branches, got {self.agent_obs_dims}"
                )
            if sum(self.agent_obs_dims) != self.num_actor_obs:
                raise ValueError(
                    f"sum(agent_obs_dims)={sum(self.agent_obs_dims)} must equal "
                    f"num_actor_obs={self.num_actor_obs}"
                )
            human_obs_dim = self.agent_obs_dims[self.human_branch_idx]
            exo_actor_obs_dim = self.agent_obs_dims[self.exo_branch_idx]

        self.exo_obs_dim = int(exo_actor_obs_dim)
        self.actor_input_dims = [0, 0]
        self.actor_input_dims[self.human_branch_idx] = int(human_obs_dim)
        self.actor_input_dims[self.exo_branch_idx] = self.exo_obs_dim

        act_fn = resolve_nn_activation(activation)
        self.human_actor = _TerrainHumanActor(
            input_dim=human_obs_dim,
            output_dim=self.human_action_dim,
            hidden_dims=actor_hidden_dims,
            activation=act_fn,
            terrain_cnn_enabled=self.terrain_cnn_enabled,
            terrain_patch_shape=self.terrain_patch_shape,
            terrain_feature_dim=self.terrain_feature_dim,
            terrain_attention_enabled=self.terrain_attention_enabled,
        )
        self.exo_actor = _ExoskeletonActor(
            input_dim=self.exo_obs_dim,
            output_dim=self.exo_action_dim,
            hidden_dims=exo_actor_hidden_dims,
            activation=act_fn,
        )

        critic_layers: list[nn.Module] = [
            nn.Linear(self.num_critic_obs, critic_hidden_dims[0]),
            copy.deepcopy(act_fn),
        ]
        for hidden_in, hidden_out in zip(critic_hidden_dims[:-1], critic_hidden_dims[1:]):
            critic_layers.extend(
                [
                    nn.Linear(hidden_in, hidden_out),
                    copy.deepcopy(act_fn),
                ]
            )
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        init_noise_std = float(init_noise_std)
        if init_noise_std <= 0.0:
            raise ValueError(f"init_noise_std must be positive, got {init_noise_std}")
        self.noise_std_type = str(noise_std_type)
        self.latent_dim = int(actor_hidden_dims[-1])
        self.latent_cov_scale = 1.0 if latent_cov_scale is None else float(latent_cov_scale)
        if self.latent_cov_scale < 0.0:
            raise ValueError(
                f"latent_cov_scale must be non-negative, got {self.latent_cov_scale}"
            )
        initial_log_std = torch.log(torch.tensor(init_noise_std))
        # Preserve the existing terrain policy's action + latent covariance noise.
        self.log_std = nn.Parameter(
            torch.full((1, self.human_action_dim + self.latent_dim), initial_log_std)
        )
        # The independent exoskeleton branch follows ActorCriticMultiAgent.
        self.exo_log_std = nn.Parameter(
            torch.full((self.exo_action_dim,), initial_log_std)
        )
        self.human_distribution: MultivariateNormal | None = None
        self.exo_distribution: Normal | None = None
        self.distribution = None
        Normal.set_default_validate_args(False)
        MultivariateNormal.set_default_validate_args(False)

        # The physical model exposes humanoid actuators first and exo motors last.
        self.output_branch_order = [self.human_branch_idx, self.exo_branch_idx]
        self.output_action_dims = [self.human_action_dim, self.exo_action_dim]
        self.register_buffer("_action_layout_version", torch.tensor(3, dtype=torch.int32))

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        branch_means: list[torch.Tensor | None] = [None, None]
        branch_means[self.human_branch_idx] = self.human_distribution.mean
        branch_means[self.exo_branch_idx] = self.exo_distribution.mean
        return self._compose_env_actions([mean for mean in branch_means if mean is not None])

    @property
    def action_std(self) -> torch.Tensor:
        branch_stds: list[torch.Tensor | None] = [None, None]
        branch_stds[self.human_branch_idx] = self.human_distribution.stddev
        branch_stds[self.exo_branch_idx] = self.exo_distribution.stddev
        return self._compose_env_actions([std for std in branch_stds if std is not None])

    @property
    def entropy(self) -> torch.Tensor:
        return self.human_distribution.entropy() + self.exo_distribution.entropy().sum(dim=-1)

    @property
    def entropy_per_agent(self) -> torch.Tensor:
        branch_entropy: list[torch.Tensor | None] = [None, None]
        branch_entropy[self.human_branch_idx] = self.human_distribution.entropy()
        branch_entropy[self.exo_branch_idx] = self.exo_distribution.entropy().sum(dim=-1)
        return torch.stack(
            [entropy for entropy in branch_entropy if entropy is not None], dim=-1
        )

    @property
    def last_terrain_attention(self) -> torch.Tensor | None:
        encoder = self.human_actor.terrain_encoder
        return getattr(encoder, "last_attention", None)

    def _terrain_dim_for_obs(self, observation_dim: int) -> int:
        if not self.terrain_cnn_enabled:
            return 0
        if observation_dim < self.terrain_obs_dim:
            raise ValueError(
                f"Observation dim {observation_dim} is smaller than terrain dim {self.terrain_obs_dim}"
            )
        return self.terrain_obs_dim

    def _infer_exo_obs_dim(self, observation_dim: int) -> int:
        terrain_dim = self._terrain_dim_for_obs(observation_dim)
        base_state_dim = (
            int(observation_dim)
            - self.EXO_PHASE_DIM
            - self.muscle_obs_dim
            - terrain_dim
        )
        required_selected_dim = max(stop for _, stop in self._EXO_SELECTED_RANGES)
        if base_state_dim < required_selected_dim:
            raise ValueError(
                "exo_obs requires enough base-state features for the selected-site blocks: "
                f"base_state_dim={base_state_dim}, required={required_selected_dim}"
            )
        joint_end = base_state_dim - self.EXO_FUTURE_REF_DIM
        joint_dim = max(0, joint_end - self.EXO_BASE_JOINT_START)
        return self.EXO_SELECTED_SITE_DIM + joint_dim

    def _build_exo_obs(self, observations: torch.Tensor) -> torch.Tensor:
        terrain_dim = self._terrain_dim_for_obs(int(observations.shape[-1]))
        base_state_end = observations.shape[-1] - self.muscle_obs_dim - terrain_dim
        base_state = observations[..., self.EXO_PHASE_DIM:base_state_end]
        selected = [base_state[..., start:stop] for start, stop in self._EXO_SELECTED_RANGES]

        joint_end = base_state.shape[-1] - self.EXO_FUTURE_REF_DIM
        if joint_end > self.EXO_BASE_JOINT_START:
            selected.append(base_state[..., self.EXO_BASE_JOINT_START:joint_end])
        exo_observations = torch.cat(selected, dim=-1)
        if exo_observations.shape[-1] != self.exo_obs_dim:
            raise RuntimeError(
                f"Built exo observation dim {exo_observations.shape[-1]}, expected {self.exo_obs_dim}"
            )
        return exo_observations

    def _add_exo_obs_noise(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.training or self.exo_obs_noise_std <= 0.0:
            return observations
        return observations + torch.randn_like(observations) * self.exo_obs_noise_std

    def _branch_observations(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_actor_obs:
            human_observations = observations
            if self.first256:
                exo_observations = observations[..., :256]
            elif self.exo_obs:
                exo_observations = self._build_exo_obs(observations)
            else:
                exo_observations = observations
        else:
            chunks = torch.split(observations, self.agent_obs_dims, dim=-1)
            human_observations = chunks[self.human_branch_idx]
            exo_observations = chunks[self.exo_branch_idx]
        return human_observations, self._add_exo_obs_noise(exo_observations)

    def _compute_actor_means(self, observations: torch.Tensor) -> list[torch.Tensor]:
        human_observations, exo_observations = self._branch_observations(observations)
        branch_means: list[torch.Tensor | None] = [None, None]
        branch_means[self.human_branch_idx] = self.human_actor(human_observations)
        branch_means[self.exo_branch_idx] = self.exo_actor(exo_observations)
        return [mean for mean in branch_means if mean is not None]

    def _compose_env_actions(self, branch_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [branch_tensors[branch_idx] for branch_idx in self.output_branch_order], dim=-1
        )

    def _split_env_actions_by_branch(self, actions: torch.Tensor) -> list[torch.Tensor]:
        environment_chunks = torch.split(actions, self.output_action_dims, dim=-1)
        branch_chunks: list[torch.Tensor | None] = [None, None]
        for chunk, branch_idx in zip(environment_chunks, self.output_branch_order):
            branch_chunks[branch_idx] = chunk
        return [chunk for chunk in branch_chunks if chunk is not None]

    def update_distribution(self, observations: torch.Tensor):
        branch_means = self._compute_actor_means(observations)
        human_mean = branch_means[self.human_branch_idx]
        exo_mean = branch_means[self.exo_branch_idx]

        standard_deviations = torch.exp(self.log_std)
        human_action_variance = standard_deviations[:, : self.human_action_dim].square()
        latent_variance = standard_deviations[:, self.human_action_dim :].square()
        human_head_weight = self.human_actor.action_head.weight
        covariance = (
            human_head_weight * latent_variance[..., None, :]
        ).matmul(human_head_weight.t())
        covariance = covariance * self.latent_cov_scale
        diagonal = torch.arange(self.human_action_dim, device=covariance.device)
        covariance[..., diagonal, diagonal] += human_action_variance

        self.human_distribution = MultivariateNormal(human_mean, covariance)
        exo_std = torch.exp(self.exo_log_std).unsqueeze(0).expand_as(exo_mean)
        self.exo_distribution = Normal(exo_mean, exo_std)
        self.distribution = (self.human_distribution, self.exo_distribution)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        branch_actions: list[torch.Tensor | None] = [None, None]
        branch_actions[self.human_branch_idx] = self.human_distribution.sample()
        branch_actions[self.exo_branch_idx] = self.exo_distribution.sample()
        return self._compose_env_actions(
            [actions for actions in branch_actions if actions is not None]
        )

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self._compose_env_actions(self._compute_actor_means(observations))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        branch_actions = self._split_env_actions_by_branch(actions)
        human_log_prob = self.human_distribution.log_prob(
            branch_actions[self.human_branch_idx]
        )
        exo_log_prob = self.exo_distribution.log_prob(
            branch_actions[self.exo_branch_idx]
        ).sum(dim=-1)
        return human_log_prob + exo_log_prob

    def get_actions_log_prob_per_agent(self, actions: torch.Tensor) -> torch.Tensor:
        branch_actions = self._split_env_actions_by_branch(actions)
        branch_log_prob: list[torch.Tensor | None] = [None, None]
        branch_log_prob[self.human_branch_idx] = self.human_distribution.log_prob(
            branch_actions[self.human_branch_idx]
        )
        branch_log_prob[self.exo_branch_idx] = self.exo_distribution.log_prob(
            branch_actions[self.exo_branch_idx]
        ).sum(dim=-1)
        return torch.stack(
            [log_prob for log_prob in branch_log_prob if log_prob is not None], dim=-1
        )

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def evaluate_per_agent(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.evaluate(critic_observations).expand(-1, self.num_agents)

    def _output_action_slice(self, branch_idx: int) -> slice:
        order_index = self.output_branch_order.index(branch_idx)
        start = sum(self.output_action_dims[:order_index])
        return slice(start, start + self.output_action_dims[order_index])

    @staticmethod
    def _copy_if_compatible(current_state: dict, target_key: str, tensor: torch.Tensor) -> bool:
        target = current_state.get(target_key)
        if target is None or target.shape != tensor.shape:
            return False
        current_state[target_key] = tensor
        return True

    def _load_legacy_humanoid_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> bool:
        if "action_head.weight" not in state_dict:
            return False

        current_state = self.state_dict()
        loaded_keys: list[str] = []
        prefix_map = {
            "terrain_encoder.": "human_actor.terrain_encoder.",
            "backbone.": "human_actor.backbone.",
            "action_head.": "human_actor.action_head.",
            "critic.": "critic.",
        }
        for source_key, tensor in state_dict.items():
            for source_prefix, target_prefix in prefix_map.items():
                if source_key.startswith(source_prefix):
                    target_key = target_prefix + source_key[len(source_prefix) :]
                    if self._copy_if_compatible(current_state, target_key, tensor):
                        loaded_keys.append(target_key)
                    break

        legacy_log_std = state_dict.get("log_std")
        if legacy_log_std is not None:
            legacy_log_std = legacy_log_std.detach()
            if legacy_log_std.shape == current_state["log_std"].shape:
                current_state["log_std"] = legacy_log_std.to(current_state["log_std"])
                loaded_keys.append("log_std")
            else:
                print(
                    "ActorCriticTerrainExo: legacy human log_std shape does not match "
                    f"({tuple(legacy_log_std.shape)} != {tuple(current_state['log_std'].shape)}); skipped."
                )

        if not loaded_keys:
            return False
        super().load_state_dict(current_state, strict=False)
        print(
            "ActorCriticTerrainExo: loaded legacy terrain checkpoint into the humanoid "
            "actor; exoskeleton actor kept its initialization."
        )
        return True

    def load_state_dict(self, state_dict, strict: bool = True):
        clean_state = {str(key).replace("module.", ""): value for key, value in state_dict.items()}
        if self._load_legacy_humanoid_checkpoint(clean_state):
            return True
        super().load_state_dict(clean_state, strict=strict)
        return True


# Convenient direct import parity with the other actor_critic_* modules.
ActorCritic = ActorCriticTerrainExo


__all__ = ["ActorCritic", "ActorCriticTerrainExo"]
