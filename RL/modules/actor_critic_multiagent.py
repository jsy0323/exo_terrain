from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from ..utils import resolve_nn_activation


class _ActorBranch(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims, activation):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for h_in, h_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers += [nn.Linear(h_in, h_out), activation]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorCriticMultiAgent(nn.Module):
    is_recurrent = False
    EXO_PHASE_DIM = 4
    EXO_BASE_JOINT_START = 340
    EXO_FUTURE_REF_DIM = 313
    EXO_SELECTED_SITE_DIM = 89

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        exo_action_dim: int = 2,
        first256: bool = False,
        exo_obs: bool = False,
        exo_obs_noise_std: float = 0,
        muscle_action_dim: int = 80,
        agent_obs_dims=None,
        agent_action_dims=None,
        shared_actor_obs: bool = True,
        human_branch_idx: int = 1,
        **kwargs,
    ):
        """Two-branch actor critic for shared-state exoskeleton control.

        Args:
            num_actor_obs: Total observation dimension for policy input.
            num_critic_obs: Total observation dimension for critic input.
            num_actions: Total environment action dimension.
            exo_action_dim: Action dimension produced by the exo branch. Defaults to 2.
            first256: If True, the exo branch only consumes the first 256 dimensions of the
                shared policy observation.
            agent_obs_dims: If `shared_actor_obs=False`, list of per-branch observation dims whose
                sum must equal `num_actor_obs`.
            agent_action_dims: Optional explicit `[exo_dim, human_dim]` or `[human_dim, exo_dim]`
                branch layout. When omitted, the human branch keeps `num_actions - exo_action_dim`
                outputs and the exo branch keeps `exo_action_dim`.
            shared_actor_obs: If True, both branches consume the same full observation.
            human_branch_idx: Which branch should receive legacy human checkpoint weights.
        """
        if kwargs:
            print(
                "ActorCriticMultiAgent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()
        act_fn = resolve_nn_activation(activation)
        total_action_dim = int(num_actions)
        self.num_agents = 2
        self.shared_actor_obs = bool(shared_actor_obs)
        self.human_branch_idx = int(human_branch_idx)
        if not (0 <= self.human_branch_idx < self.num_agents):
            raise ValueError(
                f"human_branch_idx must be in [0, {self.num_agents}), got {self.human_branch_idx}"
            )
        self.exo_branch_idx = 1 - self.human_branch_idx
        self.exo_action_dim = int(exo_action_dim)
        if self.exo_action_dim <= 0:
            raise ValueError(f"exo_action_dim must be positive, got {self.exo_action_dim}")
        self.first256 = bool(first256)
        self.exo_obs = bool(exo_obs)
        self.exo_obs_noise_std = float(exo_obs_noise_std)
        self.muscle_obs_dim = int(muscle_action_dim)
        self.exo_obs_dim = 256 if self.first256 else (
            self._infer_exo_obs_dim(int(num_actor_obs)) if self.exo_obs else int(num_actor_obs)
        )
        self.exo_obs_noise_std = float(exo_obs_noise_std)
        if self.exo_obs_noise_std < 0.0:
            raise ValueError(
                f"exo_obs_noise_std must be non-negative, got {self.exo_obs_noise_std}"
            )
        # Environment actuator order is human controls first, exo motors last.
        self.output_branch_order = [self.human_branch_idx, self.exo_branch_idx]

        if agent_action_dims is None:
            human_action_dim = total_action_dim - self.exo_action_dim
            if human_action_dim <= 0:
                raise ValueError(
                    f"num_actions={total_action_dim} must be larger than exo_action_dim={self.exo_action_dim}"
                )
            agent_action_dims = [0, 0]
            agent_action_dims[self.exo_branch_idx] = self.exo_action_dim
            agent_action_dims[self.human_branch_idx] = human_action_dim

        self.agent_action_dims = [int(v) for v in agent_action_dims]
        if len(self.agent_action_dims) != self.num_agents:
            raise ValueError(
                f"ActorCriticMultiAgent expects exactly two branches, got agent_action_dims={self.agent_action_dims}"
            )
        if self.agent_action_dims[self.exo_branch_idx] != self.exo_action_dim:
            raise ValueError(
                f"Exo branch must output {self.exo_action_dim} actions, "
                f"got {self.agent_action_dims[self.exo_branch_idx]}"
            )
        if sum(self.agent_action_dims) != total_action_dim:
            raise ValueError(
                f"sum(agent_action_dims)={sum(self.agent_action_dims)} must equal num_actions={total_action_dim}"
            )
        self.human_action_dim = self.agent_action_dims[self.human_branch_idx]
        self.output_action_dims = [
            self.agent_action_dims[branch_idx] for branch_idx in self.output_branch_order
        ]
        self.register_buffer("_action_layout_version", torch.tensor(2, dtype=torch.int32))

        if self.shared_actor_obs:
            # Each branch uses the full observation (reward/state are shared)
            self.agent_obs_dims = [int(num_actor_obs)] * self.num_agents
        else:
            if self.first256:
                raise ValueError("first256 requires shared_actor_obs=True.")
            if agent_obs_dims is None:
                raise ValueError(
                    "agent_obs_dims must be provided when shared_actor_obs=False."
                )
            self.agent_obs_dims = [int(v) for v in agent_obs_dims]

            if len(self.agent_obs_dims) != self.num_agents:
                raise ValueError(
                    f"ActorCriticMultiAgent expects exactly two observation branches, got {self.agent_obs_dims}"
                )
            if sum(self.agent_obs_dims) != int(num_actor_obs):
                raise ValueError(
                    f"sum(agent_obs_dims)={sum(self.agent_obs_dims)} must equal num_actor_obs={num_actor_obs}"
                )

        if self.first256 and int(num_actor_obs) < self.exo_obs_dim:
            raise ValueError(
                f"first256 requires num_actor_obs >= {self.exo_obs_dim}, got {num_actor_obs}"
            )
        if self.exo_obs and int(num_actor_obs) < self.exo_obs_dim:
            raise ValueError(
                f"exo_obs requires num_actor_obs >= {self.exo_obs_dim}, got {num_actor_obs}"
            )

        self.actor_input_dims = list(self.agent_obs_dims)
        if self.shared_actor_obs and self.first256:
            self.actor_input_dims[self.exo_branch_idx] = self.exo_obs_dim
        if self.shared_actor_obs and self.exo_obs:
            self.actor_input_dims[self.exo_branch_idx] = self.exo_obs_dim

        self.actor_branches = nn.ModuleList(
            _ActorBranch(obs_dim, act_dim, actor_hidden_dims, act_fn)
            for obs_dim, act_dim in zip(self.actor_input_dims, self.agent_action_dims)
        )

        critic_layers = [nn.Linear(int(num_critic_obs), critic_hidden_dims[0]), act_fn]
        for h_in, h_out in zip(critic_hidden_dims[:-1], critic_hidden_dims[1:]):
            critic_layers += [nn.Linear(h_in, h_out), act_fn]
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

        self.noise_std_type = noise_std_type
        init_std = float(init_noise_std) if init_noise_std is not None else 1.0
        self.log_std = nn.Parameter(torch.log(torch.ones(total_action_dim) * init_std))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def entropy_per_agent(self):
        ent = self.distribution.entropy()
        branch_chunks = self._split_env_actions_by_branch(ent)
        return torch.stack([chunk.sum(dim=-1) for chunk in branch_chunks], dim=-1)

    @property
    def exo_actor(self):
        return self.actor_branches[self.exo_branch_idx]

    @property
    def human_actor(self):
        return self.actor_branches[self.human_branch_idx]

    def _output_action_slice(self, branch_idx: int) -> slice:
        order_idx = self.output_branch_order.index(branch_idx)
        start = sum(self.output_action_dims[:order_idx])
        end = start + self.output_action_dims[order_idx]
        return slice(start, end)

    def _add_exo_obs_noise(self, exo_obs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.exo_obs_noise_std <= 0.0:
            return exo_obs
        return exo_obs + torch.randn_like(exo_obs) * self.exo_obs_noise_std

    def _infer_exo_obs_dim(self, num_actor_obs: int) -> int:
        base_obs_dim = (
            int(num_actor_obs) - self.EXO_PHASE_DIM - self.muscle_obs_dim
        )
        joint_dim = base_obs_dim - self.EXO_BASE_JOINT_START - self.EXO_FUTURE_REF_DIM
        if joint_dim <= 0:
            raise ValueError(
                "exo_obs requires a positive joint block size, got "
                f"base_obs_dim={base_obs_dim}, joint_dim={joint_dim}"
            )
        return self.EXO_SELECTED_SITE_DIM + joint_dim

    def _build_exo_obs(self, observations: torch.Tensor) -> torch.Tensor:
        base_obs_end = observations.shape[-1] - self.muscle_obs_dim
        if base_obs_end <= self.EXO_PHASE_DIM:
            raise ValueError(
                f"Observation dim {observations.shape[-1]} is too small for exo_obs extraction."
            )
        base_obs = observations[..., self.EXO_PHASE_DIM:base_obs_end]
        joint_end = base_obs.shape[-1] - self.EXO_FUTURE_REF_DIM
        if joint_end <= self.EXO_BASE_JOINT_START:
            raise ValueError(
                "Base observation is too small to contain the full joint block: "
                f"base_obs_dim={base_obs.shape[-1]}, joint_end={joint_end}"
            )

        return torch.cat(
            [
                base_obs[..., 0:14],
                base_obs[..., 53:56],
                base_obs[..., 152:158],
                base_obs[..., 233:236],
                base_obs[..., 293:296],
                base_obs[..., 62:65],
                base_obs[..., 170:176],
                base_obs[..., 242:245],
                base_obs[..., 302:305],
                base_obs[..., 71:74],
                base_obs[..., 188:194],
                base_obs[..., 251:254],
                base_obs[..., 311:314],
                base_obs[..., 20:23],
                base_obs[..., 86:92],
                base_obs[..., 200:203],
                base_obs[..., 260:263],
                base_obs[..., 35:38],
                base_obs[..., 116:122],
                base_obs[..., 215:218],
                base_obs[..., 275:278],
                base_obs[..., self.EXO_BASE_JOINT_START:joint_end],
            ],
            dim=-1,
        )

    def _compute_actor_means(self, observations: torch.Tensor):
        if self.shared_actor_obs:
            branch_inputs = [observations] * self.num_agents
            if self.first256:
                branch_inputs[self.exo_branch_idx] = observations[..., : self.exo_obs_dim]
            if self.exo_obs:
                branch_inputs[self.exo_branch_idx] = self._build_exo_obs(observations)
            branch_inputs[self.exo_branch_idx] = self._add_exo_obs_noise(
                branch_inputs[self.exo_branch_idx]
            )

            return [
                branch(obs_i)
                for branch, obs_i in zip(self.actor_branches, branch_inputs)
            ]
        obs_chunks = list(torch.split(observations, self.agent_obs_dims, dim=-1))
        obs_chunks[self.exo_branch_idx] = self._add_exo_obs_noise(
            obs_chunks[self.exo_branch_idx]
        )
        return [
            branch(obs_i)
            for branch, obs_i in zip(self.actor_branches, obs_chunks)
        ]

    def _compose_env_actions(self, branch_tensors):
        return torch.cat(
            [branch_tensors[branch_idx] for branch_idx in self.output_branch_order], dim=-1
        )

    def _split_env_actions_by_branch(self, actions: torch.Tensor):
        env_chunks = torch.split(actions, self.output_action_dims, dim=-1)
        branch_chunks = [None] * self.num_agents
        for chunk, branch_idx in zip(env_chunks, self.output_branch_order):
            branch_chunks[branch_idx] = chunk
        return branch_chunks

    def update_distribution(self, observations: torch.Tensor):
        means = self._compute_actor_means(observations)
        mean = self._compose_env_actions(means)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actions_log_prob_per_agent(self, actions: torch.Tensor) -> torch.Tensor:
        logp = self.distribution.log_prob(actions)
        branch_chunks = self._split_env_actions_by_branch(logp)
        return torch.stack([chunk.sum(dim=-1) for chunk in branch_chunks], dim=-1)

    def act_inference(self, observations):
        return self._compose_env_actions(self._compute_actor_means(observations))

    def evaluate_per_agent(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value.expand(-1, self.num_agents)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def _linear_indices(self, module: nn.Sequential) -> list[int]:
        return [idx for idx, layer in enumerate(module) if isinstance(layer, nn.Linear)]

    def _copy_if_shape_matches(self, current_state, target_key, tensor):
        if target_key in current_state and current_state[target_key].shape == tensor.shape:
            current_state[target_key] = tensor
            return True
        return False

    def _load_log_std_into_human_branch(self, current_state, legacy_log_std: torch.Tensor | None):
        if legacy_log_std is None:
            return

        legacy_vec = legacy_log_std.detach()
        if legacy_vec.ndim > 1:
            legacy_vec = legacy_vec.reshape(-1)

        if legacy_vec.numel() == self.log_std.numel():
            current_state["log_std"] = legacy_vec.clone()
            return

        if legacy_vec.numel() < self.human_action_dim:
            print(
                f"ActorCriticMultiAgent: skipping legacy log_std load due to shape mismatch "
                f"({tuple(legacy_log_std.shape)} is smaller than human dim {self.human_action_dim})"
            )
            return

        human_slice = self._output_action_slice(self.human_branch_idx)
        current_state["log_std"][human_slice] = legacy_vec[: self.human_action_dim].clone()

    def _remap_old_branch_order_log_std(self, current_state, legacy_log_std: torch.Tensor | None):
        if legacy_log_std is None:
            return

        legacy_vec = legacy_log_std.detach()
        if legacy_vec.ndim > 1:
            legacy_vec = legacy_vec.reshape(-1)
        if legacy_vec.numel() != self.log_std.numel():
            return

        old_chunks = []
        start = 0
        for dim in self.agent_action_dims:
            end = start + dim
            old_chunks.append(legacy_vec[start:end])
            start = end

        current_state["log_std"] = self._compose_env_actions(old_chunks).clone()

    def _load_legacy_actor_branch(self, current_state, state_dict):
        branch_prefix = f"actor_branches.{self.human_branch_idx}.net."
        for key, tensor in state_dict.items():
            if not key.startswith("actor."):
                continue
            suffix = key[len("actor.") :]
            self._copy_if_shape_matches(current_state, branch_prefix + suffix, tensor)

    def _load_legacy_backbone_head_branch(self, current_state, state_dict):
        branch_net = self.actor_branches[self.human_branch_idx].net
        target_linear_indices = self._linear_indices(branch_net)

        source_backbone_indices = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict.keys()
                if key.startswith("backbone.") and key.endswith((".weight", ".bias"))
            }
        )
        if len(source_backbone_indices) + 1 != len(target_linear_indices):
            print(
                "ActorCriticMultiAgent: skipping legacy backbone/action_head load because the MLP depth does not match."
            )
            return

        for src_idx, tgt_idx in zip(source_backbone_indices, target_linear_indices[:-1]):
            for suffix in ("weight", "bias"):
                src_key = f"backbone.{src_idx}.{suffix}"
                tgt_key = f"actor_branches.{self.human_branch_idx}.net.{tgt_idx}.{suffix}"
                tensor = state_dict.get(src_key)
                if tensor is not None:
                    self._copy_if_shape_matches(current_state, tgt_key, tensor)

        for suffix in ("weight", "bias"):
            src_key = f"action_head.{suffix}"
            tgt_key = f"actor_branches.{self.human_branch_idx}.net.{target_linear_indices[-1]}.{suffix}"
            tensor = state_dict.get(src_key)
            if tensor is not None:
                self._copy_if_shape_matches(current_state, tgt_key, tensor)

    def _load_critic_sequential(self, current_state, state_dict):
        for key, tensor in state_dict.items():
            if key.startswith("critic."):
                self._copy_if_shape_matches(current_state, key, tensor)

    def _load_critic_backbone_head(self, current_state, state_dict):
        target_linear_indices = self._linear_indices(self.critic)
        source_backbone_indices = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict.keys()
                if key.startswith("critic_backbone.") and key.endswith((".weight", ".bias"))
            }
        )
        if not source_backbone_indices:
            return

        if len(source_backbone_indices) + 1 != len(target_linear_indices):
            print(
                "ActorCriticMultiAgent: skipping native critic remap because the critic MLP depth does not match."
            )
            return

        for src_idx, tgt_idx in zip(source_backbone_indices, target_linear_indices[:-1]):
            for suffix in ("weight", "bias"):
                src_key = f"critic_backbone.{src_idx}.{suffix}"
                tgt_key = f"critic.{tgt_idx}.{suffix}"
                tensor = state_dict.get(src_key)
                if tensor is not None:
                    self._copy_if_shape_matches(current_state, tgt_key, tensor)

        head_weight = state_dict.get("critic_head.weight")
        head_bias = state_dict.get("critic_head.bias")
        last_linear_idx = target_linear_indices[-1]
        target_weight_key = f"critic.{last_linear_idx}.weight"
        target_bias_key = f"critic.{last_linear_idx}.bias"

        if head_weight is not None:
            target_weight = current_state[target_weight_key]
            if head_weight.shape == target_weight.shape:
                current_state[target_weight_key] = head_weight
            elif head_weight.ndim == 2 and head_weight.shape[1] == target_weight.shape[1]:
                current_state[target_weight_key] = head_weight.mean(dim=0, keepdim=True)
        if head_bias is not None:
            target_bias = current_state[target_bias_key]
            if head_bias.shape == target_bias.shape:
                current_state[target_bias_key] = head_bias
            elif head_bias.ndim == 1 and target_bias.numel() == 1:
                current_state[target_bias_key] = head_bias.mean().reshape_as(target_bias)

    def load_state_dict(self, state_dict, strict=True):
        """Load a checkpoint.

        Supports:
            - Native multi-agent checkpoints
            - Previous multi-agent checkpoints using `critic_backbone`/`critic_head`
            - Legacy single-agent checkpoints (loads the actor into the configured human branch)
        """
        # Normalize key names (remove DataParallel prefixes)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        current_state = self.state_dict()
        checkpoint_layout_version = state_dict.get("_action_layout_version")

        if any(k.startswith("actor.") for k in state_dict.keys()):
            self._load_critic_sequential(current_state, state_dict)
            self._load_legacy_actor_branch(current_state, state_dict)
            self._load_log_std_into_human_branch(current_state, state_dict.get("log_std"))
            super().load_state_dict(current_state, strict=False)
            return True

        if "action_head.weight" in state_dict and any(k.startswith("backbone.") for k in state_dict.keys()):
            self._load_critic_sequential(current_state, state_dict)
            self._load_legacy_backbone_head_branch(current_state, state_dict)
            self._load_log_std_into_human_branch(current_state, state_dict.get("log_std"))
            super().load_state_dict(current_state, strict=False)
            return True

        if any(k.startswith("critic_backbone.") for k in state_dict.keys()) or "critic_head.weight" in state_dict:
            remapped_state = current_state.copy()
            for key, tensor in state_dict.items():
                if key.startswith("actor_branches."):
                    remapped_state[key] = tensor
            if "log_std" in state_dict:
                if checkpoint_layout_version is None:
                    self._remap_old_branch_order_log_std(remapped_state, state_dict["log_std"])
                else:
                    remapped_state["log_std"] = state_dict["log_std"]
            self._load_critic_backbone_head(remapped_state, state_dict)
            super().load_state_dict(remapped_state, strict=False)
            return True

        if checkpoint_layout_version is None and "log_std" in state_dict and any(
            k.startswith("actor_branches.") for k in state_dict.keys()
        ):
            remapped_state = current_state.copy()
            for key, tensor in state_dict.items():
                if key != "log_std":
                    remapped_state[key] = tensor
            self._remap_old_branch_order_log_std(remapped_state, state_dict["log_std"])
            super().load_state_dict(remapped_state, strict=False)
            return True

        super().load_state_dict(state_dict, strict=strict)
        return True
