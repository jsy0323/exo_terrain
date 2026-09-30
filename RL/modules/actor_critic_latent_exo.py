# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from ..utils import resolve_nn_activation


class ActorCritic(nn.Module):
    """Actor-critic with two extra action dimensions.

    The network splits the action head into:
      - actor1: matches the legacy action dimensions so its weights can be
        initialized from an existing policy.
      - actor2: produces the two new action dimensions that will be trained
        from scratch.
    Both heads share the same latent backbone and critic.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 0.1,
        alpha: float = 1.0,
        noise_std_type: str = "scalar",
        extra_action_dims: int = 2,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticExo.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        if num_actions <= extra_action_dims:
            raise ValueError(
                f"num_actions ({num_actions}) must be larger than extra_action_dims ({extra_action_dims})."
            )

        activation = resolve_nn_activation(activation)
        self.extra_action_dim = extra_action_dims
        self.base_action_dim = num_actions - extra_action_dims
        self.action_dim = num_actions

        # Policy backbone shared by both action heads.
        self.backbone = nn.Sequential(
            nn.Linear(num_actor_obs, actor_hidden_dims[0]),
            activation,
            *sum(
                [
                    [nn.Linear(h_in, h_out), activation]
                    for h_in, h_out in zip(actor_hidden_dims[:-1], actor_hidden_dims[1:])
                ],
                [],
            ),
        )
        self.action_head_main = nn.Linear(actor_hidden_dims[-1], self.base_action_dim)
        self.action_head_exo = nn.Linear(actor_hidden_dims[-1], self.extra_action_dim)
        for head in (self.action_head_main, self.action_head_exo):
            head.weight.data.mul_(0.1)
            head.bias.data.mul_(0.0)

        # Value function.
        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation]
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Backbone MLP: {self.backbone}")
        print(f"Critic MLP: {self.critic}")

        # Std parameters (split so legacy checkpoints can populate the main head and latent part).
        self.latent_dim = actor_hidden_dims[-1]
        self.log_std_actions_main = nn.Parameter(torch.ones(1, self.base_action_dim) * -1.6, requires_grad=True)
        self.log_std_actions_exo = nn.Parameter(torch.ones(1, self.extra_action_dim) * -1.6, requires_grad=True)
        self.log_std_latent = nn.Parameter(torch.ones(1, self.latent_dim) * -1.6, requires_grad=True)

        self.alpha = alpha
        self._eps = 1e-6
        self.distribution = None
        MultivariateNormal.set_default_validate_args(False)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

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
        return self.distribution.entropy()

    def update_distribution(self, observations, step_idx: int = 0):
        """Update the full action distribution."""
        h = self.backbone(observations)  # (B, latent_dim)
        a_mean_main = self.action_head_main(h)  # (B, base_action_dim)
        a_mean_exo = self.action_head_exo(h)  # (B, extra_action_dim)
        a_mean = torch.cat([a_mean_main, a_mean_exo], dim=-1)  # (B, action_dim)

        action_var_main = torch.exp(self.log_std_actions_main).pow(2)
        action_var_exo = torch.exp(self.log_std_actions_exo).pow(2)
        latent_var = torch.exp(self.log_std_latent).pow(2)

        W_main = self.action_head_main.weight  # (base_action_dim, latent_dim)
        W_exo = self.action_head_exo.weight  # (extra_action_dim, latent_dim)
        W_full = torch.cat([W_main, W_exo], dim=0)  # (action_dim, latent_dim)
        sigma_mat = (W_full * latent_var[..., None, :]).matmul(W_full.t())  # (1, action_dim, action_dim)

        diag_noise = torch.cat([action_var_main, action_var_exo], dim=-1)
        sigma_mat[..., torch.arange(self.action_dim), torch.arange(self.action_dim)] += diag_noise

        self.distribution = MultivariateNormal(a_mean, sigma_mat)
        return self.distribution

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions)

    def act_inference(self, observations):
        h = self.backbone(observations)
        mean_main = self.action_head_main(h)
        mean_exo = self.action_head_exo(h)
        return torch.cat([mean_main, mean_exo], dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load parameters, accepting legacy ActorCritic checkpoints."""
        # Direct load for native exo checkpoints
        if "action_head_main.weight" in state_dict:
            super().load_state_dict(state_dict, strict=strict)
            return True
        # Legacy (ActorCriticLatentTest) checkpoint
        legacy_head_w = state_dict.get("action_head.weight")
        legacy_head_b = state_dict.get("action_head.bias") 
        legacy_log_std = state_dict.get("log_std")

        if legacy_head_w is None or legacy_head_b is None:
            super().load_state_dict(state_dict, strict=strict)
            return True

        current_state = self.state_dict()
        for key, tensor in state_dict.items():
            clean_key = key.replace("module.", "")
            if clean_key.startswith("backbone.") or clean_key.startswith("critic."):
                if clean_key in current_state and current_state[clean_key].shape == tensor.shape:
                    current_state[clean_key] = tensor

        current_state["action_head_main.weight"] = legacy_head_w
        current_state["action_head_main.bias"] = legacy_head_b

        if legacy_log_std is not None:
            legacy_action_dim = legacy_log_std.shape[1] - self.latent_dim
            if legacy_action_dim >= self.base_action_dim:
                current_state["log_std_actions_main"] = legacy_log_std[:, : self.base_action_dim].clone()
                current_state["log_std_latent"] = legacy_log_std[:, -self.latent_dim :].clone()
            else:
                print(
                    f"ActorCriticExo: skipping legacy log_std load due to shape mismatch "
                    f"({legacy_log_std.shape} vs expected {(1, self.base_action_dim + self.latent_dim)})"
                )

        super().load_state_dict(current_state, strict=False)
        return True
