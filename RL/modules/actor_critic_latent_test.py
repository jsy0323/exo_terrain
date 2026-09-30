# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import copy
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from ..utils import resolve_nn_activation


class TerrainAttentionEncoder(nn.Module):
    """CNN terrain encoder with base-observation-conditioned spatial attention."""

    def __init__(self, patch_shape, feature_dim: int, base_obs_dim: int, activation: nn.Module):
        super().__init__()
        h, w = patch_shape
        self.num_cells = int(h * w)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            copy.deepcopy(activation),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            copy.deepcopy(activation),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(int(base_obs_dim), 16),
            nn.Sigmoid(),
        )
        self.attention_logits = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=1),
            copy.deepcopy(activation),
            nn.Conv2d(8, 1, kernel_size=1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * h * w, int(feature_dim)),
            copy.deepcopy(activation),
        )
        self.last_attention = None

    def forward(self, terrain_img, base_obs):
        features = self.conv(terrain_img)
        gate = self.context_gate(base_obs).view(base_obs.shape[0], 16, 1, 1)
        gated_features = features * (1.0 + gate)

        weights = torch.softmax(self.attention_logits(gated_features).flatten(2), dim=-1)
        weights = weights.view(terrain_img.shape[0], 1, terrain_img.shape[2], terrain_img.shape[3])
        self.last_attention = weights.detach()

        attended_features = gated_features * (weights * self.num_cells)
        return self.projection(attended_features)


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        alpha=1.0,
        noise_std_type: str = "scalar",
        terrain_cnn_enabled: bool = False,
        terrain_patch_shape=None,
        terrain_feature_dim: int = 64,
        terrain_attention_enabled: bool = False,
        **kwargs,
    ):
    
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        self.terrain_cnn_enabled = bool(terrain_cnn_enabled)
        self.terrain_patch_shape = tuple(terrain_patch_shape or (0, 0))
        self.terrain_obs_dim = int(self.terrain_patch_shape[0] * self.terrain_patch_shape[1])
        self.terrain_feature_dim = int(terrain_feature_dim)
        self.terrain_attention_enabled = bool(terrain_attention_enabled)
        if self.terrain_cnn_enabled:
            if self.terrain_obs_dim <= 0:
                raise ValueError("terrain_patch_shape must be set when terrain_cnn_enabled=True")
            if num_actor_obs <= self.terrain_obs_dim:
                raise ValueError(
                    f"num_actor_obs={num_actor_obs} must be larger than terrain_obs_dim={self.terrain_obs_dim}"
                )
            self.base_actor_obs_dim = int(num_actor_obs - self.terrain_obs_dim)
            h, w = self.terrain_patch_shape
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
                    nn.Linear(16 * h * w, self.terrain_feature_dim),
                    copy.deepcopy(activation),
                )
            mlp_input_dim_a = self.base_actor_obs_dim + self.terrain_feature_dim
        else:
            self.base_actor_obs_dim = int(num_actor_obs)
            self.terrain_encoder = None
            mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy 
        self.backbone = nn.Sequential(
            nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]),
            activation,
            *sum([[nn.Linear(h_in, h_out), activation]
                  for h_in,h_out in zip(actor_hidden_dims[:-1],actor_hidden_dims[1:])], [])
        )
        self.action_head = nn.Linear(actor_hidden_dims[-1], num_actions)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0) 

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        if self.terrain_cnn_enabled:
            print(f"Terrain CNN: {self.terrain_encoder}")
            print(f"Terrain attention enabled: {self.terrain_attention_enabled}")
        print(f"Backbone MLP: {self.backbone}")
        print(f"Critic MLP: {self.critic}")

        # std
        # std 需要缩放
        self.action_dim = num_actions
        self.latent_dim = actor_hidden_dims[-1]
        self.log_std = nn.Parameter(
            torch.ones(1, self.action_dim + self.latent_dim) * -1.6,
            requires_grad=True
            )
        self.alpha = alpha
        self._eps = 1e-6                
        self.distribution = None
        MultivariateNormal.set_default_validate_args(False) 

    def _actor_features(self, observations):
        if not self.terrain_cnn_enabled:
            return observations
        base_obs = observations[:, : self.base_actor_obs_dim]
        terrain_obs = observations[:, self.base_actor_obs_dim : self.base_actor_obs_dim + self.terrain_obs_dim]
        h, w = self.terrain_patch_shape
        terrain_img = terrain_obs.reshape(observations.shape[0], 1, h, w)
        if self.terrain_attention_enabled:
            terrain_feat = self.terrain_encoder(terrain_img, base_obs)
        else:
            terrain_feat = self.terrain_encoder(terrain_img)
        return torch.cat((base_obs, terrain_feat), dim=-1)

    @staticmethod
    # not used at the moment
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

    def update_distribution(self, observations,step_idx=0):
        """
        根据给定观测更新动作分布；核心与 PolicyLattice 保持一致。
        """
        actor_obs = self._actor_features(observations)
        h = self.backbone(actor_obs)                    # (B, latent_dim)
        a_mean = self.action_head(h)                    # (B, action_dim)

        # 1) 取出 std 向量并拆分
        std = torch.exp(self.log_std)                   # (1, action_dim + latent_dim)
        action_std  = std[:, :self.action_dim]          # (1, A)
        latent_std  = std[:, self.action_dim:]          # (1, L)

        action_var  = action_std.pow(2)                 # (1, A)
        latent_var  = latent_std.pow(2)                 # (1, L)

        W = self.action_head.weight                     # (A, L)
        sigma_mat = (W * latent_var[..., None, :]).matmul(W.t())  # (B, A, A)
        sigma_mat[..., torch.arange(self.action_dim), torch.arange(self.action_dim)] += action_var

        # 3) 保存分布
        self.distribution = MultivariateNormal(a_mean, sigma_mat)

        return self.distribution

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions)

    def act_inference(self, observations):
        actions_mean = self.action_head(self.backbone(self._actor_features(observations)))
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        own_state = self.state_dict()
        adapted_state = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in own_state:
                skipped.append(key)
                continue

            target = own_state[key]
            if tuple(value.shape) == tuple(target.shape):
                adapted_state[key] = value
            elif key == "backbone.0.weight" and value.ndim == 2 and target.ndim == 2:
                copied = target.clone()
                rows = min(copied.shape[0], value.shape[0])
                cols = min(copied.shape[1], value.shape[1])
                copied[:rows, :cols] = value[:rows, :cols]
                adapted_state[key] = copied
                print(
                    f"Partially loaded {key}: old {tuple(value.shape)} -> new {tuple(target.shape)}"
                )
            elif key == "critic.0.weight" and value.ndim == 2 and target.ndim == 2:
                copied = target.clone()
                rows = min(copied.shape[0], value.shape[0])
                cols = min(copied.shape[1], value.shape[1])
                copied[:rows, :cols] = value[:rows, :cols]
                adapted_state[key] = copied
                print(
                    f"Partially loaded {key}: old {tuple(value.shape)} -> new {tuple(target.shape)}"
                )
            else:
                skipped.append(key)

        missing, unexpected = super().load_state_dict(adapted_state, strict=False)
        if skipped:
            print(f"Skipped incompatible checkpoint tensors: {skipped}")
        if missing:
            print(f"New tensors initialized from scratch: {missing}")
        if unexpected:
            print(f"Unexpected tensors while loading checkpoint: {unexpected}")
        return True
