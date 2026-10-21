import numpy as np
import torch
import gymnasium as gym
from RL.env.vec_env import VecEnv
from RL.env.observations import build_critic_observations
from stable_baselines3.common.vec_env import VecEnv as SB3VecEnv

idx_parts = [
        [4],
        list(range(30, 33)),
        list(range(45, 48)),
        list(range(66, 69)),
        list(range(75, 78)),
        list(range(164, 210)),
        list(range(18, 78))
    ]

def get_amp(obs):

    idx = [i for part in idx_parts for i in part]  # 展开索引

    # torch.Tensor 情况
    if isinstance(obs, torch.Tensor):
        amp = obs[:, idx]
        assert amp.shape[1] == 119, f"Output dim mismatch: got {amp.shape[1]} expected 119"
        return amp

    # numpy.ndarray 情况
    elif isinstance(obs, np.ndarray):
        amp = obs[:, idx]
        assert amp.shape[1] == 119, f"Output dim mismatch: got {amp.shape[1]} expected 119"
        return amp

    else:
        raise TypeError(f"Unsupported type: {type(obs)}. Expected torch.Tensor or np.ndarray.")

class SB3RslVecEnv(VecEnv):
    """Wraps a Stable-Baselines3 VecEnv to match rsl_rl.env.VecEnv API."""

    def __init__(
        self,
        sb3_vec_env: SB3VecEnv,
        clip_actions: float = None,
        is_finite_horizon: bool = False,
        device: str = "cpu",
        use_amp: bool = False,
    ):
        # underlying SB3 vectorized env
        self.env = sb3_vec_env
        self.num_envs = self.env.num_envs
        self.clip_actions = clip_actions
        self.device = torch.device(device)

        # build a fake cfg object for RSL-RL
        self._cfg = type("Cfg", (), {"is_finite_horizon": is_finite_horizon})

        # gym spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # dims
        self.num_obs = gym.spaces.flatdim(self.observation_space)
        self.num_actions = gym.spaces.flatdim(self.action_space)
        self.num_privileged_obs = 0

        # episode bookkeeping
        self.max_episode_length = getattr(self.env, "max_episode_steps", None) or 1000
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # AMP meta
        self._use_amp = use_amp
        self._amp_pos_dim = 119
        self._amp_dt = 0.02

        # placeholders
        self._obs = None
        self._obs_tem = None
        self._prev_amp_positions = None
        self._critic_obs = None
        self.reset()

    @property
    def cfg(self) -> object:
        return self._cfg

    def get_observations(self):
        obs_tensor = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
        extras = {"observations": {"policy": obs_tensor}}
        if self._critic_obs is not None:
            extras['observations']['critic'] = torch.tensor(
                self._critic_obs, dtype=torch.float32, device=self.device
            )

        if self._use_amp:
            prev_amp = torch.tensor(
                self._prev_amp_positions, dtype=torch.float32, device=self.device
            )
            amp_pos = get_amp(obs_tensor)
            amp_features = self._compose_amp_features(amp_pos, prev_amp)
            extras["observations"]["amp"] = amp_features
            extras["observations"]["amp_term"] = amp_features

        return obs_tensor, extras

    def reset(self):
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        self._obs = np.array(obs)
        self._obs_tem = self._obs.copy()
        self._update_critic_observations(self.env.reset_infos)
        if self._use_amp:
            self._prev_amp_positions = get_amp(self._obs).copy()
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_np = actions.cpu().numpy()

        out = self.env.step(actions_np)
        obs_batch, reward_batch, done_batch, info_batch = out[:4]
        obs_term = obs_batch.copy()
        reset_idxs = np.nonzero(done_batch)[0]
        for i in reset_idxs:
            term_obs = info_batch[i].get("terminal_observation")
            if term_obs is not None:
                obs_term[i] = term_obs

        self._obs = np.array(obs_batch)
        self._obs_tem = np.array(obs_term)
        # SB3 returns reset observations for done envs; pair them with reset
        # privileged state rather than the terminal state's info payload.
        current_infos = [self.env.reset_infos[i] if done_batch[i] else info_batch[i]
                         for i in range(self.num_envs)]
        self._update_critic_observations(current_infos)

        obs_tensor = torch.tensor(self._obs, dtype=torch.float32, device=self.device)
        obs_term_tensor = torch.tensor(
            self._obs_tem, dtype=torch.float32, device=self.device
        )
        rewards_tensor = torch.tensor(reward_batch, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(done_batch, dtype=torch.long, device=self.device)

        extras = {"observations": {"policy": obs_tensor}}
        if self._critic_obs is not None:
            extras['observations']['critic'] = torch.tensor(
                self._critic_obs, dtype=torch.float32, device=self.device
            )

        if self._use_amp:
            amp_pos = get_amp(obs_tensor)
            amp_pos_term = get_amp(obs_term_tensor)
            prev_amp = torch.tensor(
                self._prev_amp_positions, dtype=torch.float32, device=self.device
            )

            amp_term_features = self._compose_amp_features(amp_pos_term, prev_amp)

            prev_for_curr = prev_amp.clone()
            done_mask = dones_tensor.bool()
            if done_mask.any():
                prev_for_curr[done_mask] = amp_pos[done_mask]
            amp_features = self._compose_amp_features(amp_pos, prev_for_curr)

            self._prev_amp_positions = amp_pos.cpu().numpy()

            extras["observations"]["amp"] = amp_features
            extras["observations"]["amp_term"] = amp_term_features

        if not self.cfg.is_finite_horizon:
            truncated_flags = [
                info.get("TimeLimit.truncated", False)
                for info in info_batch
            ]
            extras["time_outs"] = torch.tensor(
                truncated_flags, dtype=torch.bool, device=self.device
            )
        return obs_tensor, rewards_tensor, dones_tensor, extras

    def _update_critic_observations(self, infos):
        critic = build_critic_observations(self._obs, infos)
        dim = 0 if critic is None else critic.shape[-1]
        if self.num_privileged_obs and dim != self.num_privileged_obs:
            raise ValueError('Critic observation dimension changed during rollout')
        self._critic_obs = critic
        self.num_privileged_obs = dim

    def close(self):
        self.env.close()

    def seed(self, seed: int = -1) -> int:
        try:
            self.env.seed(seed)
        except AttributeError:
            self.env.reset(seed=seed)
        return seed

    def _compose_amp_features(
        self, positions: torch.Tensor, prev_positions: torch.Tensor
    ) -> torch.Tensor:
        pelvis = positions[:, 0:1]
        pelvis_prev = prev_positions[:, 0:1]
        pelvis_vel = (pelvis - pelvis_prev) / self._amp_dt

        endpoints = positions[:, 1:13]

        joint = positions[:, 13:59]
        joint_prev = prev_positions[:, 13:59]
        joint_vel = (joint - joint_prev) / self._amp_dt

        limbs = positions[:, 59:119]
        limb_prev = prev_positions[:, 59:119]
        limb_vel = (limbs - limb_prev) / self._amp_dt

        return torch.cat(
            (pelvis_vel, endpoints, joint, joint_vel, limb_vel),
            dim=1,
        )
