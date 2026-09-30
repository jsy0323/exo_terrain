import numpy as np
import gymnasium as gym
import torch
from RL.env import VecEnv


class GymMujocoWrapper(VecEnv):
    """
    A wrapper for Gymnasium MuJoCo environments that adapts them to the interface expected by RSL-RL.
    """

    def __init__(
        self,
        env,
        clip_actions: float = 10.0,
        is_finite_horizon: bool = True,
        auto_reset: bool = True,
        device: str = "cpu",
    ):
        if isinstance(env, list):
            self.envs = env
        else:
            self.envs = [env]
        self.num_envs = len(self.envs)
        self.clip_actions = clip_actions
        self.auto_reset = auto_reset
        self.device = torch.device(device)

        self._cfg = type("Cfg", (), {"is_finite_horizon": is_finite_horizon})
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.num_actions = gym.spaces.flatdim(self.action_space)
        self.num_obs = gym.spaces.flatdim(self.observation_space)
        self.num_privileged_obs = 0

        self.max_episode_length = getattr(self.envs[0].spec, "max_episode_steps", 1000)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._obs = None
        self._modify_action_space()
        self.reset()

    @property
    def cfg(self) -> object:
        return self._cfg

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        obs_tensor = torch.tensor(self._obs, dtype=torch.float, device=self.device)
        return obs_tensor, {"observations": {"policy": obs_tensor}}

    def reset(
        self,
        id=None,
        frame=None,
        seed=None,
        terrain_seed=None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset wrapped environments, optionally with a reproducible terrain.

        ``seed`` is forwarded through Gymnasium's normal reset path.  The
        musculoskeletal environment keeps a separate RNG for terrain
        generation, so ``terrain_seed`` resets that generator explicitly.
        """
        obs_list = []
        for env in self.envs:
            if terrain_seed is not None:
                base_env = env.unwrapped
                if not hasattr(base_env, "_terrain_rng"):
                    raise AttributeError(
                        "terrain_seed was provided, but the wrapped environment "
                        "does not expose a terrain RNG."
                    )
                base_env._terrain_rng = np.random.default_rng(int(terrain_seed))
            obs, _ = env.reset(seed=seed, traj_id=id, frame=frame)
            obs_list.append(obs)
        self._obs = np.stack(obs_list, axis=0)
        return self.get_observations()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_np = actions.cpu().numpy()

        obs_list, rewards, dones = [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, terminated, truncated, info = env.step(actions_np[i])
            done = terminated or truncated
            if done and self.auto_reset:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(rew)
            dones.append(done)

        self._obs = np.stack(obs_list, axis=0)
        obs_tensor = torch.tensor(self._obs, dtype=torch.float, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.long, device=self.device)


        return obs_tensor, rewards_tensor, dones_tensor, info

    def close(self):
        for env in self.envs:
            env.close()

    def seed(self, seed: int = -1) -> int:
        for env in self.envs:
            env.reset(seed=seed)
        return seed
    def speed(self):
        # import ipdb;ipdb.set_trace()
        return self.envs[0].unwrapped.ref_traj.speed,self.envs[0].unwrapped.ref_traj.traj_id
    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
            dtype=self.action_space.dtype,
        )
