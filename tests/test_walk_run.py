"""Regressions for command timing, reward routing and pretrained policy migration."""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from Env.MFG_Musculoskelet_V11.mfgenv.speed_command import SpeedCommand
from Env.MFG_Musculoskelet_V11.mfgenv.ReferTraj_V8 import TrajectoryManager, ReferenceTrajectories
from Env.MFG_Musculoskelet_V11.mfgenv.mfg_MSenv import MFG_Musculoskeletal_V11
from Env.MFG_Musculoskelet_V11.mfgenv.reward import compute_reward, get_goal_reward
from RL.modules.actor_critic_latent_test import ActorCritic
from RL.modules.normalizer import EmpiricalNormalization
from RL.runners.on_policy_runner import OnPolicyRunner


class WalkRunTests(unittest.TestCase):
    def test_ramp_endpoints_and_acceleration(self):
        for acceleration in (0.3, 0.65, 1.0):
            command = SpeedCommand({"acceleration_range": [acceleration, acceleration]})
            command.reset(np.random.default_rng(42))
            command.update(1.0)
            self.assertEqual(command.speed, 1.4)
            self.assertEqual(command.acceleration, 0.0)
            command.update(2.5)
            self.assertAlmostEqual(command.speed, 1.4 + acceleration * 0.5)
            self.assertEqual(command.acceleration, acceleration)
            command.update(20.0)
            self.assertEqual(command.speed, 2.8)
            self.assertEqual(command.acceleration, 0.0)
            previous_speed = 1.4
            for time in np.arange(0.02, 10.01, 0.02):
                command.update(time)
                self.assertLessEqual(abs(command.speed - previous_speed), acceleration * 0.02 + 1e-12)
                previous_speed = command.speed

    def test_invalid_command_values(self):
        for config in ({"end_speed": -1.0}, {"hold_seconds": -1},
                       {"acceleration_range": [1, 0.3]}, {"start_speed": float("nan")}):
            with self.assertRaises(ValueError):
                SpeedCommand(config)

    def test_speed_reward_in_imitation_and_fixed_direction(self):
        env = SimpleNamespace(
            phase=SimpleNamespace(name="IMITATION"),
            model=SimpleNamespace(body=lambda name: SimpleNamespace(id=0)),
            data=SimpleNamespace(xpos=np.zeros((1, 3)), qvel=np.array([1.4, 0.0, 0.0])),
            reward_weights={"goal": 1.0},
            goal_weights={"position": 0.0, "speed": 1.0},
            goal_reward={"speed_sigma": 0.5, "lateral_speed_sigma": 0.35, "speed_direction": "command"},
            speed_command=SpeedCommand({}),
            speed_command_direction=np.array([1.0, 0.0, 0.0]),
        )
        reward, info = compute_reward(env, {})
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["goal"]["speed"]["target_speed"], 1.4)
        env.data.qvel[1] = 0.35
        self.assertAlmostEqual(get_goal_reward(env)[0], np.exp(-1))
        env.data.qvel[:2] = [0, 1.4]
        self.assertLess(get_goal_reward(env)[0], 1e-3)

    def test_checkpoint_added_inputs_preserve_actions(self):
        with redirect_stdout(StringIO()):
            original = ActorCritic(5, 5, 3, actor_hidden_dims=[8, 4], critic_hidden_dims=[8, 4])
            expanded = ActorCritic(7, 7, 3, actor_hidden_dims=[8, 4], critic_hidden_dims=[8, 4],
                                   zero_initialize_added_observations=True)
            expanded.load_state_dict(original.state_dict())
        obs = torch.randn(4, 5)
        expanded_obs = torch.cat((obs, torch.randn(4, 2)), dim=-1)
        torch.testing.assert_close(original.act_inference(obs), expanded.act_inference(expanded_obs))
        torch.testing.assert_close(original.evaluate(obs), expanded.evaluate(expanded_obs))
        old_norm = EmpiricalNormalization([5])
        old_norm.update(torch.randn(10, 5))
        new_norm = EmpiricalNormalization([7])
        with redirect_stdout(StringIO()):
            OnPolicyRunner._load_normalizer_state_flexible(
                new_norm, old_norm.state_dict(), "test", preserve_count_on_resize=True)
        torch.testing.assert_close(new_norm._mean[:, :5], old_norm._mean)
        self.assertEqual(new_norm.count.item(), old_norm.count.item())
        torch.testing.assert_close(new_norm._std[:, 5:], torch.ones(1, 2))

    def test_real_environment_commands_and_reference_loop(self):
        with open("Env/MFG_Musculoskelet_V11/cfg/mujocoenv_default.yaml", encoding="utf8") as handle:
            base = yaml.safe_load(handle)
        with open("configs/humanoid_walk_run.yaml", encoding="utf8") as handle:
            config = yaml.safe_load(handle)
        base = OmegaConf.to_container(OmegaConf.merge(base, config["env"]["config"]), resolve=True)
        manager = TrajectoryManager(**base["traj_manager"])
        base["traj_manager"] = manager
        env = MFG_Musculoskeletal_V11(base)
        try:
            obs, _ = env.reset(seed=42)
            self.assertEqual(obs.shape, (471,))
            self.assertEqual(env.action_space.shape, (106,))
            self.assertEqual(env.model.nhfield, 0)
            self.assertEqual(env.num_critic_extra_obs, 0)
            np.testing.assert_array_equal(obs[-2:], [np.float32(1.4), 0])
            self.assertAlmostEqual(env.opt_time * env.frame_skip, 0.02)
            env.step_count = 100
            obs, _, _, truncated, info = env.step(np.zeros(106))
            self.assertFalse(truncated)
            self.assertGreater(obs[-2], 1.4)
            self.assertGreater(obs[-1], 0)
            self.assertTrue(info["reward_info"]["terrain"]["skipped"])
            self.assertTrue(info["reward_info"]["com"]["skipped"])
            self.assertGreater(info["reward_info"]["goal"]["speed"]["target_speed"], 1.4)
            env.reset(seed=42)
            first_acceleration = env.speed_command.ramp_acceleration
            env.reset(seed=42)
            self.assertEqual(env.speed_command.ramp_acceleration, first_acceleration)

            ref = ReferenceTrajectories(manager, traj_id=0, loop=True)
            ref.reset(phase=0.0, randomize_traj=False)
            ref._pos = ref.step_frames - ref.increment
            expected_qpos, expected_qvel = ref.peek(ref.increment)
            window_qpos, window_qvel = ref.get_reference_window(start_steps=1)
            np.testing.assert_allclose(window_qpos[0], expected_qpos)
            np.testing.assert_allclose(window_qvel[0], expected_qvel)
            ref.next()
            np.testing.assert_allclose(ref.get_qpos(), expected_qpos)
            self.assertFalse(ref.has_reached_end)
            legacy = ReferenceTrajectories(manager, traj_id=0)
            legacy._pos = legacy.traj_frames - 2
            legacy.next()
            self.assertTrue(legacy.has_reached_end)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
