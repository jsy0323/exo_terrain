"""Actor input isolation, automatic reset alignment and asymmetric PPO flow."""
import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env_wrapper import SB3RslVecEnv
from env_wrapper_eval import GymMujocoWrapper
from RL.env.observations import build_critic_observations
from RL.modules.actor_critic_terrainexo import ActorCriticTerrainExo
from RL.modules.normalizer import EmpiricalNormalization
from RL.runners.on_policy_runner import OnPolicyRunner


class ObservationProbe(gym.Env):
    """Episode IDs distinguish terminal privileged state from reset state."""
    def __init__(self, privileged=True):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (300,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (7,), np.float32)
        self.privileged = privileged
        self.episode = 0
        self.tick = 0

    def _observation(self):
        value = self.episode + self.tick / 10.
        obs = np.full(300, value, dtype=np.float32)
        info = {}
        if self.privileged:
            info['critic_extra_obs'] = np.full(48, value * 100, dtype=np.float32)
        return obs, info

    def reset(self, *, seed=None, options=None, traj_id=None, frame=None):
        super().reset(seed=seed)
        self.episode += 1
        self.tick = 0
        return self._observation()

    def step(self, action):
        self.tick += 1
        obs, info = self._observation()
        return obs, 1., self.tick >= 2, False, info


def make_probe():
    return ObservationProbe()


def policy_kwargs():
    return dict(actor_hidden_dims=(32, 16), critic_hidden_dims=(32, 16),
                exo_actor_hidden_dims=(16, 8), agent_action_dims=(2, 5),
                human_branch_idx=1, shared_actor_obs=True, first256=True,
                terrain_patch_shape=(4, 5), terrain_feature_dim=8,
                terrain_attention_enabled=True, muscle_action_dim=5,
                exo_action_dim=2, init_noise_std=.1)


def assert_paired(obs, extras):
    critic = extras['observations']['critic']
    assert obs.shape == (2, 300)
    assert critic.shape == (2, 348)
    torch.testing.assert_close(critic[:, :300], obs)
    torch.testing.assert_close(critic[:, 300:], (obs[:, :1] * 100).expand(-1, 48))


@pytest.mark.parametrize('subprocess', [False, True])
def test_sb3_critic_tail_matches_reset_observations(subprocess):
    sb3 = (SubprocVecEnv([make_probe, make_probe], start_method='spawn')
           if subprocess else DummyVecEnv([make_probe, make_probe]))
    env = SB3RslVecEnv(sb3, use_amp=True)
    try:
        obs, extras = env.get_observations()
        assert env.num_obs == 300
        assert env.num_privileged_obs == 348
        assert_paired(obs, extras)
        for _ in range(4):
            obs, _, _, extras = env.step(torch.zeros(2, 7))
            assert_paired(obs, extras)
            assert extras['observations']['amp'].shape[0] == 2
    finally:
        env.close()


def test_evaluation_wrapper_preserves_actor_inputs_and_reset_alignment():
    env = GymMujocoWrapper([make_probe(), make_probe()])
    try:
        obs, extras = env.get_observations()
        assert_paired(obs, extras)
        for _ in range(4):
            obs, _, _, extras = env.step(torch.zeros(2, 7))
            assert_paired(obs, extras)
    finally:
        env.close()


def test_legacy_environment_without_privileged_state_keeps_existing_api():
    env = SB3RslVecEnv(DummyVecEnv([lambda: ObservationProbe(privileged=False)]))
    try:
        obs, extras = env.get_observations()
        assert obs.shape == (1, 300)
        assert env.num_privileged_obs == 0
        assert 'critic' not in extras['observations']
    finally:
        env.close()


def test_missing_privileged_tail_cannot_silently_drop_a_worker():
    with pytest.raises(ValueError, match='Missing critic_extra_obs'):
        build_critic_observations(np.zeros((2, 300)),
                                  [{'critic_extra_obs': np.zeros(48)}, {}])


def test_both_actors_are_independent_of_critic_only_state():
    policy = ActorCriticTerrainExo(300, 348, 7, **policy_kwargs()).eval()
    obs = torch.randn(2, 300)
    critic = torch.cat((obs, torch.zeros(2, 48)), -1)
    changed = critic.clone()
    changed[:, 300:] = 1.
    # Make the value head depend explicitly on a privileged coordinate.
    with torch.no_grad():
        for p in policy.critic.parameters():
            p.zero_()
        linears = [layer for layer in policy.critic if isinstance(layer, torch.nn.Linear)]
        linears[0].weight[0, -1] = 1.
        for layer in linears[1:]:
            layer.weight[0, 0] = 1.
    before = policy.act_inference(obs).detach().clone()
    value = policy.evaluate(critic)
    changed_value = policy.evaluate(changed)
    assert not torch.equal(value, changed_value)
    torch.testing.assert_close(policy.act_inference(obs), before)
    changed_value.sum().backward()
    assert all(p.grad is None for p in policy.human_actor.parameters())
    assert all(p.grad is None for p in policy.exo_actor.parameters())


def test_old_checkpoint_keeps_both_actors_and_extends_only_critic():
    source = ActorCriticTerrainExo(300, 300, 7, **policy_kwargs()).eval()
    target = ActorCriticTerrainExo(300, 348, 7, **policy_kwargs()).eval()
    target.load_state_dict(source.state_dict())
    obs = torch.randn(2, 300)
    torch.testing.assert_close(target.act_inference(obs), source.act_inference(obs))
    torch.testing.assert_close(target.evaluate(torch.cat((obs, torch.randn(2, 48)), -1)),
                               source.evaluate(obs))
    assert torch.count_nonzero(target.critic[0].weight[:, 300:]) == 0


def runner_config():
    return dict(algorithm=dict(class_name='PPO', num_learning_epochs=1, num_mini_batches=1),
                policy=dict(class_name='ActorCriticTerrainExo', **policy_kwargs()),
                discriminator=dict(use=False), empirical_normalization=True,
                num_steps_per_env=4, save_interval=100)


def test_runner_ppo_update_uses_separate_actor_and_critic_buffers(tmp_path):
    env = SB3RslVecEnv(DummyVecEnv([make_probe, make_probe]))
    try:
        runner = OnPolicyRunner(env, runner_config(), log_dir=str(tmp_path))
        assert runner.privileged_obs_type == 'critic'
        assert runner.alg.policy.num_actor_obs == 300
        assert runner.alg.policy.num_critic_obs == 348
        assert runner.alg.policy.exo_obs_dim == 256
        # Exercise the real rollout/update without starting an external logger.
        runner.writer = object()
        runner.disable_logs = True
        runner.learn(1)
        storage = runner.alg.storage
        assert storage.observations.shape == (4, 2, 300)
        assert storage.privileged_observations.shape == (4, 2, 348)
        torch.testing.assert_close(storage.observations,
                                   storage.privileged_observations[:, :, :300])
        assert runner.obs_normalizer.count == 10
        assert runner.privileged_obs_normalizer.count == 10
        assert torch.isfinite(storage.returns).all()
        # Checkpoint keeps two normalization shapes and inference needs only policy obs.
        checkpoint = torch.load(tmp_path / 'model_best.pt', weights_only=False)
        assert checkpoint['obs_norm_state_dict']['_mean'].shape == (1, 300)
        assert checkpoint['privileged_obs_norm_state_dict']['_mean'].shape == (1, 348)
        runner.eval_mode()
        obs, _ = env.get_observations()
        assert runner.get_inference_policy()(obs).shape == (2, 7)
    finally:
        env.close()


def test_expanded_normalizer_relearns_new_feature_statistics():
    old = EmpiricalNormalization([300])
    old(torch.randn(4, 300))
    old.count.fill_(1000000)
    target = EmpiricalNormalization([348])
    OnPolicyRunner._load_normalizer_state_flexible(target, old.state_dict(), 'critic')
    torch.testing.assert_close(target._mean[:, :300], old._mean)
    assert target.count == 0
    target(torch.full((2, 348), 5.))
    torch.testing.assert_close(target._mean[:, 300:], torch.full((1, 48), 5.))


def test_runner_migrates_old_checkpoint_and_can_continue_ppo(tmp_path):
    old_env = SB3RslVecEnv(DummyVecEnv([lambda: ObservationProbe(False),
                                     lambda: ObservationProbe(False)]))
    env = SB3RslVecEnv(DummyVecEnv([make_probe, make_probe]))
    try:
        old = OnPolicyRunner(old_env, runner_config(), log_dir=str(tmp_path))
        old.writer, old.disable_logs = object(), True
        old.learn(1)
        path = tmp_path / 'old_actor_checkpoint.pt'
        old.save(path)
        runner = OnPolicyRunner(env, runner_config(), log_dir=str(tmp_path))
        runner.load(path, load_optimizer=True)
        assert not runner.alg.optimizer.state
        old.eval_mode()
        runner.eval_mode()
        obs = torch.randn(2, 300)
        torch.testing.assert_close(runner.get_inference_policy()(obs),
                                   old.get_inference_policy()(obs))
        runner.writer, runner.disable_logs = object(), True
        runner.learn(1)
        assert torch.isfinite(runner.alg.storage.returns).all()
    finally:
        env.close()
        old_env.close()


def test_visualizer_accepts_different_actor_and_critic_dimensions():
    from types import SimpleNamespace
    from visualize import PolicyVisualizer

    policy = ActorCriticTerrainExo(300, 348, 7, **policy_kwargs())
    visualizer = PolicyVisualizer.__new__(PolicyVisualizer)
    visualizer.env = SimpleNamespace(num_actions=7)
    visualizer.cfg = {'policy': {'terrain_patch_shape': (4, 5),
                                 'terrain_feature_dim': 8}}
    spec = visualizer._infer_terrain_exo_spec(policy.state_dict(), 300)
    assert spec['num_critic_obs'] == 348
    assert spec['first256'] is True
