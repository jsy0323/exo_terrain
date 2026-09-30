from __future__ import annotations

import torch

from RL.modules.actor_critic_latent_test import ActorCritic as LegacyTerrainActorCritic
from RL.modules.actor_critic_terrainexo import ActorCriticTerrainExo


def _make_policy(*, first256: bool = False) -> ActorCriticTerrainExo:
    num_obs = 300 if first256 else 50
    patch_shape = (4, 5)
    return ActorCriticTerrainExo(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=7,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        exo_actor_hidden_dims=(16, 8),
        terrain_patch_shape=patch_shape,
        terrain_feature_dim=8,
        terrain_attention_enabled=True,
        exo_action_dim=2,
        agent_action_dims=(2, 5),
        human_branch_idx=1,
        shared_actor_obs=True,
        first256=first256,
        exo_obs=False,
        init_noise_std=0.2,
    )


def test_policy_ppo_interface_shapes_and_gradients():
    policy = _make_policy()
    observations = torch.randn(4, 50)

    actions = policy.act(observations)
    log_prob = policy.get_actions_log_prob(actions.detach())
    values = policy.evaluate(observations)
    loss = -log_prob.mean() - 1.0e-3 * policy.entropy.mean() + values.square().mean()
    loss.backward()

    assert actions.shape == (4, 7)
    assert policy.action_mean.shape == (4, 7)
    assert policy.action_std.shape == (4, 7)
    assert log_prob.shape == (4,)
    assert policy.entropy.shape == (4,)
    assert policy.entropy_per_agent.shape == (4, 2)
    assert values.shape == (4, 1)
    assert policy.human_actor.action_head.weight.grad is not None
    assert policy.exo_actor.net[-1].weight.grad is not None
    assert policy.critic[-1].weight.grad is not None

    attention = policy.last_terrain_attention
    assert attention.shape == (4, 1, 4, 5)
    torch.testing.assert_close(
        attention.flatten(2).sum(dim=-1),
        torch.ones(4, 1),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_human_then_exo_action_order_and_first256_isolation():
    policy = _make_policy(first256=True)
    policy.eval()

    for parameter in policy.human_actor.parameters():
        parameter.data.zero_()
    for parameter in policy.exo_actor.parameters():
        parameter.data.zero_()
    policy.human_actor.action_head.bias.data.copy_(torch.arange(1.0, 6.0))
    policy.exo_actor.net[-1].bias.data.copy_(torch.tensor([7.0, 8.0]))

    observations = torch.randn(3, 300)
    actions = policy.act_inference(observations)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]).expand(3, -1)
    torch.testing.assert_close(actions, expected)

    exo_before = policy.exo_actor(observations[:, :256])
    terrain_changed = observations.clone()
    terrain_changed[:, -20:] += 100.0
    exo_after = policy.exo_actor(terrain_changed[:, :256])
    torch.testing.assert_close(exo_before, exo_after)


def test_legacy_terrain_checkpoint_initializes_only_humanoid_branch():
    legacy = LegacyTerrainActorCritic(
        num_actor_obs=50,
        num_critic_obs=50,
        num_actions=5,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        terrain_cnn_enabled=True,
        terrain_patch_shape=(4, 5),
        terrain_feature_dim=8,
        terrain_attention_enabled=True,
    )
    policy = _make_policy()
    exo_before = {key: value.clone() for key, value in policy.exo_actor.state_dict().items()}

    assert policy.load_state_dict(legacy.state_dict()) is True

    observations = torch.randn(3, 50)
    legacy.eval()
    policy.eval()
    torch.testing.assert_close(
        policy.human_actor(observations),
        legacy.act_inference(observations),
    )
    for key, value in policy.exo_actor.state_dict().items():
        torch.testing.assert_close(value, exo_before[key])
    torch.testing.assert_close(policy.log_std, legacy.log_std)


def test_native_checkpoint_round_trip():
    source = _make_policy()
    target = _make_policy()
    target.load_state_dict(source.state_dict())
    source.eval()
    target.eval()

    observations = torch.randn(2, 50)
    torch.testing.assert_close(
        target.act_inference(observations),
        source.act_inference(observations),
    )
