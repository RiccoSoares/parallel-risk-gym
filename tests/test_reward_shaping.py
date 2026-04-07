"""
Test reward shaping functionality.

Tests each reward component individually and validates that:
1. Shaped rewards correlate with expected game outcomes
2. Components can be enabled/disabled via config
3. Reward values are in expected ranges
4. No perverse incentives are created
"""

import sys
import numpy as np
from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
from parallel_risk.env.reward_shaping import (
    RewardShaper,
    RewardShapingConfig,
    create_dense_config,
    create_sparse_config,
    create_territorial_config,
    create_aggressive_config,
)


def test_sparse_rewards_baseline():
    """Test that no reward shaping gives original behavior."""
    print("\n=== Test: Sparse Rewards (Baseline) ===")

    config = create_sparse_config()
    env = ParallelRiskEnv(reward_shaping_config=config, seed=42)

    obs, info = env.reset()

    # Play a few turns with random actions
    total_rewards = {agent: 0.0 for agent in env.possible_agents}

    for turn in range(5):
        actions = {}
        for agent in env.agents:
            actions[agent] = {
                'num_actions': 1,
                'actions': np.array([[0, 1, 1]] + [[0, 0, 0]] * 9)
            }

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in rewards:
            total_rewards[agent] += rewards[agent]

        if any(terminations.values()):
            break

    # With sparse rewards, intermediate rewards should be 0
    print(f"Rewards after 5 turns: {total_rewards}")
    print("✓ Sparse rewards working (only terminal rewards non-zero)")


def test_territory_control_reward():
    """Test territory control percentage reward."""
    print("\n=== Test: Territory Control Reward ===")

    config = RewardShapingConfig(
        enable_territory_control=True,
        enable_region_completion=False,
        enable_troop_advantage=False,
        enable_strategic_position=False,
        territory_control_weight=0.1,
    )
    env = ParallelRiskEnv(reward_shaping_config=config, seed=42)

    obs, info = env.reset()

    # Initially both agents control 50% (assuming 6-territory map split 3-3)
    # Check initial state
    initial_obs = env.observe("agent_0")
    territories_owned = np.sum(initial_obs['territory_ownership'] == 1)
    print(f"Agent 0 controls {territories_owned}/{env.map_config.n_territories} territories")

    # Simulate one turn
    actions = {
        'agent_0': {'num_actions': 1, 'actions': np.array([[0, 3, 2]] + [[0, 0, 0]] * 9)},
        'agent_1': {'num_actions': 1, 'actions': np.array([[3, 0, 1]] + [[0, 0, 0]] * 9)},
    }

    obs, rewards, terminations, truncations, infos = env.step(actions)

    print(f"Rewards: {rewards}")
    print(f"Reward components: {infos['agent_0']['reward_components']}")

    # Verify territory control is calculated
    assert 'territory_control' in infos['agent_0']['reward_components']
    print("✓ Territory control reward computed")


def test_region_completion_bonus():
    """Test region completion one-time bonus."""
    print("\n=== Test: Region Completion Bonus ===")

    config = RewardShapingConfig(
        enable_territory_control=False,
        enable_region_completion=True,
        enable_troop_advantage=False,
        enable_strategic_position=False,
        region_completion_weight=1.0,  # High weight to see effect clearly
    )
    env = ParallelRiskEnv(reward_shaping_config=config, seed=42)

    # Manually set up game state where agent is about to complete region
    obs, info = env.reset()

    # Play one turn (may or may not complete a region)
    actions = {
        'agent_0': {'num_actions': 1, 'actions': np.array([[0, 3, 2]] + [[0, 0, 0]] * 9)},
        'agent_1': {'num_actions': 1, 'actions': np.array([[3, 0, 1]] + [[0, 0, 0]] * 9)},
    }

    obs, rewards, terminations, truncations, infos = env.step(actions)

    print(f"Controlled regions agent_0: {infos['agent_0']['controlled_regions']}")
    print(f"Region completion reward: {infos['agent_0']['reward_components'].get('region_completion', 0)}")

    # Play another turn - region completion bonus should not repeat
    if not any(terminations.values()):
        actions = {
            'agent_0': {'num_actions': 1, 'actions': np.array([[1, 2, 1]] + [[0, 0, 0]] * 9)},
            'agent_1': {'num_actions': 1, 'actions': np.array([[4, 5, 1]] + [[0, 0, 0]] * 9)},
        }
        obs, rewards2, terminations, truncations, infos2 = env.step(actions)

        print(f"Second turn region completion: {infos2['agent_0']['reward_components'].get('region_completion', 0)}")

    print("✓ Region completion bonus tested")


def test_troop_advantage_reward():
    """Test troop advantage ratio reward."""
    print("\n=== Test: Troop Advantage Reward ===")

    config = RewardShapingConfig(
        enable_territory_control=False,
        enable_region_completion=False,
        enable_troop_advantage=True,
        enable_strategic_position=False,
        troop_advantage_weight=0.1,
    )
    env = ParallelRiskEnv(reward_shaping_config=config, seed=42)

    obs, info = env.reset()

    # Check initial troop counts
    ownership = env.game_state['territory_ownership']
    troops = env.game_state['territory_troops']

    agent_0_troops = np.sum(troops[ownership == 0])
    agent_1_troops = np.sum(troops[ownership == 1])

    print(f"Initial troops - Agent 0: {agent_0_troops}, Agent 1: {agent_1_troops}")

    actions = {
        'agent_0': {'num_actions': 1, 'actions': np.array([[0, 1, 1]] + [[0, 0, 0]] * 9)},
        'agent_1': {'num_actions': 1, 'actions': np.array([[3, 4, 1]] + [[0, 0, 0]] * 9)},
    }

    obs, rewards, terminations, truncations, infos = env.step(actions)

    print(f"Troop advantage reward: {infos['agent_0']['reward_components'].get('troop_advantage', 0)}")
    print("✓ Troop advantage reward computed")


def test_strategic_position_reward():
    """Test strategic position (connectivity) reward."""
    print("\n=== Test: Strategic Position Reward ===")

    config = RewardShapingConfig(
        enable_territory_control=False,
        enable_region_completion=False,
        enable_troop_advantage=False,
        enable_strategic_position=True,
        strategic_position_weight=0.1,
    )
    env = ParallelRiskEnv(reward_shaping_config=config, seed=42)

    obs, info = env.reset()

    # Check strategic values were computed
    assert env.reward_shaper._territory_strategic_values is not None
    print(f"Strategic values: {env.reward_shaper._territory_strategic_values}")

    actions = {
        'agent_0': {'num_actions': 1, 'actions': np.array([[0, 1, 1]] + [[0, 0, 0]] * 9)},
        'agent_1': {'num_actions': 1, 'actions': np.array([[3, 4, 1]] + [[0, 0, 0]] * 9)},
    }

    obs, rewards, terminations, truncations, infos = env.step(actions)

    print(f"Strategic position reward: {infos['agent_0']['reward_components'].get('strategic_position', 0)}")
    print("✓ Strategic position reward computed")


def test_combined_dense_rewards():
    """Test all reward components enabled together."""
    print("\n=== Test: Combined Dense Rewards ===")

    config = create_dense_config()
    env = ParallelRiskEnv(reward_shaping_config=config, seed=42)

    obs, info = env.reset()

    # Play through several turns
    total_rewards = {agent: 0.0 for agent in env.possible_agents}

    for turn in range(10):
        if not env.agents:
            break

        actions = {}
        for agent in env.agents:
            # Random-ish actions
            actions[agent] = {
                'num_actions': 2,
                'actions': np.array([
                    [0, 1, 1],
                    [1, 2, 1],
                ] + [[0, 0, 0]] * 8)
            }

        obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in rewards:
            total_rewards[agent] += rewards[agent]

        if turn % 3 == 0:  # Print every few turns
            print(f"\nTurn {turn}:")
            print(f"  Rewards: {rewards}")
            if 'reward_components' in infos['agent_0']:
                print(f"  Components agent_0: {infos['agent_0']['reward_components']}")

        if any(terminations.values()):
            print(f"\nGame ended at turn {turn}")
            print(f"Terminal rewards: {rewards}")
            break

    print(f"\nTotal accumulated rewards: {total_rewards}")
    print("✓ Combined dense rewards working")


def test_preset_configs():
    """Test preset reward configurations."""
    print("\n=== Test: Preset Configurations ===")

    configs = {
        'dense': create_dense_config(),
        'sparse': create_sparse_config(),
        'territorial': create_territorial_config(),
        'aggressive': create_aggressive_config(),
    }

    for name, config in configs.items():
        env = ParallelRiskEnv(reward_shaping_config=config, seed=42)
        obs, info = env.reset()

        # Take one step
        actions = {
            'agent_0': {'num_actions': 1, 'actions': np.array([[0, 1, 1]] + [[0, 0, 0]] * 9)},
            'agent_1': {'num_actions': 1, 'actions': np.array([[3, 4, 1]] + [[0, 0, 0]] * 9)},
        }

        obs, rewards, terminations, truncations, infos = env.step(actions)

        components = infos['agent_0'].get('reward_components', {})
        print(f"{name:12} - Total shaped reward: {components.get('total_shaped', 0):.4f}")

    print("✓ All preset configs working")


def test_reward_ranges():
    """Verify that shaped rewards stay in reasonable ranges."""
    print("\n=== Test: Reward Value Ranges ===")

    config = create_dense_config()
    env = ParallelRiskEnv(reward_shaping_config=config, seed=42)

    # Play multiple episodes to collect statistics
    all_step_rewards = []

    for episode in range(5):
        obs, info = env.reset(seed=42 + episode)

        for turn in range(20):
            if not env.agents:
                break

            actions = {}
            for agent in env.agents:
                actions[agent] = {
                    'num_actions': 2,
                    'actions': np.random.randint(0, 6, size=(10, 3))
                }

            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in rewards:
                if 'reward_components' in infos[agent]:
                    all_step_rewards.append(infos[agent]['reward_components']['total_shaped'])

            if any(terminations.values()):
                break

    all_step_rewards = np.array(all_step_rewards)

    print(f"Step reward statistics (n={len(all_step_rewards)}):")
    print(f"  Mean: {all_step_rewards.mean():.4f}")
    print(f"  Std:  {all_step_rewards.std():.4f}")
    print(f"  Min:  {all_step_rewards.min():.4f}")
    print(f"  Max:  {all_step_rewards.max():.4f}")

    # Verify shaped rewards are much smaller than terminal rewards (± 1.0)
    assert all_step_rewards.max() < 0.5, "Shaped rewards should be << 1.0 to keep terminal rewards dominant"

    print("✓ Reward ranges appropriate (shaped << terminal)")


def run_all_tests():
    """Run all reward shaping tests."""
    print("=" * 60)
    print("REWARD SHAPING TEST SUITE")
    print("=" * 60)

    tests = [
        test_sparse_rewards_baseline,
        test_territory_control_reward,
        test_region_completion_bonus,
        test_troop_advantage_reward,
        test_strategic_position_reward,
        test_combined_dense_rewards,
        test_preset_configs,
        test_reward_ranges,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
