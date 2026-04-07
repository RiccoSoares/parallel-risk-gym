"""
Test RLlib wrapper for Parallel Risk.

Verifies that:
1. Environment can be created
2. Observation/action spaces are correct
3. reset() and step() work
4. Integration with RLlib works
"""

import sys
import numpy as np

from parallel_risk.training.rllib_wrapper import RLlibParallelRiskEnv, make_rllib_env
from parallel_risk.env.reward_shaping import create_dense_config


def test_env_creation():
    """Test that environment can be created."""
    print("\n=== Test: Environment Creation ===")

    env = RLlibParallelRiskEnv()
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Agent IDs: {env.get_agent_ids()}")


def test_reset():
    """Test environment reset."""
    print("\n=== Test: Reset ===")

    env = RLlibParallelRiskEnv()
    obs, info = env.reset()

    print(f"✓ Reset successful")
    print(f"  Number of agents: {len(obs)}")
    print(f"  Observation shape: {obs['agent_0'].shape}")
    print(f"  Observation dtype: {obs['agent_0'].dtype}")

    # Verify observation is in expected space
    assert env.observation_space.contains(obs['agent_0']), "Observation not in space"
    print(f"✓ Observation in valid space")


def test_step():
    """Test environment step with random actions."""
    print("\n=== Test: Step with Random Actions ===")

    env = RLlibParallelRiskEnv({"action_budget": 3})
    obs, info = env.reset(seed=42)

    # Sample random actions
    actions = {
        agent: env.action_space.sample()
        for agent in env.get_agent_ids()
    }

    print(f"Action budget: {env.action_budget}")
    print(f"Action space sample: {actions['agent_0']}")

    # Step
    obs, rewards, terminateds, truncateds, infos = env.step(actions)

    print(f"✓ Step successful")
    print(f"  Rewards: {rewards}")
    print(f"  Terminated: {terminateds}")
    print(f"  Info keys: {list(infos['agent_0'].keys())}")


def test_episode():
    """Run a short episode to ensure everything works."""
    print("\n=== Test: Full Episode ===")

    env = RLlibParallelRiskEnv({
        "action_budget": 5,
        "max_turns": 20,
        "seed": 42,
    })

    obs, info = env.reset(seed=42)

    episode_rewards = {agent: [] for agent in env.get_agent_ids()}
    turn = 0

    while True:
        # Random actions
        actions = {
            agent: env.action_space.sample()
            for agent in env.get_agent_ids() if agent in obs
        }

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for agent in env.get_agent_ids():
            if agent in rewards:
                episode_rewards[agent].append(rewards[agent])

        turn += 1

        if terminateds.get("__all__", False):
            print(f"✓ Episode ended at turn {turn}")
            break

        if turn > 25:
            print(f"⚠️  Episode exceeded expected length")
            break

    print(f"  Total rewards:")
    for agent, rewards_list in episode_rewards.items():
        print(f"    {agent}: {sum(rewards_list):.3f} (over {len(rewards_list)} steps)")


def test_with_reward_shaping():
    """Test with reward shaping enabled."""
    print("\n=== Test: With Reward Shaping ===")

    config = {
        "action_budget": 5,
        "reward_shaping_config": create_dense_config(),
        "seed": 42,
    }

    env = RLlibParallelRiskEnv(config)
    obs, info = env.reset(seed=42)

    # Take a few steps
    for i in range(5):
        actions = {
            agent: env.action_space.sample()
            for agent in env.get_agent_ids() if agent in obs
        }

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        if i == 0:
            print(f"✓ Reward shaping enabled")
            print(f"  Step {i} rewards: {rewards}")
            if 'reward_components' in infos.get('agent_0', {}):
                print(f"  Reward components: {infos['agent_0']['reward_components']}")

        if terminateds.get("__all__", False):
            break


def test_make_env_factory():
    """Test the factory function."""
    print("\n=== Test: Factory Function ===")

    config = {
        "map_name": "simple_6",
        "action_budget": 5,
    }

    env = make_rllib_env(config)
    obs, info = env.reset()

    print(f"✓ Factory function works")
    print(f"  Created environment with map: {env.map_name}")


def test_observation_flattening():
    """Test that observation flattening is correct."""
    print("\n=== Test: Observation Flattening ===")

    env = RLlibParallelRiskEnv()
    obs, info = env.reset(seed=42)

    # Check observation size
    n_territories = env.env.map_config.n_territories
    n_regions = len(env.env.map_config.regions)

    expected_size = (
        n_territories +  # ownership
        n_territories +  # troops
        n_territories * n_territories +  # adjacency
        1 +  # income
        1 +  # turn
        n_regions  # regions
    )

    actual_size = obs['agent_0'].shape[0]

    print(f"  Expected size: {expected_size}")
    print(f"  Actual size: {actual_size}")

    assert actual_size == expected_size, f"Size mismatch: {actual_size} != {expected_size}"
    print(f"✓ Observation flattening correct")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("RLLIB WRAPPER TEST SUITE")
    print("=" * 70)

    tests = [
        test_env_creation,
        test_reset,
        test_step,
        test_observation_flattening,
        test_episode,
        test_with_reward_shaping,
        test_make_env_factory,
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

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
