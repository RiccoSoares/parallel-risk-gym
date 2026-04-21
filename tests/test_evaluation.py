"""Tests for evaluation module."""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parallel_risk.agents.random_agent import RandomAgent
from parallel_risk.training.rllib.wrapper import make_rllib_env


def test_random_agent_creation():
    """Test RandomAgent initialization."""
    print("\nTest: RandomAgent creation")

    agent = RandomAgent(n_territories=6, action_budget=5, mode="rllib")

    assert agent.n_territories == 6
    assert agent.action_budget == 5
    assert agent.mode == "rllib"

    print("✓ RandomAgent created successfully")


def test_random_agent_rllib_action():
    """Test RandomAgent generates valid RLlib actions."""
    print("\nTest: RandomAgent RLlib action format")

    agent = RandomAgent(n_territories=6, action_budget=5, mode="rllib")

    action = agent.get_action()

    # Should be tuple of tuples
    assert isinstance(action, tuple)
    assert len(action) == 5  # action_budget

    for sub_action in action:
        assert isinstance(sub_action, tuple)
        assert len(sub_action) == 3  # [source, dest, troops]
        source, dest, troops = sub_action
        assert 0 <= source < 6
        assert 0 <= dest < 6
        assert 1 <= troops <= 20

    print(f"✓ Generated valid RLlib action: {action[0]}, ...")


def test_random_agent_raw_action():
    """Test RandomAgent generates valid raw actions."""
    print("\nTest: RandomAgent raw action format")

    agent = RandomAgent(n_territories=6, action_budget=5, mode="raw")

    action = agent.get_action()

    # Should be dict with num_actions and actions
    assert isinstance(action, dict)
    assert "num_actions" in action
    assert "actions" in action

    num_actions = action["num_actions"]
    actions_array = action["actions"]

    assert 0 <= num_actions <= 5
    assert actions_array.shape == (5, 3)
    assert actions_array.dtype == np.int32

    print(f"✓ Generated valid raw action with {num_actions} actions")


def test_random_agent_in_environment():
    """Test RandomAgent works with actual RLlib environment."""
    print("\nTest: RandomAgent in RLlib environment")

    # Create environment
    env_config = {
        "map_name": "simple_6",
        "max_turns": 10,
        "action_budget": 5,
        "reward_shaping_type": "sparse",
    }
    env = make_rllib_env(env_config)

    # Get map info
    n_territories = env.env.map_config.n_territories

    # Create random agents
    agent_0 = RandomAgent(n_territories=n_territories, action_budget=5, mode="rllib")
    agent_1 = RandomAgent(n_territories=n_territories, action_budget=5, mode="rllib")

    # Run one episode
    obs, info = env.reset(seed=42)

    step_count = 0
    max_steps = 100

    while step_count < max_steps:
        # Generate actions
        actions = {
            "agent_0": agent_0.get_action(),
            "agent_1": agent_1.get_action(),
        }

        # Step environment
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Check if done
        done = {k: terminateds.get(k, False) or truncateds.get(k, False)
               for k in ["agent_0", "agent_1"]}

        if done.get("agent_0") or done.get("agent_1"):
            break

        step_count += 1

    print(f"✓ Completed episode in {step_count} steps")
    print(f"  Final rewards: {rewards}")


def test_evaluation_imports():
    """Test that evaluation module can be imported."""
    print("\nTest: Evaluation module imports")

    try:
        from parallel_risk.evaluation import evaluate_policy
        print("✓ evaluate_policy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluate_policy: {e}")
        raise

    try:
        from parallel_risk.evaluation.visualize import plot_win_rate_curve
        print("✓ visualize module imported successfully")
    except ImportError as e:
        # Matplotlib is optional for tests
        print(f"⚠ visualize module requires matplotlib (optional): {e}")
        print("✓ This is okay - visualization is optional")


def run_all_tests():
    """Run all evaluation tests."""
    print("="*60)
    print("EVALUATION MODULE TESTS")
    print("="*60)

    tests = [
        test_random_agent_creation,
        test_random_agent_rllib_action,
        test_random_agent_raw_action,
        test_random_agent_in_environment,
        test_evaluation_imports,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
