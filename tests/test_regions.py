from parallel_risk import ParallelRiskEnv
import numpy as np

def test_region_bonuses():
    """Test that region bonuses work correctly"""
    print("="*60)
    print("TEST: Region Bonus System")
    print("="*60)

    env = ParallelRiskEnv()
    observations, infos = env.reset()

    print("\nInitial state:")
    print("  North region [0,1,2]: agent_0 owns [0,1], agent_1 owns [2]")
    print("  South region [3,4,5]: agent_1 owns [3,4], agent_0 owns [5]")
    print("  Expected: Both agents get base income only (5 troops)")
    env.render()

    print("\n" + "="*60)
    print("Scenario 1: agent_0 captures north region")
    print("="*60)

    # Build up forces on territory 1
    for turn in range(5):
        actions = {
            'agent_0': {
                'num_actions': 1,
                'actions': np.array([
                    [1, 1, 5],  # Deploy all to territory 1
                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                ], dtype=np.int32)
            },
            'agent_1': {
                'num_actions': 0,
                'actions': np.zeros((10, 3), dtype=np.int32)
            }
        }
        obs, rewards, terms, truncs, infos = env.step(actions)

    print(f"\nAfter building up (agent_0 income from infos: {infos['agent_0']['income']}):")
    env.render()

    # Attack territory 2 to complete north region
    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [1, 2, 20],  # Attack territory 2 with overwhelming force
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
            ], dtype=np.int32)
        },
        'agent_1': {
            'num_actions': 0,
            'actions': np.zeros((10, 3), dtype=np.int32)
        }
    }

    obs, rewards, terms, truncs, infos = env.step(actions)

    print("\nAfter capturing territory 2:")
    env.render()
    print(f"\nExpected: agent_0 should now control north region")
    print(f"agent_0 income should be 7 (5 base + 2 north bonus)")
    print(f"Actual income from infos: {infos['agent_0']['income']}")
    print(f"Controlled regions: {infos['agent_0']['controlled_regions']}")

    # Verify observation includes region control
    print(f"\nObservation region_control for agent_0: {obs['agent_0']['region_control']}")
    print(f"Expected: [1, 0] (north=yes, south=no)")

    print("\n" + "="*60)
    print("Scenario 2: agent_1 completes south region")
    print("="*60)

    # agent_1 needs to capture territory 5 to complete south
    # First build up forces
    for turn in range(5):
        actions = {
            'agent_0': {
                'num_actions': 0,
                'actions': np.zeros((10, 3), dtype=np.int32)
            },
            'agent_1': {
                'num_actions': 1,
                'actions': np.array([
                    [4, 4, 5],  # Deploy to territory 4
                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                ], dtype=np.int32)
            }
        }
        obs, rewards, terms, truncs, infos = env.step(actions)

    print(f"\nAfter building up agent_1 forces:")
    env.render()

    # Attack territory 5
    actions = {
        'agent_0': {
            'num_actions': 0,
            'actions': np.zeros((10, 3), dtype=np.int32)
        },
        'agent_1': {
            'num_actions': 1,
            'actions': np.array([
                [4, 5, 20],  # Attack territory 5
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
            ], dtype=np.int32)
        }
    }

    obs, rewards, terms, truncs, infos = env.step(actions)

    print("\nAfter capturing territory 5:")
    env.render()
    print(f"\nExpected: agent_1 should now control south region")
    print(f"agent_1 income should be 7 (5 base + 2 south bonus)")
    print(f"Actual income from infos: {infos['agent_1']['income']}")
    print(f"Controlled regions: {infos['agent_1']['controlled_regions']}")

    print("\n" + "="*60)
    print("Scenario 3: Losing region control")
    print("="*60)

    # agent_0 loses territory 2, breaking north region control
    # First build agent_1 forces near territory 2
    for turn in range(3):
        actions = {
            'agent_0': {
                'num_actions': 0,
                'actions': np.zeros((10, 3), dtype=np.int32)
            },
            'agent_1': {
                'num_actions': 1,
                'actions': np.array([
                    [5, 5, 7],  # Deploy to territory 5
                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                ], dtype=np.int32)
            }
        }
        obs, rewards, terms, truncs, infos = env.step(actions)

    # Attack territory 2 back
    actions = {
        'agent_0': {
            'num_actions': 0,
            'actions': np.zeros((10, 3), dtype=np.int32)
        },
        'agent_1': {
            'num_actions': 1,
            'actions': np.array([
                [5, 2, 30],  # Attack territory 2
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
            ], dtype=np.int32)
        }
    }

    obs, rewards, terms, truncs, infos = env.step(actions)

    print("\nAfter agent_1 recaptures territory 2:")
    env.render()
    print(f"\nExpected: agent_0 loses north region control (lost territory 2)")
    print(f"agent_0 income should drop to 5 (5 base + 0 bonus)")
    print(f"agent_0 controlled regions: {infos['agent_0']['controlled_regions']}")
    print(f"agent_0 income: {infos['agent_0']['income']}")

    print("\n" + "="*60)
    print("✅ ALL REGION BONUS TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_region_bonuses()
