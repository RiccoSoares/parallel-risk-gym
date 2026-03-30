from parallel_risk import ParallelRiskEnv
import numpy as np

def test_basic_mechanics():
    """Test basic game mechanics"""
    print("="*60)
    print("TEST 1: Basic Deployment")
    print("="*60)

    env = ParallelRiskEnv()
    observations, infos = env.reset()
    env.render()

    # agent_0 owns territories [0, 1, 5]
    # Let's deploy some troops
    actions = {
        'agent_0': {
            'num_actions': 2,
            'actions': np.array([
                [0, 0, 3],  # Deploy 3 to territory 0
                [1, 1, 2],  # Deploy 2 to territory 1
                [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
            ], dtype=np.int32)
        },
        'agent_1': {
            'num_actions': 1,
            'actions': np.array([
                [2, 2, 5],  # Deploy all 5 to territory 2
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
            ], dtype=np.int32)
        }
    }

    obs, rewards, terms, truncs, infos = env.step(actions)
    print("\nAfter deployment:")
    env.render()
    print(f"Invalid actions - agent_0: {infos['agent_0']['invalid_actions']}, agent_1: {infos['agent_1']['invalid_actions']}")

    print("\n" + "="*60)
    print("TEST 2: Transfer")
    print("="*60)

    # agent_0 transfers troops from 0 to 1
    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [0, 1, 2],  # Transfer 2 from territory 0 to 1
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
    print("\nAfter transfer:")
    env.render()
    print(f"Territory 0 should have 4 troops (6-2), Territory 1 should have 9 troops (7+2)")

    print("\n" + "="*60)
    print("TEST 3: Attack")
    print("="*60)

    # agent_0 attacks territory 2 from territory 1
    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [1, 2, 5],  # Attack territory 2 with 5 troops from territory 1
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
    print("\nAfter attack:")
    env.render()
    print(f"Combat: 5 attackers vs 8 defenders (8*1.5=12)")
    print(f"Attack power (5) < Defense power (12), so defender should hold")

    print("\n" + "="*60)
    print("TEST 4: Successful Attack")
    print("="*60)

    # Build up agent_0's forces then attack
    env.reset()

    # Multiple turns to build up forces
    for turn in range(10):
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
        env.step(actions)

    print("\nAfter building up forces:")
    env.render()

    # Now attack territory 2 with overwhelming force
    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [1, 2, 40],  # Attack with 40 troops
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
    print("\nAfter overwhelming attack:")
    env.render()
    print("Territory 2 should now be owned by agent_0")

    print("\n" + "="*60)
    print("TEST 5: Victory Condition")
    print("="*60)

    # Set up a near-victory state manually
    env.game_state['territory_ownership'] = np.array([0, 0, 0, 0, 0, 1], dtype=np.int8)
    env.game_state['territory_troops'] = np.array([10, 10, 10, 10, 10, 2], dtype=np.int32)

    print("\nBefore final attack (agent_1 only has territory 5):")
    env.render()

    # agent_0 attacks territory 5
    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [2, 5, 8],  # Attack territory 5 with 8 troops (should win)
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
    print("\nAfter final attack:")
    env.render()
    print(f"\nRewards: {rewards}")
    print(f"Terminations: {terms}")

    if terms['agent_0']:
        print("\n✅ VICTORY CONDITION WORKING!")
        print(f"Winner: {'agent_0' if rewards['agent_0'] > 0 else 'agent_1'}")
    else:
        print("\n❌ Game should have ended")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_basic_mechanics()
