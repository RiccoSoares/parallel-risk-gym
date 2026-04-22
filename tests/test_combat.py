from parallel_risk import ParallelRiskEnv
import numpy as np

def test_combat_mechanics():
    """Test the new percentage-based combat resolution"""
    print("="*60)
    print("TEST: New Combat Mechanics")
    print("="*60)
    print("\nRules:")
    print("  - Defenders lose 70% of attacking troops")
    print("  - Attackers lose 60% of defending troops")
    print("  - Attacker wins if defenders <= 0")

    env = ParallelRiskEnv()

    print("\n" + "="*60)
    print("Scenario 1: 10 attackers vs 5 defenders")
    print("="*60)

    # Manually set up a combat scenario
    env.reset()
    env.game_state['territory_troops'][1] = 11  # agent_0's territory (needs 1 to stay)
    env.game_state['territory_troops'][2] = 5   # agent_1's territory

    print("\nBefore attack:")
    print(f"  Territory 1 (agent_0): 11 troops")
    print(f"  Territory 2 (agent_1): 5 troops")

    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [1, 2, 10],  # Attack with 10 troops
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

    print("\nCombat calculation:")
    print(f"  Defender casualties: 70% of 10 = 7")
    print(f"  Attacker casualties: 60% of 5 = 3")
    print(f"  Defenders remaining: 5 - 7 = -2 → ATTACKER WINS")
    print(f"  Attackers remaining: 10 - 3 = 7")

    print("\nAfter attack:")
    print(f"  Territory 1 (agent_0): {env.game_state['territory_troops'][1]} troops (should be 1)")
    print(f"  Territory 2: {env.game_state['territory_troops'][2]} troops, owner: agent_{env.game_state['territory_ownership'][2]}")
    print(f"  Expected: Territory 2 captured by agent_0 with 7 troops")

    print("\n" + "="*60)
    print("Scenario 2: 10 attackers vs 15 defenders")
    print("="*60)

    env.reset()
    env.game_state['territory_troops'][1] = 11
    env.game_state['territory_troops'][2] = 15

    print("\nBefore attack:")
    print(f"  Territory 1 (agent_0): 11 troops")
    print(f"  Territory 2 (agent_1): 15 troops")

    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [1, 2, 10],  # Attack with 10 troops
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

    print("\nCombat calculation:")
    print(f"  Defender casualties: 70% of 10 = 7")
    print(f"  Attacker casualties: 60% of 15 = 9")
    print(f"  Defenders remaining: 15 - 7 = 8 → DEFENDER HOLDS")
    print(f"  Attackers remaining: 10 - 9 = 1 (returns to source)")

    print("\nAfter attack:")
    print(f"  Territory 1 (agent_0): {env.game_state['territory_troops'][1]} troops (should be 2: 11 - 10 + 1)")
    print(f"  Territory 2: {env.game_state['territory_troops'][2]} troops, owner: agent_{env.game_state['territory_ownership'][2]}")
    print(f"  Expected: Territory 2 still owned by agent_1 with 8 troops, Territory 1 has 2 troops")

    print("\n" + "="*60)
    print("Scenario 3: 20 attackers vs 10 defenders")
    print("="*60)

    env.reset()
    env.game_state['territory_troops'][1] = 21
    env.game_state['territory_troops'][2] = 10

    print("\nBefore attack:")
    print(f"  Territory 1 (agent_0): 21 troops")
    print(f"  Territory 2 (agent_1): 10 troops")

    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [1, 2, 20],  # Attack with 20 troops
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

    print("\nCombat calculation:")
    print(f"  Defender casualties: 70% of 20 = 14")
    print(f"  Attacker casualties: 60% of 10 = 6")
    print(f"  Defenders remaining: 10 - 14 = -4 → ATTACKER WINS")
    print(f"  Attackers remaining: 20 - 6 = 14")

    print("\nAfter attack:")
    print(f"  Territory 1 (agent_0): {env.game_state['territory_troops'][1]} troops (should be 1)")
    print(f"  Territory 2: {env.game_state['territory_troops'][2]} troops, owner: agent_{env.game_state['territory_ownership'][2]}")
    print(f"  Expected: Territory 2 captured by agent_0 with 14 troops")

    print("\n" + "="*60)
    print("Scenario 4: Edge case - Equal forces (10 vs 10)")
    print("="*60)

    env.reset()
    env.game_state['territory_troops'][1] = 11
    env.game_state['territory_troops'][2] = 10

    print("\nBefore attack:")
    print(f"  Territory 1 (agent_0): 11 troops")
    print(f"  Territory 2 (agent_1): 10 troops")

    actions = {
        'agent_0': {
            'num_actions': 1,
            'actions': np.array([
                [1, 2, 10],  # Attack with 10 troops
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

    print("\nCombat calculation:")
    print(f"  Defender casualties: 70% of 10 = 7")
    print(f"  Attacker casualties: 60% of 10 = 6")
    print(f"  Defenders remaining: 10 - 7 = 3 → DEFENDER HOLDS")
    print(f"  Attackers remaining: 10 - 6 = 4 (return to source)")

    print("\nAfter attack:")
    print(f"  Territory 1 (agent_0): {env.game_state['territory_troops'][1]} troops (should be 5: 11 - 10 + 4)")
    print(f"  Territory 2: {env.game_state['territory_troops'][2]} troops, owner: agent_{env.game_state['territory_ownership'][2]}")
    print(f"  Expected: Territory 2 still owned by agent_1 with 3 troops, Territory 1 has 5 troops")

    print("\n" + "="*60)
    print("✅ ALL COMBAT TESTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_combat_mechanics()
