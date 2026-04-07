"""
Example: Using reward shaping in Parallel Risk.

This script demonstrates how to enable and use reward shaping,
and visualizes the reward components over a game.
"""

import numpy as np
from parallel_risk.env.parallel_risk_env import ParallelRiskEnv
from parallel_risk.env.reward_shaping import (
    create_dense_config,
    create_sparse_config,
    create_territorial_config,
)


def random_agent_actions(env, agent):
    """Generate random valid-ish actions for an agent."""
    num_actions = np.random.randint(1, 4)
    actions = np.random.randint(0, env.map_config.n_territories, size=(10, 3))
    return {
        'num_actions': num_actions,
        'actions': actions
    }


def run_example_game(reward_config, config_name="Unknown"):
    """Run a single game and track rewards."""
    print(f"\n{'=' * 70}")
    print(f"Running game with: {config_name}")
    print('=' * 70)

    env = ParallelRiskEnv(reward_shaping_config=reward_config, seed=42)
    obs, info = env.reset()

    episode_rewards = {agent: [] for agent in env.possible_agents}
    episode_components = {agent: [] for agent in env.possible_agents}

    turn = 0
    while env.agents and turn < 30:
        # Generate actions for both agents
        actions = {agent: random_agent_actions(env, agent) for agent in env.agents}

        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Track rewards
        for agent in env.possible_agents:
            episode_rewards[agent].append(rewards[agent])
            if 'reward_components' in infos[agent]:
                episode_components[agent].append(infos[agent]['reward_components'])

        # Print occasional updates
        if turn % 10 == 0 or any(terminations.values()):
            print(f"\nTurn {turn}:")
            for agent in env.possible_agents:
                print(f"  {agent}:")
                print(f"    Step reward: {rewards[agent]:.4f}")
                if 'reward_components' in infos[agent]:
                    comps = infos[agent]['reward_components']
                    print(f"    Components: {dict((k, f'{v:.4f}') for k, v in comps.items())}")
                print(f"    Total so far: {sum(episode_rewards[agent]):.4f}")

        if any(terminations.values()):
            print(f"\n🏁 Game ended at turn {turn}")
            winner = [a for a, t in terminations.items() if t and rewards[a] > 0][0]
            print(f"   Winner: {winner}")
            break

        turn += 1

    # Summary statistics
    print(f"\n📊 Episode Summary ({config_name}):")
    for agent in env.possible_agents:
        total = sum(episode_rewards[agent])
        avg_per_step = np.mean(episode_rewards[agent]) if episode_rewards[agent] else 0
        print(f"  {agent}:")
        print(f"    Total reward: {total:.4f}")
        print(f"    Average per step: {avg_per_step:.4f}")
        print(f"    Steps: {len(episode_rewards[agent])}")


def compare_configs():
    """Compare different reward shaping configurations."""
    print("\n" + "=" * 70)
    print("COMPARING REWARD SHAPING CONFIGURATIONS")
    print("=" * 70)

    configs = [
        (create_sparse_config(), "Sparse (No Shaping)"),
        (create_territorial_config(), "Territorial"),
        (create_dense_config(), "Dense (All Components)"),
    ]

    for config, name in configs:
        run_example_game(config, name)


def demonstrate_reward_components():
    """Show each reward component individually."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL REWARD COMPONENTS DEMONSTRATION")
    print("=" * 70)

    from parallel_risk.env.reward_shaping import RewardShapingConfig

    components = [
        (RewardShapingConfig(
            enable_territory_control=True,
            enable_region_completion=False,
            enable_troop_advantage=False,
            enable_strategic_position=False,
            territory_control_weight=0.1,  # Higher weight to see effect
        ), "Territory Control Only"),

        (RewardShapingConfig(
            enable_territory_control=False,
            enable_region_completion=True,
            enable_troop_advantage=False,
            enable_strategic_position=False,
            region_completion_weight=0.5,  # Higher weight to see effect
        ), "Region Completion Only"),

        (RewardShapingConfig(
            enable_territory_control=False,
            enable_region_completion=False,
            enable_troop_advantage=True,
            enable_strategic_position=False,
            troop_advantage_weight=0.1,  # Higher weight to see effect
        ), "Troop Advantage Only"),

        (RewardShapingConfig(
            enable_territory_control=False,
            enable_region_completion=False,
            enable_troop_advantage=False,
            enable_strategic_position=True,
            strategic_position_weight=0.1,  # Higher weight to see effect
        ), "Strategic Position Only"),
    ]

    for config, name in components:
        run_example_game(config, name)


def main():
    """Run all examples."""
    import sys

    print("\n🎮 Parallel Risk - Reward Shaping Examples\n")

    if len(sys.argv) > 1 and sys.argv[1] == '--components':
        # Show individual components with high weights
        demonstrate_reward_components()
    else:
        # Compare standard configs
        compare_configs()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("\nTip: Run with --components flag to see individual reward components")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
