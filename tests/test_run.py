from parallel_risk import ParallelRiskEnv
import numpy as np

def random_policy(env, agent):
    """Generate random valid actions for an agent"""
    action_space = env.action_spaces[agent]
    max_actions = action_space['num_actions'].n - 1  # Max num_actions value
    n_territories = env.map_config.n_territories

    # Random number of actions (0 to max_actions)
    num_actions = np.random.randint(0, max_actions + 1)

    # Generate random action triples
    actions = np.zeros((env.max_actions_per_turn, 3), dtype=np.int32)
    for i in range(num_actions):
        source = np.random.randint(0, n_territories)
        dest = np.random.randint(0, n_territories)
        troops = np.random.randint(1, 6)  # 1-5 troops per action
        actions[i] = [source, dest, troops]

    return {
        'num_actions': num_actions,
        'actions': actions
    }

if __name__ == "__main__":
    env = ParallelRiskEnv()

    observations, infos = env.reset()
    print("=== INITIAL STATE ===")
    env.render()

    print(f"\nInitial Observations (agent_0):")
    print(f"  Ownership: {observations['agent_0']['territory_ownership']}")
    print(f"  Troops: {observations['agent_0']['territory_troops']}")
    print(f"  Income: {observations['agent_0']['available_income'][0]}")

    step_count = 0
    while env.agents and step_count < 50:  # Safety limit
        # Generate actions for all agents
        actions = {agent: random_policy(env, agent) for agent in env.agents}

        print(f"\n{'='*60}")
        print(f"STEP {step_count + 1}")
        print(f"{'='*60}")

        # Display actions
        for agent in env.agents:
            num_act = actions[agent]['num_actions']
            print(f"\n{agent} submitting {num_act} actions:")
            for i in range(num_act):
                src, dst, trp = actions[agent]['actions'][i]
                print(f"  Action {i+1}: ({src}, {dst}, {trp})")

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Display results
        print("\nResults:")
        env.render()

        print("\nRewards:", rewards)
        print("Terminations:", terminations)

        for agent in env.agents if env.agents else list(rewards.keys()):
            if agent in infos and 'invalid_actions' in infos[agent]:
                print(f"{agent} invalid actions: {infos[agent]['invalid_actions']}")

        step_count += 1

        if any(terminations.values()):
            print("\n" + "="*60)
            print("GAME OVER!")
            print("="*60)
            winner = max(rewards, key=rewards.get)
            print(f"Winner: {winner}")
            print(f"Final rewards: {rewards}")
            break

    if step_count >= 50:
        print("\n" + "="*60)
        print("Reached step limit without termination")
        print("="*60)
