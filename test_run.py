from env.parallel_risk_env import ParallelRiskEnv
import numpy as np

def random_policy(obs_space):
    return lambda obs: obs_space.sample()

if __name__ == "__main__":
    env = ParallelRiskEnv()

    observations, _ = env.reset()
    print(f"Initial Observations: {observations}")

    policies = {agent: random_policy(env.action_spaces[agent]) for agent in env.agents}

    for step in range(10):
        actions = {agent: policies[agent](observations[agent]) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f"Step {step}:")
        print(f"  Actions: {actions}")
        print(f"  Rewards: {rewards}")
        print(f"  Terminations: {terminations}")
        if not env.agents:
            print("All agents done.")
            break
