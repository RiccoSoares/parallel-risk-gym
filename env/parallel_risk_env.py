import pettingzoo
from gymnasium import spaces
import numpy as np

class ParallelRiskEnv(pettingzoo.ParallelEnv):
    metadata = {'name': 'parallel_risk_env_v0'}

    def __init__(self, map_name:str = "dummy_map", num_samples:int = 64):
        self.map_name = map_name
        self.num_samples = num_samples
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents
        #Code below contains dummy logic
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(4) 
            for agent in self.agents
        }
        self._counters = {agent: 0 for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._counters = {agent: 0 for agent in self.agents}
        observations = {
            agent: np.random.rand(5).astype(np.float32)
            for agent in self.agents
        }
        return observations, {}

    def step(self, actions):
        # Dummy logic: if a counter reaches 5, the agent is "done"
        terminations = {}
        truncations = {agent: False for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        for agent, action in actions.items():
            self._counters[agent] += 1
            rewards[agent] = np.random.uniform(-1, 1)
            terminations[agent] = self._counters[agent] >= 5

        self.agents = [agent for agent in self.agents if not terminations[agent]]

        observations = {
            agent: np.random.rand(5).astype(np.float32)
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def render(self):
        print(f"Turn counters: {self._counters}")

    def observe(self, agent):
        pass
