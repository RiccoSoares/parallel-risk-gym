"""
Vectorized environment for parallel data collection.

Runs multiple ParallelRiskEnv instances in separate processes for faster
experience collection during training.
"""

import multiprocessing as mp
from typing import List, Dict, Any, Callable, Optional
import numpy as np
import sys

import torch
from torch_geometric.data import Data

from parallel_risk import ParallelRiskEnv
from parallel_risk.env.reward_shaping import RewardShapingConfig

# Use 'spawn' on macOS to avoid fork issues
if sys.platform == 'darwin':
    mp_context = mp.get_context('spawn')
else:
    mp_context = mp.get_context('forkserver')
Process = mp_context.Process


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn: Callable[[], ParallelRiskEnv]
):
    """
    Worker process that runs a single environment.

    Communicates with main process via pipes.
    """
    parent_remote.close()
    env = env_fn()
    map_config = env.map_config

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                obs, rewards, terminateds, truncateds, infos = env.step(data)

                # Check if episode ended (auto-reset)
                done = terminateds.get('__all__', False) or truncateds.get('__all__', False)

                if done:
                    # Auto-reset and return new observations
                    new_obs, new_infos = env.reset()
                    graph_data = {
                        agent: _obs_to_graph_data(new_obs[agent], map_config)
                        for agent in new_obs.keys()
                    }
                    # Store final info in 'terminal_info'
                    for agent in new_infos.keys():
                        new_infos[agent]['terminal_info'] = infos.get(agent, {})
                    infos = new_infos
                else:
                    # Convert observations to graph format (as numpy for pickling)
                    graph_data = {
                        agent: _obs_to_graph_data(obs[agent], map_config)
                        for agent in obs.keys()
                    }

                remote.send((graph_data, rewards, terminateds, truncateds, infos))

            elif cmd == 'reset':
                obs, infos = env.reset(seed=data.get('seed'), options=data.get('options'))
                graph_data = {
                    agent: _obs_to_graph_data(obs[agent], map_config)
                    for agent in obs.keys()
                }
                remote.send((graph_data, infos))

            elif cmd == 'close':
                env.close()
                remote.close()
                break

            elif cmd == 'get_spaces':
                remote.send((env.observation_spaces, env.action_spaces))

            elif cmd == 'get_map_config':
                # Send serializable map config info
                remote.send({
                    'n_territories': map_config.n_territories,
                    'n_regions': len(map_config.regions),
                    'adjacency_matrix': map_config.adjacency_matrix,
                    'regions': map_config.regions,
                    'region_bonuses': map_config.region_bonuses,
                })

            else:
                raise ValueError(f"Unknown command: {cmd}")

        except EOFError:
            break


def _obs_to_graph_data(obs: Dict[str, Any], map_config) -> Dict[str, Any]:
    """
    Convert observation to serializable graph data (numpy arrays).

    We can't pickle torch tensors across processes easily, so we convert
    to numpy here and reconstruct tensors in the main process.
    """
    n_territories = map_config.n_territories
    n_regions = len(map_config.regions)

    ownership = obs['territory_ownership']
    troops = obs['territory_troops']
    adjacency_matrix = obs['adjacency_matrix']
    available_income = obs['available_income']
    turn_number = obs['turn_number']
    region_control = obs['region_control']

    # Node features
    troops_log_normalized = np.log1p(troops) / np.log1p(100.0)
    in_degree = adjacency_matrix.sum(axis=1)
    in_degree_normalized = in_degree / n_territories

    # Region membership
    territory_to_region = np.zeros((n_territories, n_regions), dtype=np.float32)
    for region_idx, (region_name, territories) in enumerate(map_config.regions.items()):
        for territory_id in territories:
            territory_to_region[territory_id, region_idx] = 1.0

    node_features = np.stack([
        troops_log_normalized,
        ownership,
        in_degree_normalized,
    ] + [territory_to_region[:, i] for i in range(n_regions)], axis=1).astype(np.float32)

    # Edge index
    edge_sources, edge_targets = [], []
    for i in range(n_territories):
        for j in range(n_territories):
            if adjacency_matrix[i, j] == 1:
                edge_sources.append(i)
                edge_targets.append(j)
    edge_index = np.array([edge_sources, edge_targets], dtype=np.int64)

    # Global features
    global_features = np.concatenate([
        available_income.astype(np.float32) / 20.0,
        turn_number.astype(np.float32) / 100.0,
        region_control.astype(np.float32),
    ])

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'global_features': global_features,
        'n_territories': n_territories,
        'n_regions': n_regions,
    }


def _graph_data_to_torch(data: Dict[str, Any], device: torch.device) -> Data:
    """Convert numpy graph data back to PyTorch Geometric Data object."""
    x = torch.tensor(data['node_features'], dtype=torch.float32, device=device)
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long, device=device)
    global_features = torch.tensor(data['global_features'], dtype=torch.float32, device=device)

    if global_features.dim() == 1:
        global_features = global_features.unsqueeze(0)

    graph = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=data['n_territories'],
    )
    graph.global_features = global_features
    graph.n_territories = data['n_territories']
    graph.n_regions = data['n_regions']

    return graph


class VecEnv:
    """
    Vectorized environment running multiple ParallelRiskEnv in parallel.

    Each environment runs in its own process, allowing true parallelism
    for data collection.

    Usage:
        def make_env():
            return ParallelRiskEnv(map_name='simple_6')

        vec_env = VecEnv([make_env for _ in range(4)])
        obs, infos = vec_env.reset()

        # obs is now a list of dicts, one per environment
        # Each dict maps agent_id -> PyG Data object
    """

    def __init__(
        self,
        env_fns: List[Callable[[], ParallelRiskEnv]],
        device: Optional[torch.device] = None
    ):
        """
        Initialize vectorized environment.

        Args:
            env_fns: List of callables that create ParallelRiskEnv instances
            device: Torch device for graph tensors
        """
        self.num_envs = len(env_fns)
        self.device = device if device is not None else torch.device('cpu')
        self.waiting = False

        # Create pipes for communication (using correct context)
        self.remotes, self.work_remotes = zip(*[mp_context.Pipe() for _ in range(self.num_envs)])

        # Start worker processes
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            process = Process(
                target=_worker,
                args=(work_remote, remote, env_fn),
                daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get map config from first environment
        self.remotes[0].send(('get_map_config', None))
        self.map_config_info = self.remotes[0].recv()

    def reset(self, seed: Optional[int] = None) -> tuple:
        """
        Reset all environments.

        Args:
            seed: Optional seed (each env gets seed+i)

        Returns:
            observations: List of dicts mapping agent -> PyG Data
            infos: List of info dicts
        """
        for i, remote in enumerate(self.remotes):
            env_seed = seed + i if seed is not None else None
            remote.send(('reset', {'seed': env_seed}))

        results = [remote.recv() for remote in self.remotes]

        all_obs = []
        all_infos = []
        for graph_data, infos in results:
            # Convert numpy to torch
            obs = {
                agent: _graph_data_to_torch(data, self.device)
                for agent, data in graph_data.items()
            }
            all_obs.append(obs)
            all_infos.append(infos)

        return all_obs, all_infos

    def step_async(self, actions: List[Dict[str, Any]]):
        """
        Send actions to all environments (non-blocking).

        Args:
            actions: List of action dicts, one per environment
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self) -> tuple:
        """
        Wait for all environments to complete step.

        Returns:
            observations: List of dicts mapping agent -> PyG Data
            rewards: List of reward dicts
            terminateds: List of terminated dicts
            truncateds: List of truncated dicts
            infos: List of info dicts
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        all_obs, all_rewards, all_terminateds, all_truncateds, all_infos = [], [], [], [], []

        for graph_data, rewards, terminateds, truncateds, infos in results:
            obs = {
                agent: _graph_data_to_torch(data, self.device)
                for agent, data in graph_data.items()
            }
            all_obs.append(obs)
            all_rewards.append(rewards)
            all_terminateds.append(terminateds)
            all_truncateds.append(truncateds)
            all_infos.append(infos)

        return all_obs, all_rewards, all_terminateds, all_truncateds, all_infos

    def step(self, actions: List[Dict[str, Any]]) -> tuple:
        """
        Step all environments synchronously.

        Args:
            actions: List of action dicts, one per environment

        Returns:
            Same as step_wait()
        """
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        """Close all environments and worker processes."""
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except BrokenPipeError:
                pass

        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()

    @property
    def observation_space(self):
        """Return observation space description."""
        n_territories = self.map_config_info['n_territories']
        n_regions = self.map_config_info['n_regions']
        feature_dim = 3 + n_regions

        return {
            'type': 'graph',
            'node_features_dim': feature_dim,
            'global_features_dim': 2 + n_regions,
            'n_territories': n_territories,
            'n_regions': n_regions,
        }


class EnvFactory:
    """
    Picklable factory for creating ParallelRiskEnv instances.

    Using a class instead of a closure allows pickling across processes.
    """

    def __init__(
        self,
        map_name: str,
        max_turns: int,
        use_reward_shaping: bool,
        seed: Optional[int]
    ):
        self.map_name = map_name
        self.max_turns = max_turns
        self.use_reward_shaping = use_reward_shaping
        self.seed = seed

    def __call__(self) -> ParallelRiskEnv:
        reward_config = RewardShapingConfig() if self.use_reward_shaping else None
        return ParallelRiskEnv(
            map_name=self.map_name,
            max_turns=self.max_turns,
            seed=self.seed,
            reward_shaping_config=reward_config
        )


def make_vec_env(
    num_envs: int,
    map_name: str = 'simple_6',
    max_turns: int = 50,
    use_reward_shaping: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> VecEnv:
    """
    Create a vectorized environment with multiple workers.

    Args:
        num_envs: Number of parallel environments
        map_name: Map configuration name
        max_turns: Maximum turns per episode
        use_reward_shaping: Enable reward shaping
        seed: Base random seed
        device: Torch device

    Returns:
        VecEnv instance
    """
    env_fns = [
        EnvFactory(
            map_name=map_name,
            max_turns=max_turns,
            use_reward_shaping=use_reward_shaping,
            seed=seed + i if seed is not None else None
        )
        for i in range(num_envs)
    ]

    return VecEnv(env_fns, device=device)
