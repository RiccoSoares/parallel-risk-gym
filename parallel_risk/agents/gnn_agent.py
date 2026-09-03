"""GNN policy agent wrapping GCNPolicy for evaluation and comparison.

Loads a trained GCNPolicy from a PyTorch checkpoint and wraps it
behind a simple get_action() interface compatible with the evaluation
infrastructure.

Requires Phase 2 dependencies: pip install -r requirements/torchrl.txt
"""

import numpy as np

try:
    import torch
    from torch_geometric.data import Batch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from parallel_risk.env.map_config import MapConfig
from parallel_risk.training.torchrl.graph_wrapper import env_to_graph


class GNNAgent:
    """GNN policy agent wrapping GCNPolicy + ActionDecoder.

    Converts flat dict observations (from env._get_observation()) to
    PyG graphs, runs the GCN forward pass, and decodes actions with
    autoregressive masking.

    Usage:
        agent = GNNAgent.from_checkpoint('checkpoints/.../checkpoint_000050.pt',
                                          map_config)
        action = agent.get_action(obs['agent_0'])
    """

    def __init__(
        self,
        policy,
        decoder,
        map_config: MapConfig,
        action_budget: int,
        device: 'torch.device',
        deterministic: bool = False,  # match training eval: stochastic avoids argmax collapse
    ):
        self.policy = policy
        self.decoder = decoder
        self.map_config = map_config
        self.action_budget = action_budget
        self.device = device
        self.deterministic = deterministic

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        map_config: MapConfig,
        device: str = 'cpu',
        deterministic: bool = False,  # match training eval: stochastic avoids argmax collapse
    ) -> 'GNNAgent':
        """Load a GCNPolicy from a .pt checkpoint file.

        Args:
            checkpoint_path: Path to a .pt checkpoint saved by PPOTrainer.
            map_config: MapConfig for the environment the checkpoint was trained on.
            device: Torch device string ('cpu' or 'cuda').
            deterministic: If True, uses argmax action selection (greedy).

        Returns:
            Loaded GNNAgent ready for inference.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is not installed. "
                "Install Phase 2 dependencies: pip install -r requirements/torchrl.txt"
            )

        from parallel_risk.models.gnn_gcn import GCNPolicy
        from parallel_risk.models.action_decoder import ActionDecoder

        torch_device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=torch_device)
        config = checkpoint['config']

        n_regions = len(map_config.regions)
        node_features_dim = 3 + n_regions   # troops_norm, ownership, in_degree, region_one_hot...
        global_features_dim = 2 + n_regions  # income_norm, turn_norm, region_control...

        action_budget = config['env'].get('action_budget', 5)
        policy = GCNPolicy(
            node_features_dim=node_features_dim,
            global_features_dim=global_features_dim,
            hidden_dim=config['model'].get('hidden_dim', 128),
            num_layers=config['model'].get('num_layers', 3),
            action_budget=action_budget,
            max_troops=20,
            dropout=config['model'].get('dropout', 0.1),
        )
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.to(torch_device)
        policy.eval()

        decoder = ActionDecoder(action_budget=action_budget, max_troops=20)

        return cls(
            policy=policy,
            decoder=decoder,
            map_config=map_config,
            action_budget=action_budget,
            device=torch_device,
            deterministic=deterministic,
        )

    def get_action(self, obs: dict) -> dict:
        """Generate an action from a flat dict observation.

        Args:
            obs: Flat dict from env._get_observation() with keys:
                 territory_ownership, territory_troops, adjacency_matrix,
                 available_income, turn_number, region_control.

        Returns:
            Action dict {'num_actions': int, 'actions': np.ndarray((10,3))}.
        """
        graph = env_to_graph(obs, self.map_config, self.device)
        batched = Batch.from_data_list([graph])

        with torch.no_grad():
            action_logits, _, _ = self.policy(batched)

        actions_tensor, _ = self.decoder.decode_actions(
            action_logits,
            batched.batch,
            deterministic=self.deterministic,
            return_log_probs=False,
            observations=[graph],
        )

        # actions_tensor: [1, action_budget, 3]
        action_array_raw = actions_tensor[0].cpu().numpy()  # [action_budget, 3]

        actions_array = np.zeros((10, 3), dtype=np.int32)
        actions_array[:self.action_budget] = action_array_raw.astype(np.int32)

        return {'num_actions': self.action_budget, 'actions': actions_array}
