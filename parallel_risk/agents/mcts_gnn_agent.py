"""MCTS agent guided by a trained GNN (AlphaZero-style).

Wires a `GCNPolicy` into `DuctMCTS` in three places:
  - as the leaf value estimator (replaces random rollouts),
  - as the action sampler for progressive widening,
  - as the prior over actions in the PUCT selection formula.

Preserves the existing MCTS baseline: when this class is not used,
`DuctMCTS` behaves identically to before (see the `policy_fn=None` branch
in `parallel_risk/agents/mcts_agent.py`).

Requires Phase 2 dependencies: pip install -r requirements/torchrl.txt
"""

import numpy as np

try:
    import torch
    from torch_geometric.data import Batch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from parallel_risk.agents.mcts_agent import DuctMCTS, RiskSimulator
from parallel_risk.env.map_config import MapConfig
from parallel_risk.training.torchrl.graph_wrapper import env_to_graph


class GNNActionSampler:
    """Sampler that draws stochastic actions from a `GCNPolicy`.

    Implements the `.get_action_raw(obs) -> action_dict` contract expected
    by `DuctMCTS`. Stateless — a single instance can be shared by both
    agents (obs is already agent-relative).

    Mirrors the sampling pipeline in `GNNAgent.get_action`; kept as a
    separate class so we can hand the instance to `DuctMCTS` without
    dragging in the `GNNAgent` public API.

    `max_regions` matches the multi-map padding scheme used by
    `PPOTrainer` — required so a policy trained across maps with
    different region counts sees a consistent input feature dim.
    """

    def __init__(self, policy, decoder, map_config: MapConfig,
                 action_budget: int, device, max_regions: int = None):
        self.policy = policy
        self.decoder = decoder
        self.map_config = map_config
        self.action_budget = action_budget
        self.device = device
        self.max_regions = (
            max_regions if max_regions is not None else len(map_config.regions)
        )

    def get_action_raw(self, obs: dict) -> dict:
        graph = env_to_graph(obs, self.map_config, self.device,
                             max_regions=self.max_regions)
        batched = Batch.from_data_list([graph])

        with torch.no_grad():
            action_logits, _, _ = self.policy(batched)

        actions_tensor, _ = self.decoder.decode_actions(
            action_logits,
            batched.batch,
            deterministic=False,
            return_log_probs=False,
            observations=[graph],
        )

        action_array_raw = actions_tensor[0].cpu().numpy()  # [action_budget, 3]
        actions_array = np.zeros((10, 3), dtype=np.int32)
        actions_array[:self.action_budget] = action_array_raw.astype(np.int32)
        return {'num_actions': self.action_budget, 'actions': actions_array}


class MCTSGNNAgent:
    """Decoupled UCT with PUCT selection guided by a trained GNN.

    Usage:
        agent = MCTSGNNAgent.from_checkpoint(
            'checkpoints/multi_map_training/all_3_maps/final.pt',
            env.map_config, simulation_budget=200,
        )
        action = agent.get_action(env.game_state, 'agent_0')
    """

    def __init__(
        self,
        policy,
        decoder,
        map_config: MapConfig,
        simulation_budget: int = 200,
        c_puct: float = 1.4,
        action_budget: int = 5,
        max_troops: int = 20,
        max_turns: int = 100,
        uct_c: float = 1.41,
        pw_alpha: float = 0.5,
        max_rollout_turns: int = 50,
        device: str = 'cpu',
        max_regions: int = None,
        pw_sampler: str = 'masked_random',
        use_value_fn: bool = True,
    ):
        """
        pw_sampler:
            - 'masked_random' (default): use MaskedRandomAgentRLlib for tree
              exploration (progressive widening + root seed). GNN is still
              used as the PUCT prior (via `policy_fn`) and leaf evaluator
              (via `value_fn`, when enabled). Matches AlphaZero's separation
              of "explorer" and "evaluator" and gives the tree the same
              exploration breadth as vanilla MCTS.
            - 'gnn': use GNNActionSampler for PW (older behavior). At
              random init the GNN produces peaked distributions and PW
              barely widens beyond 1-2 modal actions, which severely
              handicaps MCTS+GNN vs MCTS(uniform) at low budget.

        use_value_fn:
            - True (default): use the GNN value head as the leaf evaluator.
              Correct once the value head has been trained.
            - False: fall back to random rollouts at leaves (like vanilla
              MCTS). Useful at cold-start when the GNN's value predictions
              are random noise around 0 and would otherwise wipe out the
              Q signal. Effectively "MCTS + GNN prior only".
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is not installed. "
                "Install Phase 2 dependencies: pip install -r requirements/torchrl.txt"
            )

        self.policy = policy
        self.decoder = decoder
        self.map_config = map_config
        self.action_budget = action_budget
        self.simulation_budget = simulation_budget
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_regions = (
            max_regions if max_regions is not None else len(map_config.regions)
        )
        self.pw_sampler = pw_sampler

        policy.eval()

        if pw_sampler == 'masked_random':
            from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
            sampler = MaskedRandomAgentRLlib(
                n_territories=map_config.n_territories,
                adjacency_matrix=map_config.adjacency_matrix,
                action_budget=action_budget,
                max_troops=max_troops,
            )
        elif pw_sampler == 'gnn':
            sampler = GNNActionSampler(
                policy=policy,
                decoder=decoder,
                map_config=map_config,
                action_budget=action_budget,
                device=self.device,
                max_regions=self.max_regions,
            )
        else:
            raise ValueError(f"pw_sampler must be 'masked_random' or 'gnn', got {pw_sampler}")
        simulator = RiskSimulator(map_config, max_turns=max_turns)

        self.use_value_fn = use_value_fn
        value_fn = self._build_value_fn(simulator) if use_value_fn else None

        self.mcts = DuctMCTS(
            simulator=simulator,
            action_sampler_0=sampler,
            action_sampler_1=sampler,
            uct_c=uct_c,
            pw_alpha=pw_alpha,
            max_rollout_turns=max_rollout_turns,
            value_fn=value_fn,
            policy_fn=self._build_policy_fn(simulator),
            c_puct=c_puct,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        map_config: MapConfig,
        simulation_budget: int = 200,
        c_puct: float = 1.4,
        device: str = 'cpu',
        max_regions: int = None,
        pw_sampler: str = 'masked_random',
        **kwargs,
    ) -> 'MCTSGNNAgent':
        """Load a GCNPolicy from a .pt checkpoint and wrap it.

        Reuses the same checkpoint schema and config-unpacking as
        `GNNAgent.from_checkpoint` so both loaders stay in lockstep.

        `max_regions` should match the training-time padding when the
        checkpoint was produced by multi-map training. If None, defaults
        to this map's own region count (correct for single-map checkpoints).
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
        # Prefer the max_regions captured in the checkpoint's model kwargs
        # (multi-map trainers write it there). Fall back to caller override,
        # then this map's own region count for legacy single-map checkpoints.
        model_cfg = config.get('model', {})
        ckpt_max_regions = model_cfg.get('max_regions')
        effective_max_regions = (
            max_regions if max_regions is not None
            else ckpt_max_regions if ckpt_max_regions is not None
            else n_regions
        )
        node_features_dim = 3 + effective_max_regions
        global_features_dim = 2 + effective_max_regions

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
            simulation_budget=simulation_budget,
            c_puct=c_puct,
            action_budget=action_budget,
            device=torch_device,
            max_regions=effective_max_regions,
            pw_sampler=pw_sampler,
            **kwargs,
        )

    def get_action(self, game_state: dict, agent_id: str) -> dict:
        """Run PUCT-guided MCTS and return the best action for agent_id."""
        root = self.mcts.make_root(game_state)
        self.mcts.run(root, self.simulation_budget)
        return self.mcts.best_action(root, agent_id)

    # ------------------------------------------------------------------
    # Neural hooks bound to DuctMCTS
    # ------------------------------------------------------------------

    def _build_value_fn(self, simulator: RiskSimulator):
        """Closure returning V(state) from agent_id's perspective.

        `DuctMCTS._rollout` always calls this for 'agent_0' and negates for
        'agent_1' (zero-sum). Because obs is agent-relative and the GNN was
        trained via self-play, the raw value output is already in the
        observing agent's frame — no sign flip needed on our side.
        """
        policy = self.policy
        map_config = self.map_config
        device = self.device
        max_regions = self.max_regions

        def value_fn(game_state: dict, agent_id: str) -> float:
            obs = simulator.state_to_obs(game_state, agent_id)
            graph = env_to_graph(obs, map_config, device, max_regions=max_regions)
            batched = Batch.from_data_list([graph])
            with torch.no_grad():
                _, values, _ = policy(batched)
            return float(values.squeeze().item())

        return value_fn

    def _build_policy_fn(self, simulator: RiskSimulator):
        """Closure returning log P(action | state, agent) under the GNN.

        Assumes `action_dict['num_actions'] == self.action_budget` (the GNN
        always emits exactly `action_budget` slots). Fails loudly otherwise
        rather than silently misaligning shapes.
        """
        policy = self.policy
        decoder = self.decoder
        map_config = self.map_config
        device = self.device
        action_budget = self.action_budget
        max_regions = self.max_regions

        def policy_fn(game_state: dict, agent_id: str, action_dict: dict) -> float:
            num_actions = int(action_dict['num_actions'])
            if num_actions != action_budget:
                raise ValueError(
                    f"policy_fn expects num_actions == action_budget ({action_budget}), "
                    f"got {num_actions}. This adapter only supports GNN-emitted actions."
                )
            obs = simulator.state_to_obs(game_state, agent_id)
            graph = env_to_graph(obs, map_config, device, max_regions=max_regions)
            batched = Batch.from_data_list([graph])
            actions_tensor = torch.as_tensor(
                action_dict['actions'][:action_budget], dtype=torch.long, device=device
            ).unsqueeze(0)  # [1, action_budget, 3]

            with torch.no_grad():
                action_logits, _, _ = policy(batched)
                log_probs = decoder.compute_log_probs(
                    action_logits,
                    actions_tensor,
                    batched.batch,
                    observations=[graph],
                )
            return float(log_probs.sum().item())

        return policy_fn
