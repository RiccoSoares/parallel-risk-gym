"""MCTS agent guided by a trained GNN (AlphaZero-style).

Wires a `GCNPolicy` into `DuctMCTS` in three places:
  - as the leaf value estimator (replaces random rollouts),
  - as the action sampler for progressive widening,
  - as the prior over actions in the PUCT selection formula.

The value and prior hooks are served by `GNNEvaluator`, a per-search cache
that runs one forward per (state, agent) and answers every value_fn /
policy_fn call about that graph from the cached outputs.

Preserves the existing MCTS baseline: when this class is not used,
`DuctMCTS` behaves identically to before (see the `policy_fn=None` branch
in `parallel_risk/agents/mcts_agent.py`).

Requires Phase 2 dependencies: pip install -r requirements/torchrl.txt
"""

import numpy as np

try:
    import torch
    from torch_geometric.data import Batch, Data
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
                 action_budget: int, device, max_regions: int = None,
                 max_actions_per_turn: int = None):
        self.policy = policy
        self.decoder = decoder
        self.map_config = map_config
        self.action_budget = action_budget
        self.device = device
        self.max_regions = (
            max_regions if max_regions is not None else len(map_config.regions)
        )
        # Env-side padding target. Must be >= action_budget. Default max(10,
        # action_budget) matches the env's default max_actions_per_turn while
        # scaling up automatically for K > 10.
        self.max_actions_per_turn = (
            max_actions_per_turn if max_actions_per_turn is not None
            else max(10, action_budget)
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
        actions_array = np.zeros((self.max_actions_per_turn, 3), dtype=np.int32)
        actions_array[:self.action_budget] = action_array_raw.astype(np.int32)
        return {'num_actions': self.action_budget, 'actions': actions_array}


class EvalEntry:
    """Cached outputs of one GNN forward for one (state, agent) graph.

    Attributes:
        action_logits: list of `action_budget` dicts with 'source' [n],
            'dest' [n] and 'troops' [1, max_troops] tensors for this graph
            only (a single-graph forward, or the slice of a paired or
            cross-game batched one).
        batched: the graph as `ActionDecoder` expects a pre-batched
            observation (`batched_obs=`): a `Data` carrying x, edge_index,
            global_features and a zero `batch` vector.
        value_tensor: the value head output for this graph, shape [1].
        value: python float of `value_tensor`, materialized on first use so
            a state whose prior is needed but whose value never is does not
            pay a device sync (`evaluate_many` fills it eagerly from its one
            host transfer).
        slot_batched, slot_logits: the same graph and logits in slot-major
            layout (K copies of the graph, copy k carrying slot k's logits),
            built by `GNNEvaluator._slot_major` on the first prior request.
    """

    __slots__ = ('action_logits', 'batched', 'value_tensor', 'value',
                 'slot_batched', 'slot_logits')

    def __init__(self, action_logits, batched, value_tensor):
        self.action_logits = action_logits
        self.batched = batched
        self.value_tensor = value_tensor
        self.value = None
        self.slot_batched = None
        self.slot_logits = None


class GNNEvaluator:
    """Per-search cache of GNN outputs: one forward per (state, agent).

    `DuctMCTS` asks the network about the same (state, agent) graph up to
    four times per node: the leaf value for agent_0 and the prior of every
    action progressive widening samples for each agent. All of them are
    functions of one forward pass, so the evaluator runs it once, keeps the
    outputs, and serves `value_fn` / `policy_fn` from the cache. The action
    log-prob is computed with `decoder.compute_log_probs` on the cached
    logits through its pre-batched geometry path (`batched_obs=`), so every
    number the search consumes is bit-identical to what the uncached
    closures produced.

    The prior of a K-slot action is evaluated in slot-major layout: a batch
    of K copies of the graph where copy k carries slot k's logits and slot
    k's (source, dest, troops), so one decoder call does three masked
    log-softmaxes over [K, n] instead of K x 3 over [1, n]. Each row is the
    same computation on the same numbers, and the sum over K is the same
    reduction, so the result is bit-identical to the per-slot loop
    (checked exactly in tests/test_mcts_gnn_agent.py) at about a third of
    the cost.

    Interface:
        new_search()                    clear the cache. `DuctMCTS.make_root`
                                        calls it, so the cache never outlives
                                        the search it was filled in.
        has(game_state, agent_id)       -> bool, is the graph cached. The
                                        search generators of `DuctMCTS` ask
                                        this to decide whether to yield a
                                        request.
        evaluate(game_state, agent_id)  -> EvalEntry, from cache or one forward.
        value_fn(game_state, agent_id)  -> float    (DuctMCTS.value_fn contract)
        policy_fn(game_state, agent_id, action_dict) -> float
                                                    (DuctMCTS.policy_fn contract)
        evaluate_many(requests, policy, device)     (static) ONE forward for
                                        the (evaluator, game_state, agent_id)
                                        requests of many games, possibly on
                                        different maps, filling each
                                        evaluator's cache. Used by the
                                        lockstep driver only.

    Cache key: (ownership bytes, troops bytes, turn, that agent's available
    income, agent id). Those five things determine the observation, hence the
    graph. Entries are only valid while the policy weights are frozen, which
    holds within one search.

    pair_agents=True forwards both agents' graphs of a state in one batch of
    2, halving the forward count. The batched CPU kernels reduce in a
    different order, so priors and values then differ from the single-graph
    forward (observed: up to 8e-6 on summed log-probs, 2e-7 on values) and
    a PUCT argmax can flip; it is off by default because the golden harness
    requires bit-identity.

    Counters `n_forwards`, `n_graphs`, `n_hits`, `n_misses` describe the
    current search (reset by `new_search`).
    """

    def __init__(self, policy, decoder, map_config: MapConfig,
                 simulator: RiskSimulator, action_budget: int, device,
                 max_regions: int = None, pair_agents: bool = False):
        self.policy = policy
        self.decoder = decoder
        self.map_config = map_config
        self.simulator = simulator
        self.action_budget = action_budget
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_regions = (
            max_regions if max_regions is not None else len(map_config.regions)
        )
        self.pair_agents = pair_agents
        self._cache = {}
        # Slot-major geometry shared by every entry of this map: node k*n+i
        # is territory i of copy k. The replicated edge_index is derived
        # from the first graph seen (it is a map constant).
        n = map_config.n_territories
        self._slot_batch = torch.arange(
            action_budget, dtype=torch.long, device=self.device).repeat_interleave(n)
        self._slot_edge_index = None
        self.new_search()

    def new_search(self) -> None:
        """Drop every cached evaluation and reset the per-search counters."""
        self._cache.clear()
        self.n_forwards = 0
        self.n_graphs = 0
        self.n_hits = 0
        self.n_misses = 0

    @staticmethod
    def _key(game_state: dict, agent_id: str) -> tuple:
        return (
            game_state['territory_ownership'].tobytes(),
            game_state['territory_troops'].tobytes(),
            int(game_state['turn_number']),
            int(game_state['available_income'][agent_id]),
            agent_id,
        )

    def has(self, game_state: dict, agent_id: str) -> bool:
        """True if (state, agent) is cached, i.e. value_fn/policy_fn will not forward."""
        return self._key(game_state, agent_id) in self._cache

    def _graph(self, game_state: dict, agent_id: str):
        obs = self.simulator.state_to_obs(game_state, agent_id)
        return env_to_graph(obs, self.map_config, self.device,
                            max_regions=self.max_regions)

    @staticmethod
    def _collate(graphs, device):
        """Batch graphs of any sizes into one Data on `device`.

        Value-identical to `Batch.from_data_list(graphs).to(device)` for the
        fields the policy reads (x, edge_index, global_features, batch).
        """
        sizes = [int(g.num_nodes) for g in graphs]
        pieces = []
        offset = 0
        for g, n in zip(graphs, sizes):
            pieces.append(g.edge_index if offset == 0 else g.edge_index + offset)
            offset += n
        data = Data(x=torch.cat([g.x for g in graphs]).to(device),
                    edge_index=torch.cat(pieces, dim=1).to(device),
                    num_nodes=offset)
        data.global_features = torch.cat([g.global_features for g in graphs]).to(device)
        data.batch = torch.repeat_interleave(
            torch.arange(len(graphs), dtype=torch.long), torch.tensor(sizes)).to(device)
        return data

    @staticmethod
    def evaluate_many(requests, policy=None, device=None) -> int:
        """Fill many evaluators' caches from ONE batched forward.

        `requests` is a list of (evaluator, game_state, agent_id) triples,
        typically every pending request of a lockstep round. Requests that
        are already cached are skipped and the rest are deduplicated by
        (map, cache key), so identical positions from different games of
        one map share a row. Graphs of different maps and sizes are collated
        into one batch. The forward runs `policy` (default: the first
        evaluator's) on `device` (default: where its parameters live); the
        outputs are moved to the evaluators' own device with one transfer
        per tensor, sliced per graph and stored as `EvalEntry` objects with
        the layout the single-graph path produces, so value_fn/policy_fn
        work unchanged. Values are materialized from that same transfer, so
        value_fn never syncs later.

        Numerically this is the paired forward's situation at batch size B:
        the batched kernels reduce in a different order than a single-graph
        forward (about 1e-5 on summed log-probs on the CPU; larger on CUDA,
        see tests/test_lockstep.py), so the synchronous path never uses it.
        Returns the number of graphs forwarded.
        """
        todo = {}    # (map identity, max_regions, cache key) -> (evaluator, state, agent)
        fills = []   # (evaluator, cache key, todo key)
        for evaluator, game_state, agent_id in requests:
            key = evaluator._key(game_state, agent_id)
            if key in evaluator._cache:
                evaluator.n_hits += 1
                continue
            evaluator.n_misses += 1
            todo_key = (id(evaluator.map_config), evaluator.max_regions, key)
            if todo_key not in todo:
                todo[todo_key] = (evaluator, game_state, agent_id)
            fills.append((evaluator, key, todo_key))
        if not todo:
            return 0

        todo_keys = list(todo)
        owners = [todo[k] for k in todo_keys]
        graphs = [ev._graph(state, agent_id) for ev, state, agent_id in owners]
        if policy is None:
            policy = owners[0][0].policy
        if device is None:
            device = next(policy.parameters()).device
        device = torch.device(device) if isinstance(device, str) else device
        with torch.no_grad():
            action_logits, values, _ = policy(GNNEvaluator._collate(graphs, device))

        # One tensor per head component, brought to the entries' device.
        out_device = owners[0][0].device
        source = torch.stack([d['source'] for d in action_logits])   # [K, total_nodes]
        dest = torch.stack([d['dest'] for d in action_logits])       # [K, total_nodes]
        troops = torch.stack([d['troops'] for d in action_logits])   # [K, B, max_troops]
        if out_device != device:
            source, dest, troops, values = (source.to(out_device), dest.to(out_device),
                                            troops.to(out_device), values.to(out_device))
        value_list = values.reshape(-1).tolist()

        entries = {}
        offset = 0
        for i, (todo_key, (ev, _, _), graph) in enumerate(zip(todo_keys, owners, graphs)):
            n = int(graph.num_nodes)
            logits_i = [{'source': source[k, offset:offset + n],
                         'dest': dest[k, offset:offset + n],
                         'troops': troops[k, i:i + 1]} for k in range(source.size(0))]
            entry = EvalEntry(logits_i, ev._as_batched(graph), values[i])
            entry.value = float(value_list[i])
            entries[todo_key] = entry
            ev.n_graphs += 1
            offset += n
        for ev in {id(ev): ev for ev, _, _ in owners}.values():
            ev.n_forwards += 1
        for ev, key, todo_key in fills:
            ev._cache[key] = entries[todo_key]
        return len(graphs)

    @staticmethod
    def _as_batched(graph):
        """Give a single graph the `batch` vector a batch-of-1 would carry.

        Value-identical to `Batch.from_data_list([graph])` for the fields the
        policy and the decoder read (x, edge_index, global_features, batch)
        without the collate cost.
        """
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long,
                                  device=graph.x.device)
        return graph

    def evaluate(self, game_state: dict, agent_id: str) -> EvalEntry:
        """Return the cached forward outputs for (state, agent), computing on a miss."""
        key = self._key(game_state, agent_id)
        entry = self._cache.get(key)
        if entry is not None:
            self.n_hits += 1
            return entry
        self.n_misses += 1
        if self.pair_agents:
            self._forward_pair(game_state)
        else:
            self._forward_single(game_state, agent_id, key)
        return self._cache[key]

    def _forward_single(self, game_state: dict, agent_id: str, key: tuple) -> None:
        batched = self._as_batched(self._graph(game_state, agent_id))
        with torch.no_grad():
            action_logits, values, _ = self.policy(batched)
        self.n_forwards += 1
        self.n_graphs += 1
        self._cache[key] = EvalEntry(action_logits, batched, values[0])

    def _forward_pair(self, game_state: dict) -> None:
        """One batch-of-2 forward filling the entries of both agents."""
        agents = self.simulator.AGENTS
        graphs = [self._graph(game_state, a) for a in agents]
        batched = Batch.from_data_list(graphs)
        with torch.no_grad():
            action_logits, values, _ = self.policy(batched)
        self.n_forwards += 1
        self.n_graphs += len(graphs)
        n = self.map_config.n_territories
        for gi, agent_id in enumerate(agents):
            nodes = slice(gi * n, (gi + 1) * n)
            logits_i = [{'source': d['source'][nodes],
                         'dest': d['dest'][nodes],
                         'troops': d['troops'][gi:gi + 1]} for d in action_logits]
            self._cache[self._key(game_state, agent_id)] = EvalEntry(
                logits_i, self._as_batched(graphs[gi]), values[gi])

    def _slot_major(self, entry: EvalEntry) -> None:
        """Build the K-copy geometry and slot-major logits of an entry once."""
        graph = entry.batched
        K = self.action_budget
        n = self.map_config.n_territories
        if self._slot_edge_index is None:
            self._slot_edge_index = torch.cat(
                [graph.edge_index + k * n for k in range(K)], dim=1)
        rep = Data(x=graph.x.repeat(K, 1), edge_index=self._slot_edge_index,
                   num_nodes=K * n)
        rep.global_features = graph.global_features.repeat(K, 1)
        rep.batch = self._slot_batch
        entry.slot_batched = rep
        entry.slot_logits = [{
            'source': torch.cat([d['source'] for d in entry.action_logits]),
            'dest': torch.cat([d['dest'] for d in entry.action_logits]),
            'troops': torch.cat([d['troops'] for d in entry.action_logits], dim=0),
        }]

    def value_fn(self, game_state: dict, agent_id: str) -> float:
        """V(state) from agent_id's perspective.

        `DuctMCTS._rollout` always calls this for 'agent_0' and negates for
        'agent_1' (zero-sum). Because obs is agent-relative and the GNN was
        trained via self-play, the raw value output is already in the
        observing agent's frame; no sign flip needed on our side.
        """
        entry = self.evaluate(game_state, agent_id)
        if entry.value is None:
            entry.value = float(entry.value_tensor.item())
        return entry.value

    def policy_fn(self, game_state: dict, agent_id: str, action_dict: dict) -> float:
        """log P(action | state, agent) under the GNN, summed over the K slots.

        Assumes `action_dict['num_actions'] == action_budget` (the GNN always
        emits exactly `action_budget` slots). Fails loudly otherwise rather
        than silently misaligning shapes.
        """
        num_actions = int(action_dict['num_actions'])
        if num_actions != self.action_budget:
            raise ValueError(
                f"policy_fn expects num_actions == action_budget ({self.action_budget}), "
                f"got {num_actions}. This adapter only supports GNN-emitted actions."
            )
        entry = self.evaluate(game_state, agent_id)
        if entry.slot_batched is None:
            self._slot_major(entry)
        actions_tensor = torch.as_tensor(
            action_dict['actions'][:self.action_budget], dtype=torch.long, device=self.device
        ).unsqueeze(1)  # [action_budget, 1, 3]: copy k evaluates slot k
        with torch.no_grad():
            log_probs = self.decoder.compute_log_probs(
                entry.slot_logits,
                actions_tensor,
                self._slot_batch,
                batched_obs=entry.slot_batched,
            )  # [action_budget, 1]
        return float(log_probs.sum().item())


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
        max_actions_per_turn: int = None,
        pair_agents: bool = False,
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

        pair_agents:
            - False (default): one single-graph forward per (state, agent);
              priors and values bit-identical to the uncached closures.
            - True: both agents' graphs of a state share one batch-of-2
              forward (half the forwards, results within ~1e-5). See
              `GNNEvaluator`.
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
        # Env-side padding target for returned actions. Defaults to
        # max(10, action_budget) so K<=10 keeps the classic (10, 3) shape and
        # K>10 automatically expands (env must also be constructed with a
        # matching max_actions_per_turn).
        self.max_actions_per_turn = (
            max_actions_per_turn if max_actions_per_turn is not None
            else max(10, action_budget)
        )

        policy.eval()

        if pw_sampler == 'masked_random':
            from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib
            sampler = MaskedRandomAgentRLlib(
                n_territories=map_config.n_territories,
                adjacency_matrix=map_config.adjacency_matrix,
                action_budget=action_budget,
                max_troops=max_troops,
                max_actions_per_turn=self.max_actions_per_turn,
            )
        elif pw_sampler == 'gnn':
            sampler = GNNActionSampler(
                policy=policy,
                decoder=decoder,
                map_config=map_config,
                action_budget=action_budget,
                device=self.device,
                max_regions=self.max_regions,
                max_actions_per_turn=self.max_actions_per_turn,
            )
        else:
            raise ValueError(f"pw_sampler must be 'masked_random' or 'gnn', got {pw_sampler}")
        simulator = RiskSimulator(map_config, max_turns=max_turns)

        self.use_value_fn = use_value_fn
        # One forward per (state, agent) per search; DuctMCTS.make_root
        # resets it through the `evaluator` hook.
        self.evaluator = GNNEvaluator(
            policy=policy,
            decoder=decoder,
            map_config=map_config,
            simulator=simulator,
            action_budget=action_budget,
            device=self.device,
            max_regions=self.max_regions,
            pair_agents=pair_agents,
        )

        self.mcts = DuctMCTS(
            simulator=simulator,
            action_sampler_0=sampler,
            action_sampler_1=sampler,
            uct_c=uct_c,
            pw_alpha=pw_alpha,
            max_rollout_turns=max_rollout_turns,
            value_fn=self.evaluator.value_fn if use_value_fn else None,
            policy_fn=self.evaluator.policy_fn,
            c_puct=c_puct,
            evaluator=self.evaluator,
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
