"""Lockstep driver: many MCTS+GNN games advanced together, one batched GNN forward per round.

A `DuctMCTS` search can be stepped as a generator that yields an evaluation
request -- the (game_state, agent_id) graphs its evaluator has not cached --
and continues once the cache holds them (see the "Evaluation requests"
section of parallel_risk/agents/mcts_agent.py). `GameBatch` holds up to G
such games, each with its own env, agents, episode generator and private
random state, and runs rounds:

    for each active game: restore its RNG state, resume it until it yields a
        request or finishes, save its RNG state
    serve every pending request with ONE forward (GNNEvaluator.evaluate_many)

Saving and restoring python `random` (the env's action shuffle) and
`np.random` (the samplers, temperature sampling) around every resume means
each game consumes exactly the random stream it would consume when played
alone with its seed, whatever else shares the batch, so the set of games is
independent of G and of the order they are started in. The Dirichlet root
noise draws from a per-game `np.random.default_rng(seed)`. A game that
finishes is replaced by the next pending one so the batch stays full.

What is not bit-identical to the one-game-at-a-time path: network outputs
come from batched forwards, whose reductions differ from single-graph
forwards (about 1e-5 on summed log-priors on the CPU, more on CUDA), so a
PUCT argmax can flip and games can diverge. `GameBatch(single_graph_forwards
=True)` serves each request with the game's own evaluator instead; that
mode is exact and is what tests/test_lockstep.py uses to check the driver.

Three episode kinds share the driver: ExIt self-play
(`exit_self_play.play_episode_exit_gen`), AZ self-play
(`self_play.play_episode_gen`) and evaluation games of MCTS+GNN against
MCTS-uniform (`eval_episode_gen`; the uniform side has no network and runs
synchronously on the CPU inside the same round). `self_play_lockstep_worker`
and `eval_lockstep_worker` are the spawn-pool entry points;
`self_play_games_lockstep` and `play_eval_games_lockstep` the in-process ones.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from parallel_risk import ParallelRiskEnv
from parallel_risk.agents.mcts_agent import MCTSAgent
from parallel_risk.agents.mcts_gnn_agent import GNNEvaluator, MCTSGNNAgent
from parallel_risk.env.map_config import MapRegistry
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models.gnn_gcn import GCNPolicy

AGENT_IDS = ('agent_0', 'agent_1')


# ---------------------------------------------------------------------------
# Devices and random state
# ---------------------------------------------------------------------------

def resolve_device(device) -> str:
    """'auto' (or None) -> 'cuda' if available else 'cpu'; anything else as given."""
    if device is None or str(device).lower() == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return str(device)


def forward_policy(policy, device):
    """The module the batched forwards run: `policy` itself on the CPU, an eval-mode copy elsewhere."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == 'cpu':
        return policy
    return copy.deepcopy(policy).to(device).eval()


def capture_rng_state() -> tuple:
    """Snapshot of the global random streams a game consumes (python `random`, `np.random`)."""
    return random.getstate(), np.random.get_state()


def restore_rng_state(state: tuple) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class LockstepGame:
    """One game inside a `GameBatch`.

    gen: the episode generator. It yields evaluation requests -- lists of
        (game_state, agent_id) -- to be served through `evaluator`, and its
        return value becomes `result` when the game is over.
    evaluator: the `GNNEvaluator` whose cache the requests refer to.
    info: free-form dict for the caller (map name, seed, colour, ...).
    """

    __slots__ = ('gen', 'evaluator', 'info', 'rng_state', 'result', 'done')

    def __init__(self, gen, evaluator, info: Dict[str, Any] = None):
        self.gen = gen
        self.evaluator = evaluator
        self.info = dict(info or {})
        self.rng_state = None
        self.result = None
        self.done = False


class GameBatch:
    """Advance up to `size` games in lockstep, one batched forward per round.

    policy: the GCNPolicy the batched forwards run (already on `device`; see
        `forward_policy`). Every game's agents keep their own CPU policy for
        the synchronous path, which the driver never uses.
    device: where the forwards run. Entries land on each evaluator's own
        device (the CPU for the agents built here), so priors are computed
        with the CPU decoder path and nothing syncs per call.
    single_graph_forwards: serve requests one graph at a time through each
        game's own evaluator (exact, slow) instead of `evaluate_many`.

    Counters over the batch's lifetime: rounds, forwards (batched forwards
    run) and graphs (graphs those forwards contained).
    """

    def __init__(self, policy, device='cpu', size: int = 16,
                 single_graph_forwards: bool = False):
        self.policy = policy
        self.device = torch.device(device) if isinstance(device, str) else device
        self.size = max(1, int(size))
        self.single_graph_forwards = single_graph_forwards
        self.rounds = 0
        self.forwards = 0
        self.graphs = 0

    def play(self, games: Iterable[LockstepGame]) -> List[LockstepGame]:
        """Play every game in `games` to completion; returns them in the order started.

        `games` may be lazy (a generator building envs and agents on demand).
        At most `size` games are in flight; a finished game is replaced by
        the next pending one. Every game must seed itself (env.reset(seed))
        on its first resume to be reproducible on its own. The caller's RNG
        state is restored on return.
        """
        started: List[LockstepGame] = []
        pending = iter(games)
        active: List[LockstepGame] = []
        outer = capture_rng_state()
        try:
            while True:
                while len(active) < self.size:
                    game = next(pending, None)
                    if game is None:
                        break
                    started.append(game)
                    active.append(game)
                if not active:
                    break
                requests = []
                for game in active:
                    request = self._resume(game)
                    if request:
                        requests.extend((game.evaluator, s, a) for s, a in request)
                active = [g for g in active if not g.done]
                if requests:
                    self._serve(requests)
                self.rounds += 1
        finally:
            restore_rng_state(outer)
        return started

    def _resume(self, game: LockstepGame):
        """Run `game` under its own RNG state until it yields a request or finishes."""
        if game.rng_state is not None:
            restore_rng_state(game.rng_state)
        try:
            request = next(game.gen)
        except StopIteration as stop:
            game.result = stop.value
            game.done = True
            request = None
        game.rng_state = capture_rng_state()
        return request

    def _serve(self, requests) -> None:
        if self.single_graph_forwards:
            for evaluator, game_state, agent_id in requests:
                evaluator.evaluate(game_state, agent_id)
            return
        self.graphs += GNNEvaluator.evaluate_many(requests, self.policy, self.device)
        self.forwards += 1

    def stats(self) -> Dict[str, float]:
        return {'rounds': self.rounds, 'forwards': self.forwards, 'graphs': self.graphs,
                'graphs_per_forward': self.graphs / self.forwards if self.forwards else 0.0}


# ---------------------------------------------------------------------------
# Self-play games (ExIt and AZ)
# ---------------------------------------------------------------------------

def build_agent(policy, decoder, map_config, mcts_config: Dict[str, Any],
                max_turns: int, max_regions: int) -> MCTSGNNAgent:
    """An `MCTSGNNAgent` on the CPU with the self-play workers' settings."""
    return MCTSGNNAgent(
        policy=policy,
        decoder=decoder,
        map_config=map_config,
        simulation_budget=int(mcts_config.get('simulation_budget', 50)),
        c_puct=float(mcts_config.get('c_puct', 1.4)),
        action_budget=decoder.action_budget,
        max_troops=20,
        max_turns=max_turns,
        uct_c=float(mcts_config.get('uct_c', 1.41)),
        pw_alpha=float(mcts_config.get('pw_alpha', 0.5)),
        max_rollout_turns=int(mcts_config.get('max_rollout_turns', 20)),
        device='cpu',
        max_regions=max_regions,
        pw_sampler=mcts_config.get('pw_sampler', 'masked_random'),
        use_value_fn=bool(mcts_config.get('use_value_fn', True)),
    )


def self_play_game_specs(map_names: Sequence[str], num_games: int, base_seed: int,
                         map_seed_offset: int) -> List[Tuple[str, int]]:
    """(map_name, seed) per game: the workers' map sequence, seeds base_seed + g.

    The map of game g is drawn from `RandomState(base_seed + map_seed_offset)`
    exactly as the one-game-at-a-time workers draw it (offset 1 for ExIt,
    0 for AZ), so a lockstep worker plays the same map mix.
    """
    map_rng = np.random.RandomState(base_seed + map_seed_offset)
    return [(map_names[map_rng.randint(len(map_names))], base_seed + g)
            for g in range(num_games)]


def self_play_games_lockstep(policy, env_configs, mcts_config: Dict[str, Any],
                             self_play_config: Dict[str, Any], num_games: int,
                             base_seed: int, max_regions: int, episode_gen,
                             games_per_batch: int = 16, device='cpu',
                             map_seed_offset: int = 1,
                             single_graph_forwards: bool = False,
                             stats: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """Play `num_games` self-play games in lockstep; returns their examples, game by game.

    policy: CPU GCNPolicy (the agents'); the batched forwards use a copy on `device`.
    episode_gen: `play_episode_exit_gen` (ExIt) or `play_episode_gen` (AZ).
    Game g plays the map `self_play_game_specs` assigns it with seed
    base_seed + g: `env.reset(seed)` for the env and sampler streams and
    `np.random.default_rng(seed)` for its Dirichlet noise.
    stats: if given, receives the batch counters (see `GameBatch.stats`).
    """
    action_budget = int(mcts_config.get('action_budget', 5))
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)
    max_turns_by_map = {e['map_name']: int(e.get('max_turns', 50)) for e in env_configs}
    map_names = list(max_turns_by_map)
    # One MapConfig per map, shared by every agent playing it: evaluate_many
    # deduplicates identical positions by map identity.
    map_configs = {m: MapRegistry.get(m) for m in map_names}
    specs = self_play_game_specs(map_names, num_games, base_seed, map_seed_offset)

    def make_game(map_name, seed):
        max_turns = max_turns_by_map[map_name]
        env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns, seed=None,
                              reward_shaping_config=None)
        agent = build_agent(policy, decoder, map_configs[map_name], mcts_config,
                            max_turns, max_regions)
        gen = episode_gen(agent, env, self_play_config, map_name, seed=seed,
                          rng=np.random.default_rng(seed))
        return LockstepGame(gen, agent.evaluator, {'map_name': map_name, 'seed': seed})

    batch = GameBatch(forward_policy(policy, device), device, games_per_batch,
                      single_graph_forwards)
    games = batch.play(make_game(m, s) for m, s in specs)
    if stats is not None:
        stats.update(batch.stats())
    examples: List[Dict[str, Any]] = []
    for game in games:
        examples.extend(game.result)
    return examples


def self_play_lockstep_worker(args):
    """Spawn-pool worker: `num_games` self-play games in lockstep, examples back.

    args: (policy_state_dict, model_kwargs, env_configs, mcts_config,
           self_play_config, num_games, base_seed, max_regions,
           games_per_batch, device, kind) with kind 'exit' or 'az'.
    """
    (policy_state_dict, model_kwargs, env_configs, mcts_config, self_play_config,
     num_games, base_seed, max_regions, games_per_batch, device, kind) = args

    torch.set_num_threads(1)  # avoid oversubscription with sibling workers
    torch.manual_seed(base_seed)

    policy = GCNPolicy(**model_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    if kind == 'exit':
        from parallel_risk.training.mcts_gnn.exit_self_play import play_episode_exit_gen as episode_gen
        map_seed_offset = 1
    elif kind == 'az':
        from parallel_risk.training.mcts_gnn.self_play import play_episode_gen as episode_gen
        map_seed_offset = 0
    else:
        raise ValueError(f"kind must be 'exit' or 'az', got {kind!r}")

    return self_play_games_lockstep(
        policy, env_configs, mcts_config, self_play_config, num_games, base_seed,
        max_regions, episode_gen, games_per_batch=games_per_batch, device=device,
        map_seed_offset=map_seed_offset)


# ---------------------------------------------------------------------------
# Evaluation games: MCTS+GNN vs MCTS-uniform
# ---------------------------------------------------------------------------

def eval_episode_gen(gnn_agent: MCTSGNNAgent, uniform_agent: MCTSAgent, env,
                     gnn_agent_id: str, seed: int):
    """One evaluation game as a generator: MCTS+GNN as `gnn_agent_id`, MCTS-uniform as the other.

    Same turn structure as the experiment scripts' eval worker: agents act
    in id order, so with the GNN as agent_1 the uniform search runs first
    and both draw from the game's `np.random` stream in that order. Yields
    the GNN search's evaluation requests; the uniform search is synchronous.
    Returns {'gnn_agent_id', 'result' ('win'|'loss'|'draw'), 'length', 'rewards'}.
    """
    obs, _ = env.reset(seed=seed)
    mcts = gnn_agent.mcts
    rewards = {a: 0.0 for a in env.possible_agents}
    done = False
    turns = 0
    while not done:
        actions = {}
        for agent_id in AGENT_IDS:
            if agent_id not in obs:
                continue
            if agent_id == gnn_agent_id:
                root = yield from mcts.make_root_gen(env.game_state)
                yield from mcts.search_gen(root, gnn_agent.simulation_budget)
                actions[agent_id] = mcts.best_action(root, agent_id)
            else:
                actions[agent_id] = uniform_agent.get_action(env.game_state, agent_id)
        obs, rewards, terms, truncs, _ = env.step(actions)
        done = terms.get('__all__', False) or truncs.get('__all__', False)
        turns += 1

    opponent_id = 'agent_1' if gnn_agent_id == 'agent_0' else 'agent_0'
    r_gnn = float(rewards.get(gnn_agent_id, 0.0))
    r_opp = float(rewards.get(opponent_id, 0.0))
    result = 'win' if r_gnn > r_opp else ('loss' if r_gnn < r_opp else 'draw')
    return {'gnn_agent_id': gnn_agent_id, 'result': result, 'length': turns,
            'rewards': {a: float(rewards.get(a, 0.0)) for a in AGENT_IDS}}


def eval_specs(map_names: Sequence[str], num_games_per_map: int,
               base_seed: int) -> List[Tuple[str, int, str]]:
    """(map_name, seed, gnn_agent_id) per eval game, colours alternating.

    Map i, game g gets seed base_seed + i * 1000 + g and the GNN plays
    agent_0 for even g: the seeding of the experiment scripts' per-map eval
    worker, so a lockstep eval plays the same games.
    """
    return [(m, base_seed + i * 1000 + g, 'agent_0' if g % 2 == 0 else 'agent_1')
            for i, m in enumerate(map_names) for g in range(num_games_per_map)]


def play_eval_games_lockstep(policy, specs: Sequence[Tuple[str, int, str]], max_turns: int,
                             action_budget: int, mcts_budget: int, max_regions: int,
                             games_per_batch: int = 16, device='cpu',
                             single_graph_forwards: bool = False,
                             stats: Dict[str, float] = None,
                             pw_sampler: str = 'masked_random') -> List[Dict[str, Any]]:
    """Play MCTS+GNN vs MCTS-uniform games in lockstep; one outcome dict per spec, in order.

    policy: CPU GCNPolicy; the batched forwards use a copy on `device`.
    specs: (map_name, seed, gnn_agent_id) per game, e.g. from `eval_specs`.
    Agents are built as the experiment scripts' eval worker builds them
    (c_puct 1.4, value head at leaves, uniform MCTS at the same budget).
    Each outcome carries map_name, seed, gnn_agent_id, result, length, rewards;
    `summarize_eval_outcomes` reduces them to per-map win/draw/score rates.
    """
    decoder = ActionDecoder(action_budget=action_budget, max_troops=20)
    map_configs: Dict[str, Any] = {}

    def make_game(map_name, seed, gnn_agent_id):
        env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns,
                              reward_shaping_config=None)
        map_config = map_configs.setdefault(map_name, env.map_config)
        gnn_agent = MCTSGNNAgent(
            policy=policy, decoder=decoder, map_config=map_config,
            simulation_budget=mcts_budget, c_puct=1.4, action_budget=action_budget,
            max_turns=max_turns, device='cpu', max_regions=max_regions,
            pw_sampler=pw_sampler)
        uniform_agent = MCTSAgent.from_env(env, simulation_budget=mcts_budget,
                                           action_budget=action_budget)
        gen = eval_episode_gen(gnn_agent, uniform_agent, env, gnn_agent_id, seed)
        return LockstepGame(gen, gnn_agent.evaluator,
                            {'map_name': map_name, 'seed': seed, 'gnn_agent_id': gnn_agent_id})

    batch = GameBatch(forward_policy(policy, device), device, games_per_batch,
                      single_graph_forwards)
    games = batch.play(make_game(*spec) for spec in specs)
    if stats is not None:
        stats.update(batch.stats())
    outcomes = []
    for game in games:
        outcome = dict(game.info)
        outcome.update(game.result)
        outcomes.append(outcome)
    return outcomes


def summarize_eval_outcomes(outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-map {wins, losses, draws, total, win_rate, draw_rate, score} from eval outcomes.

    Same schema as the experiment scripts' per-map eval results (score
    counts a draw as half a win).
    """
    per_map: Dict[str, Dict[str, float]] = {}
    for outcome in outcomes:
        counts = per_map.setdefault(outcome['map_name'], {'wins': 0, 'losses': 0, 'draws': 0})
        counts[{'win': 'wins', 'loss': 'losses', 'draw': 'draws'}[outcome['result']]] += 1
    for counts in per_map.values():
        total = counts['wins'] + counts['losses'] + counts['draws']
        denom = max(total, 1)
        counts.update(total=total, win_rate=counts['wins'] / denom,
                      draw_rate=counts['draws'] / denom,
                      score=(counts['wins'] + 0.5 * counts['draws']) / denom)
    return per_map


def eval_lockstep_worker(args):
    """Spawn-pool worker for eval games: rebuilds the policy on the CPU and plays `specs`.

    args: (policy_state_dict, model_kwargs, specs, max_turns, action_budget,
           mcts_budget, max_regions, games_per_batch, device[, pw_sampler]).
    Returns the outcome dicts of `play_eval_games_lockstep`.
    """
    pw_sampler = 'masked_random'
    if len(args) == 10:
        (policy_state_dict, model_kwargs, specs, max_turns, action_budget,
         mcts_budget, max_regions, games_per_batch, device, pw_sampler) = args
    else:
        (policy_state_dict, model_kwargs, specs, max_turns, action_budget,
         mcts_budget, max_regions, games_per_batch, device) = args

    torch.set_num_threads(1)
    torch.manual_seed(int(specs[0][1]) if specs else 0)

    policy = GCNPolicy(**model_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    return play_eval_games_lockstep(
        policy, specs, max_turns=max_turns, action_budget=action_budget,
        mcts_budget=mcts_budget, max_regions=max_regions,
        games_per_batch=games_per_batch, device=device, pw_sampler=pw_sampler)
