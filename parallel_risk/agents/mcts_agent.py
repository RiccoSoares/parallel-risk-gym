"""Monte Carlo Tree Search agent for Parallel Risk using Decoupled UCT.

Handles simultaneous-move gameplay where both players act each turn.
Reference: Lanctot et al. (2013) "Monte Carlo Tree Search for Simultaneous Move Games"
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from parallel_risk.env.combat import CombatResolver
from parallel_risk.env.map_config import MapConfig
from parallel_risk.env.validators import ActionValidator


# ---------------------------------------------------------------------------
# Action key helpers
# ---------------------------------------------------------------------------

def _action_to_key(action_dict: dict) -> tuple:
    """Convert action dict to a hashable tuple of (src, dst, troops) rows."""
    n = int(action_dict['num_actions'])
    arr = action_dict['actions']
    return tuple((int(arr[i, 0]), int(arr[i, 1]), int(arr[i, 2])) for i in range(n))


def _key_to_action(key: tuple, max_actions: int = 10) -> dict:
    """Reconstruct action dict from a hashable key.

    The output array is sized `max(len(key), max_actions)` so keys longer
    than the classic default of 10 (i.e. action_budget > 10) don't overflow.
    Env consumers that expect a fixed shape can re-pad downstream.
    """
    size = max(len(key), max_actions)
    arr = np.zeros((size, 3), dtype=np.int32)
    for i, (src, dst, troops) in enumerate(key):
        arr[i] = [src, dst, troops]
    return {'num_actions': len(key), 'actions': arr}


# ---------------------------------------------------------------------------
# RiskSimulator
# ---------------------------------------------------------------------------

class RiskSimulator:
    """Lightweight deterministic game simulator for MCTS rollouts.

    Replicates ParallelRiskEnv step logic with two differences:
    1. Actions are sorted by source territory instead of randomly shuffled.
    2. Operates directly on plain state dicts (no PettingZoo overhead).
    """

    AGENTS = ["agent_0", "agent_1"]

    def __init__(self, map_config: MapConfig, max_turns: int = 100):
        self.map_config = map_config
        self.max_turns = max_turns

    @staticmethod
    def clone_state(game_state: dict) -> dict:
        return {
            'territory_ownership': game_state['territory_ownership'].copy(),
            'territory_troops':    game_state['territory_troops'].copy(),
            'turn_number':         game_state['turn_number'],
            'income_per_turn':     game_state['income_per_turn'],
            'available_income':    game_state['available_income'].copy(),
        }

    def state_to_obs(self, game_state: dict, agent: str) -> dict:
        """Convert game_state to agent-relative flat obs dict.

        Output is compatible with MaskedRandomAgentRLlib.get_action_raw().
        Mirrors ParallelRiskEnv._get_observation() exactly.
        """
        agent_idx = self.AGENTS.index(agent)
        n = self.map_config.n_territories

        # Agent-relative ownership: +1=self, -1=enemy
        ownership = np.where(
            game_state['territory_ownership'] == agent_idx,
            np.int8(1),
            np.int8(-1),
        ).astype(np.int8)

        # Region control
        region_names = list(self.map_config.regions.keys())
        region_control = np.zeros(len(region_names), dtype=np.int8)
        for i, region_name in enumerate(region_names):
            territories = self.map_config.regions[region_name]
            if all(game_state['territory_ownership'][t] == agent_idx for t in territories):
                region_control[i] = 1

        return {
            'territory_ownership': ownership,
            'territory_troops': game_state['territory_troops'].copy(),
            'adjacency_matrix': self.map_config.adjacency_matrix.copy(),
            'available_income': np.array([game_state['available_income'][agent]], dtype=np.int32),
            'turn_number': np.array([game_state['turn_number']], dtype=np.int32),
            'region_control': region_control,
        }

    def _calculate_income(self, game_state: dict, agent: str) -> int:
        """Base income + region bonuses. Mirrors env._calculate_income()."""
        agent_idx = self.AGENTS.index(agent)
        income = game_state['income_per_turn']
        for region_name, territories in self.map_config.regions.items():
            if all(game_state['territory_ownership'][t] == agent_idx for t in territories):
                income += self.map_config.region_bonuses[region_name]
        return income

    def step(self, game_state: dict, actions: dict) -> tuple:
        """Execute one turn deterministically. Does NOT mutate input.

        Args:
            game_state: Current game state dict.
            actions: {'agent_0': action_dict, 'agent_1': action_dict}

        Returns:
            (next_state, rewards_dict, done_bool)
        """
        state = self.clone_state(game_state)

        # Recalculate income at turn start (mirrors env.step() lines 257-258)
        for agent in self.AGENTS:
            state['available_income'][agent] = self._calculate_income(state, agent)

        # Collect and classify actions
        validator = ActionValidator(state, self.map_config, self.AGENTS)
        all_actions = []
        for agent in self.AGENTS:
            if agent not in actions:
                continue
            action_dict = actions[agent]
            num_actions = int(action_dict['num_actions'])
            for i in range(num_actions):
                source, dest, troops = action_dict['actions'][i]
                source, dest, troops = int(source), int(dest), int(troops)
                action_type = validator.classify_action(source, dest)
                all_actions.append({
                    'agent': agent,
                    'source': source,
                    'dest': dest,
                    'troops': troops,
                    'type': action_type,
                })

        # Sort by source for determinism (replaces random.shuffle)
        all_actions.sort(key=lambda a: a['source'])

        # Execute validated actions (validator reads live mutating state — intentional,
        # matches env behavior: later actions see updated ownership from earlier ones)
        for action_info in all_actions:
            if validator.validate_action(action_info):
                self._execute_action(state, action_info)

        state['turn_number'] += 1
        rewards, done = self._check_terminal(state)
        return state, rewards, done

    def _execute_action(self, state: dict, action_info: dict) -> None:
        """Execute a single validated action in place. Mirrors env._execute_action()."""
        agent = action_info['agent']
        agent_idx = self.AGENTS.index(agent)
        source = action_info['source']
        dest = action_info['dest']
        troops = action_info['troops']
        action_type = action_info['type']

        if action_type == 'deploy':
            state['territory_troops'][dest] += troops
            state['available_income'][agent] -= troops

        elif action_type == 'transfer':
            state['territory_troops'][source] -= troops
            state['territory_troops'][dest] += troops

        elif action_type == 'attack':
            state['territory_troops'][source] -= troops
            defending_troops = state['territory_troops'][dest]
            result, surviving_troops = CombatResolver.resolve(troops, defending_troops)

            if result == 'attacker_wins':
                state['territory_ownership'][dest] = agent_idx
                state['territory_troops'][dest] = surviving_troops
            else:
                attacker_casualties = int(defending_troops * 0.6)
                attackers_surviving = max(0, troops - attacker_casualties)
                state['territory_troops'][source] += attackers_surviving
                state['territory_troops'][dest] = surviving_troops

    def _check_terminal(self, state: dict) -> tuple:
        """Returns (rewards_dict, done_bool). Mirrors env._check_termination()."""
        no_rewards = {a: 0.0 for a in self.AGENTS}

        territory_counts = {
            a: int(np.sum(state['territory_ownership'] == i))
            for i, a in enumerate(self.AGENTS)
        }

        # Victory: one agent owns all territories
        for agent, count in territory_counts.items():
            if count == self.map_config.n_territories:
                rewards = {a: (1.0 if a == agent else -1.0) for a in self.AGENTS}
                return rewards, True

        # Elimination: one agent has 0 territories
        eliminated = [a for a, c in territory_counts.items() if c == 0]
        if eliminated:
            remaining = [a for a in self.AGENTS if a not in eliminated]
            if len(remaining) == 1:
                winner = remaining[0]
                rewards = {a: (1.0 if a == winner else -1.0) for a in self.AGENTS}
                return rewards, True

        # Turn limit
        if state['turn_number'] >= self.max_turns:
            return no_rewards, True

        return no_rewards, False


# ---------------------------------------------------------------------------
# DUCT tree node
# ---------------------------------------------------------------------------

@dataclass
class DuctNode:
    """Node in the Decoupled UCT tree.

    Each player has independent per-action statistics:
        stats['agent_0'][action_key] = {'q': float, 'n': int}

    Joint action (a0_key, a1_key) maps to exactly one child node.
    available_actions is grown by progressive widening.
    """

    game_state: dict
    parent: Optional['DuctNode']
    parent_joint_key: Optional[tuple]
    is_terminal: bool
    terminal_rewards: Optional[dict]

    available_actions: dict = field(
        default_factory=lambda: {'agent_0': [], 'agent_1': []}
    )
    stats: dict = field(
        default_factory=lambda: {'agent_0': {}, 'agent_1': {}}
    )
    children: dict = field(default_factory=dict)  # (a0_key, a1_key) -> DuctNode
    visit_count: int = 0


# ---------------------------------------------------------------------------
# Decoupled UCT
# ---------------------------------------------------------------------------

class DuctMCTS:
    """Decoupled UCT for simultaneous-move games.

    Reference: Lanctot et al. (2013) "Monte Carlo Tree Search for
    Simultaneous Move Games: A Case Study in the Spatial Game Blokus Duo"
    """

    def __init__(
        self,
        simulator: RiskSimulator,
        action_sampler_0,
        action_sampler_1,
        uct_c: float = 1.41,
        pw_alpha: float = 0.5,
        max_rollout_turns: int = 50,
        value_fn=None,
        policy_fn=None,
        c_puct: float = 1.4,
    ):
        self.sim = simulator
        self.samplers = {'agent_0': action_sampler_0, 'agent_1': action_sampler_1}
        self.uct_c = uct_c
        self.pw_alpha = pw_alpha
        self.max_rollout_turns = max_rollout_turns
        # Optional neural value function for AlphaZero-style leaf evaluation.
        # Signature: value_fn(game_state: dict, agent_id: str) -> float
        # When None, falls back to random rollouts (default MCTS behaviour).
        self.value_fn = value_fn
        # Optional neural policy function for PUCT-guided selection.
        # Signature: policy_fn(game_state: dict, agent_id: str, action_dict: dict) -> float
        # Returns the summed log-prob of the given action under the policy.
        # When None, selection uses vanilla UCT (unchanged behaviour).
        self.policy_fn = policy_fn
        self.c_puct = c_puct

    def _add_sampled_action(self, node: DuctNode, agent_id: str) -> bool:
        """Sample one action for `agent_id` at `node` and register it.

        Uses the agent's sampler to draw a candidate action, adds it to the
        node's available_actions if new, and pre-seeds the stats dict with
        the PUCT prior (when policy_fn is set) so `_puct_select` can read it
        without an extra forward pass. When policy_fn is None, prior is 0.0
        (exp(0)=1, harmless for the vanilla UCT branch which ignores 'p').

        Returns True if a new action was added, False if the sampler
        returned a duplicate. Callers rely on this signal for exhaustion
        detection (progressive widening + `apply_root_dirichlet`).
        """
        obs = self.sim.state_to_obs(node.game_state, agent_id)
        action_dict = self.samplers[agent_id].get_action_raw(obs)
        key = _action_to_key(action_dict)
        if key in node.available_actions[agent_id]:
            return False
        node.available_actions[agent_id].append(key)
        if key not in node.stats[agent_id]:
            prior = 0.0
            if self.policy_fn is not None:
                prior = self.policy_fn(node.game_state, agent_id, action_dict)
            node.stats[agent_id][key] = {'q': 0.0, 'n': 0, 'p': prior}
        return True

    def make_root(self, game_state: dict) -> DuctNode:
        """Create root node from current game state."""
        state = RiskSimulator.clone_state(game_state)
        _, done = self.sim._check_terminal(state)
        node = DuctNode(
            game_state=state,
            parent=None,
            parent_joint_key=None,
            is_terminal=done,
            terminal_rewards=None,
        )
        if done:
            rewards, _ = self.sim._check_terminal(state)
            node.terminal_rewards = rewards
        else:
            # Seed each player's action list with one sampled action (+ prior)
            for agent in self.sim.AGENTS:
                self._add_sampled_action(node, agent)
        return node

    def run(self, root: DuctNode, budget: int) -> None:
        """Execute `budget` MCTS iterations from root."""
        for _ in range(budget):
            node, path = self._select(root)
            if node.is_terminal:
                rewards = node.terminal_rewards
            else:
                rewards = self._rollout(node.game_state)
            self._backprop(path, node, rewards)

    def best_action(self, root: DuctNode, agent_id: str) -> dict:
        """Return most-visited action for agent_id at root."""
        best_key = None
        best_n = -1
        for key in root.available_actions[agent_id]:
            n = root.stats[agent_id].get(key, {}).get('n', 0)
            if n > best_n:
                best_n = n
                best_key = key
        if best_key is None:
            # No stats yet (budget=0): fall back to first available action
            best_key = root.available_actions[agent_id][0]
        return _key_to_action(best_key)

    def policy_target(self, root: DuctNode, agent_id: str) -> dict:
        """Normalized visit distribution over the root's available actions.

        Returns {action_key: probability}. If every action has zero visits
        (e.g. budget=0), returns a uniform distribution over available actions.
        Distillation targets for MCTS+GNN training read from this.
        """
        counts = {
            key: float(root.stats[agent_id].get(key, {}).get('n', 0))
            for key in root.available_actions[agent_id]
        }
        total = sum(counts.values())
        if total <= 0:
            k = len(counts)
            if k == 0:
                return {}
            return {key: 1.0 / k for key in counts}
        return {key: n / total for key, n in counts.items()}

    def apply_root_dirichlet(self, root: DuctNode, alpha: float = 0.3,
                             noise_frac: float = 0.25,
                             min_actions: int = 8, rng=None) -> None:
        """Mix Dirichlet noise into stored per-action priors at the root.

        AlphaZero-style: for each agent, ensure at least `min_actions`
        candidates are known (calling the sampler if the seeded set is
        smaller), then draw one `Dirichlet(alpha, k)` and mix in
        probability space as
            p_mixed = (1 - frac) * softmax(raw_priors) + frac * noise
        storing `log(p_mixed)` back into `stats[agent][key]['p']`.

        Softmax-normalizing the raw priors before mixing matters — otherwise
        `exp(raw_log_prob)` on our huge joint action space is ~1e-8 and the
        noise term dominates every key uniformly, defeating exploration.

        `min_actions` guards against the degenerate case where `make_root`
        seeded only 1 action per agent: `Dirichlet([alpha])` with k=1 always
        samples [1.0] and injects zero noise.

        No-op when `noise_frac <= 0` or when the sampler yields no candidates.
        Accepts a numpy Generator `rng` for reproducibility; falls back to a
        fresh default_rng() if None.
        """
        if noise_frac <= 0.0:
            return
        if rng is None:
            rng = np.random.default_rng()
        for agent in self.sim.AGENTS:
            # Widen the action set to at least min_actions (stop early if
            # the sampler is stuck returning duplicates).
            consec_dupes = 0
            while len(root.available_actions[agent]) < min_actions:
                if root.is_terminal:
                    break
                added = self._add_sampled_action(root, agent)
                if not added:
                    consec_dupes += 1
                    if consec_dupes >= 3:
                        break
                else:
                    consec_dupes = 0

            keys = list(root.available_actions[agent])
            k = len(keys)
            if k == 0:
                continue

            # Softmax over stored raw log-priors so mixing happens in
            # probability space with comparable magnitudes.
            raw = [float(root.stats[agent].get(key, {}).get('p', 0.0)) for key in keys]
            max_p = max(raw)
            exps = [math.exp(p - max_p) for p in raw]
            z = sum(exps) or 1.0
            normalized = [e / z for e in exps]

            noise = rng.dirichlet([alpha] * k)
            for i, key in enumerate(keys):
                s = root.stats[agent].get(key)
                if s is None:
                    root.stats[agent][key] = {'q': 0.0, 'n': 0, 'p': 0.0}
                    s = root.stats[agent][key]
                p_new = (1.0 - noise_frac) * normalized[i] + noise_frac * float(noise[i])
                s['p'] = math.log(max(p_new, 1e-12))

    def sampled_action(self, root: DuctNode, agent_id: str,
                       temperature: float = 1.0) -> dict:
        """Return an action sampled from `n^(1/T) / sum n^(1/T)`.

        T <= 1e-3 falls back to `best_action` (argmax over visits) to avoid
        numerical blowups. Standard AlphaZero-style: T=1 for early moves
        (exploration), T~0 later (exploitation).
        """
        if temperature <= 1e-3:
            return self.best_action(root, agent_id)
        keys = list(root.available_actions[agent_id])
        if not keys:
            raise RuntimeError(f"No available actions at root for {agent_id}")
        counts = np.array([
            float(root.stats[agent_id].get(k, {}).get('n', 0))
            for k in keys
        ])
        if counts.sum() <= 0:
            # No visits yet: uniform sample.
            probs = np.ones(len(keys)) / len(keys)
        else:
            scaled = counts ** (1.0 / temperature)
            probs = scaled / scaled.sum()
        idx = int(np.random.choice(len(keys), p=probs))
        return _key_to_action(keys[idx])

    def _select(self, root: DuctNode) -> tuple:
        """Traverse tree using UCT. Returns (leaf_node, path).

        path entries are (node, a0_key, a1_key).
        """
        node = root
        path = []

        while not node.is_terminal and node.visit_count > 0:
            # Progressive widening: grow each player's action set if needed.
            # Guard against sampler exhaustion by counting consecutive duplicates.
            for agent in self.sim.AGENTS:
                consec_dupes = 0
                attempts = 0
                while node.visit_count ** self.pw_alpha > len(node.available_actions[agent]):
                    added = self._add_sampled_action(node, agent)
                    if added:
                        consec_dupes = 0
                    else:
                        consec_dupes += 1
                        if consec_dupes >= 3:
                            break  # sampler stuck: give up widening at this node
                    attempts += 1
                    if attempts > 20:
                        break

            a0 = self._select_action(node, 'agent_0')
            a1 = self._select_action(node, 'agent_1')
            joint_key = (a0, a1)
            path.append((node, a0, a1))

            if joint_key not in node.children:
                # Expansion: simulate the joint action and create a child
                actions = {
                    'agent_0': _key_to_action(a0),
                    'agent_1': _key_to_action(a1),
                }
                next_state, rewards, done = self.sim.step(node.game_state, actions)
                child = DuctNode(
                    game_state=next_state,
                    parent=node,
                    parent_joint_key=joint_key,
                    is_terminal=done,
                    terminal_rewards=rewards if done else None,
                )
                if not done:
                    for agent in self.sim.AGENTS:
                        self._add_sampled_action(child, agent)
                node.children[joint_key] = child
                return child, path

            node = node.children[joint_key]

        return node, path

    def _select_action(self, node: DuctNode, agent_id: str) -> tuple:
        """Dispatch to PUCT when a policy_fn is set, otherwise vanilla UCT."""
        if self.policy_fn is None:
            return self._uct_select(node, agent_id)
        return self._puct_select(node, agent_id)

    def _uct_select(self, node: DuctNode, agent_id: str) -> tuple:
        """Select action for one player using UCT formula."""
        N = node.visit_count
        best_key = None
        best_score = -math.inf

        for key in node.available_actions[agent_id]:
            s = node.stats[agent_id].get(key)
            if s is None or s['n'] == 0:
                return key  # unvisited action gets priority
            score = s['q'] + self.uct_c * math.sqrt(math.log(N) / s['n'])
            if score > best_score:
                best_score = score
                best_key = key

        return best_key

    def _puct_select(self, node: DuctNode, agent_id: str) -> tuple:
        """PUCT selection: Q + c_puct * P * sqrt(N_parent) / (1 + n).

        Priors P are softmax-normalized across the currently-known action
        set at this node so they sum to 1 (standard AlphaZero behavior).
        Without this normalization, `exp(raw_log_prob)` on a large joint
        action space (~e^-18 for ours) makes the prior term ~1e-8 —
        eight orders of magnitude smaller than Q — and PUCT collapses to
        pure Q-argmax. The stored `s['p']` is still the raw log-prior
        from `policy_fn`; softmax happens here at read time.

        Unvisited actions are returned immediately (same fallback as
        `_uct_select`) so priors don't starve exploration.
        """
        keys = node.available_actions[agent_id]
        if not keys:
            raise RuntimeError(f"_puct_select: no available actions for {agent_id}")

        raw = [float(node.stats[agent_id].get(k, {}).get('p', 0.0)) for k in keys]
        max_p = max(raw)
        exps = [math.exp(p - max_p) for p in raw]  # log-sum-exp for stability
        z = sum(exps) or 1.0
        priors = [e / z for e in exps]

        sqrt_N = math.sqrt(max(node.visit_count, 1))
        best_key = None
        best_score = -math.inf

        for key, p in zip(keys, priors):
            s = node.stats[agent_id].get(key)
            if s is None or s['n'] == 0:
                return key
            score = s['q'] + self.c_puct * p * sqrt_N / (1 + s['n'])
            if score > best_score:
                best_score = score
                best_key = key

        return best_key

    def _rollout(self, game_state: dict) -> dict:
        """Evaluate a leaf node.

        If a value_fn was provided (AlphaZero mode), calls it for a direct
        neural estimate.  Otherwise runs a random playout (standard MCTS).
        """
        if self.value_fn is not None:
            v = self.value_fn(game_state, 'agent_0')
            return {'agent_0': v, 'agent_1': -v}

        state = RiskSimulator.clone_state(game_state)
        for _ in range(self.max_rollout_turns):
            rewards, done = self.sim._check_terminal(state)
            if done:
                return rewards
            actions = {}
            for agent in self.sim.AGENTS:
                obs = self.sim.state_to_obs(state, agent)
                actions[agent] = self.samplers[agent].get_action_raw(obs)
            state, rewards, done = self.sim.step(state, actions)
            if done:
                return rewards
        return {'agent_0': 0.0, 'agent_1': 0.0}

    def _backprop(self, path: list, leaf: DuctNode, rewards: dict) -> None:
        """Update visit counts and Q-values along the path."""
        leaf.visit_count += 1
        for node, a0, a1 in reversed(path):
            node.visit_count += 1
            for agent, key in [('agent_0', a0), ('agent_1', a1)]:
                r = rewards[agent]
                if key not in node.stats[agent]:
                    node.stats[agent][key] = {'q': 0.0, 'n': 0, 'p': 0.0}
                s = node.stats[agent][key]
                s['n'] += 1
                s['q'] += (r - s['q']) / s['n']  # incremental mean


# ---------------------------------------------------------------------------
# MCTSAgent
# ---------------------------------------------------------------------------

class MCTSAgent:
    """Agent using Decoupled UCT for simultaneous-move game play.

    Usage:
        agent = MCTSAgent.from_env(env, simulation_budget=200)
        action = agent.get_action(env.game_state, 'agent_0')

    Note: get_action takes full game_state, not an observation dict.
    """

    def __init__(
        self,
        map_config: MapConfig,
        simulation_budget: int = 200,
        max_rollout_turns: int = 50,
        uct_c: float = 1.41,
        pw_alpha: float = 0.5,
        action_budget: int = 5,
        max_troops: int = 20,
        max_turns: int = 100,
        value_fn=None,
    ):
        from parallel_risk.agents.masked_random_agent import MaskedRandomAgentRLlib

        self.simulation_budget = simulation_budget
        simulator = RiskSimulator(map_config, max_turns=max_turns)
        sampler_0 = MaskedRandomAgentRLlib(
            n_territories=map_config.n_territories,
            adjacency_matrix=map_config.adjacency_matrix,
            action_budget=action_budget,
            max_troops=max_troops,
        )
        sampler_1 = MaskedRandomAgentRLlib(
            n_territories=map_config.n_territories,
            adjacency_matrix=map_config.adjacency_matrix,
            action_budget=action_budget,
            max_troops=max_troops,
        )
        self.mcts = DuctMCTS(
            simulator=simulator,
            action_sampler_0=sampler_0,
            action_sampler_1=sampler_1,
            uct_c=uct_c,
            pw_alpha=pw_alpha,
            max_rollout_turns=max_rollout_turns,
            value_fn=value_fn,
        )

    @classmethod
    def from_env(cls, env, simulation_budget: int = 200, value_fn=None, **kwargs) -> 'MCTSAgent':
        """Create MCTSAgent from a ParallelRiskEnv instance."""
        if hasattr(env, 'env'):
            base_env = env.env
        else:
            base_env = env
        return cls(
            map_config=base_env.map_config,
            simulation_budget=simulation_budget,
            max_turns=base_env.max_turns,
            value_fn=value_fn,
            **kwargs,
        )

    def get_action(self, game_state: dict, agent_id: str) -> dict:
        """Run MCTS and return the best action for agent_id.

        Args:
            game_state: Full game state dict from env.game_state.
            agent_id: 'agent_0' or 'agent_1'.

        Returns:
            Action dict {'num_actions': int, 'actions': np.ndarray((10,3))}.
        """
        root = self.mcts.make_root(game_state)
        self.mcts.run(root, self.simulation_budget)
        return self.mcts.best_action(root, agent_id)
