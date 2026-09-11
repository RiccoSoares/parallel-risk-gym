"""
Lockstep vectorized rollout engine for PPO on Parallel Risk.

`LockstepRollout` steps `num_envs` environments in lockstep: at every step the
observations of all live agents of all envs (2 per env) are collated into one
PyG Batch, forwarded ONCE through the policy, decoded ONCE with the
pre-batched geometry path of ActionDecoder, and the sampled actions are handed
to each env. The old collector forwarded a batch of 2 graphs per env step
plus a second batch of 2 for the bootstrap value; here the bootstrap value of
a non-terminal transition is the value the next step computes for the same
env, so a rollout costs one forward per lockstep step (plus one small forward
per step that truncates an episode, and one at the end).

Both PPOTrainer rollout paths use this engine: the in-process path (policy on
the trainer device, CPU or CUDA) and the pool workers (CPU, one engine per
worker process). The engine produces compact tensors (`collect`), which are
cheap to pickle across processes; `build_rollout` turns them back into the
rollout dict PPOTrainer.update_policy consumes.

Sample layout
-------------
A collection is a [T, B] grid with B = 2 * num_envs columns: column 2*i is
env i's agent_0 and column 2*i+1 its agent_1. Every column is an independent
trajectory; GAE (PPOTrainer.compute_gae) runs per column, with `dones`
marking episode boundaries and `next_values` holding the bootstrap value of
each transition:

  * non-terminal:  V(s_{t+1}) of the same column, taken from the next step's
                   forward (last step: one extra forward of the live obs);
  * truncated:     V(final observation), one extra batched forward over the
                   envs that hit the turn limit at that step;
  * terminated:    0 (conquest / elimination).

Randomness
----------
Map choice at every reset: `rng.randint(len(map_names))` (uniform), where
`rng` is `np.random` in the trainer process (as the old sequential path) or a
`np.random.RandomState(worker_seed)` in a worker. Env resets in workers are
seeded `reset_seed_base + k` for the k-th reset of the collection (unseeded
in-process, as before). Actions are sampled from the torch default generator;
one decode call per step covers all envs, so the order in which random numbers
are consumed differs from the one-env-at-a-time collector (declared in
docs/PERF_OPTIMIZATION.md).

Dropout
-------
The policy is used in whatever mode the caller left it: the trainer process
never switches its policy to eval(), so in-process rollouts sample with
dropout active exactly as the old sequential path did; workers call
policy.eval() as before. In eval mode the reused bootstrap value is exactly
the value the old code recomputed (same graph, deterministic forward; the
only difference is float summation order inside the larger batch). In train
mode it is the same dropout sample that serves as V(s_{t+1}) baseline at the
next step instead of an independent second draw (equal in distribution).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from parallel_risk import ParallelRiskEnv
from parallel_risk.env.map_config import MapRegistry
from parallel_risk.env.reward_shaping import RewardShapingConfig
from parallel_risk.training.torchrl.graph_wrapper import GraphObservationWrapper, env_to_graph


AGENTS = ('agent_0', 'agent_1')  # == sorted(obs.keys()); both agents are live until the episode ends

# Per-column tensors of a collection; concatenated along dim=1 when merging workers.
_COLUMN_KEYS = ('actions', 'log_probs', 'values', 'rewards', 'dones', 'terminateds', 'next_values')


def map_edge_index(map_name: str, max_regions: Optional[int] = None) -> torch.Tensor:
    """edge_index exactly as env_to_graph builds it for this map (it is map-constant)."""
    env = ParallelRiskEnv(map_name=map_name)
    obs, _ = env.reset()
    graph = env_to_graph(obs['agent_0'], env.map_config, torch.device('cpu'), max_regions=max_regions)
    return graph.edge_index


class _Slot:
    """One env slot: the live env (one lazily built per map), its observation and episode counters."""

    __slots__ = ('envs_by_map', 'env', 'map_idx', 'obs', 'episode_reward', 'episode_length')

    def __init__(self):
        self.envs_by_map = {}
        self.env = None
        self.map_idx = -1
        self.obs = None
        self.episode_reward = {a: 0.0 for a in AGENTS}
        self.episode_length = 0


class LockstepRollout:
    """
    N Parallel Risk envs stepped in lockstep by one policy (see module docstring).

    Args:
        policy: GCNPolicy (any device). Used in its current train/eval mode.
        action_decoder: ActionDecoder matching the policy's action budget.
        map_names: maps to sample from at every reset.
        num_envs: envs stepped in lockstep (B = 2 * num_envs samples per step).
        action_budget: actions per agent per turn (K).
        max_regions: region padding width shared by all maps (see GraphObservationWrapper).
        max_turns: env turn limit (truncation).
        env_seed: config env seed; seeds the global random/np.random once here,
            which is what constructing the envs with that seed used to do.
        use_reward_shaping: build envs with the default RewardShapingConfig.
        rng: object with .randint(n) for map choice (np.random or a RandomState).
        reset_seed_base: when given, the k-th reset of a collection is seeded
            reset_seed_base + k; when None resets are unseeded.
    """

    def __init__(self, policy, action_decoder, map_names: Sequence[str], num_envs: int,
                 action_budget: int, max_regions: Optional[int] = None, max_turns: int = 100,
                 env_seed: Optional[int] = None, use_reward_shaping: bool = True,
                 rng=None, reset_seed_base: Optional[int] = None):
        self.policy = policy
        self.action_decoder = action_decoder
        self.map_names = list(map_names)
        self.num_envs = int(num_envs)
        self.action_budget = int(action_budget)
        self.env_max_actions = max(10, self.action_budget)  # env-side padding, as before
        self.max_regions = max_regions
        self.max_turns = max_turns
        self.use_reward_shaping = use_reward_shaping
        self.rng = rng if rng is not None else np.random
        self.reset_seed_base = reset_seed_base
        self.device = next(policy.parameters()).device
        self._n_territories = [MapRegistry.get(m).n_territories for m in self.map_names]
        self._slots = [_Slot() for _ in range(self.num_envs)]
        if env_seed is not None:
            import random
            random.seed(env_seed)
            np.random.seed(env_seed)
        self._reset_count = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_map_names: List[str] = []
        self._clear_buffers()

    # ------------------------------------------------------------------ envs

    def _make_env(self, map_idx: int) -> GraphObservationWrapper:
        shaping = RewardShapingConfig() if self.use_reward_shaping else None
        env = ParallelRiskEnv(
            map_name=self.map_names[map_idx], max_turns=self.max_turns, seed=None,
            reward_shaping_config=shaping, max_actions_per_turn=self.env_max_actions,
        )
        # Graphs are built on CPU; the per-step Batch is moved to the policy device once.
        return GraphObservationWrapper(env, device=torch.device('cpu'), max_regions=self.max_regions)

    def _reset_slot(self, i: int):
        """Start a new episode in slot i on a freshly drawn map."""
        slot = self._slots[i]
        map_idx = int(self.rng.randint(len(self.map_names)))
        env = slot.envs_by_map.get(map_idx)
        if env is None:
            env = slot.envs_by_map[map_idx] = self._make_env(map_idx)
        seed = None if self.reset_seed_base is None else int(self.reset_seed_base + self._reset_count)
        self._reset_count += 1
        slot.obs, _ = env.reset(seed=seed)
        slot.env = env
        slot.map_idx = map_idx
        slot.episode_reward = {a: 0.0 for a in AGENTS}
        slot.episode_length = 0

    @property
    def slots(self) -> List[_Slot]:
        """The env slots (tests poke at slot.env.env.game_state to force terminations)."""
        return self._slots

    # ------------------------------------------------------------- collection

    def _clear_buffers(self):
        self._x, self._gf, self._num_nodes, self._map_idx = [], [], [], []
        self._actions, self._log_probs, self._values = [], [], []
        self._rewards, self._dones, self._terminateds, self._boot = [], [], [], []

    def begin(self):
        """Start a collection: fresh episodes in every slot, empty buffers and episode stats."""
        self._reset_count = 0
        self._clear_buffers()
        self.episode_rewards, self.episode_lengths, self.episode_map_names = [], [], []
        for i in range(self.num_envs):
            self._reset_slot(i)

    def _forward(self, graphs: List[Data]):
        """One policy forward over a list of CPU graphs; returns (logits, values[B], batch on device)."""
        bg = Batch.from_data_list(graphs)
        bg_dev = bg.to(self.device, non_blocking=True) if bg.x.device != self.device else bg
        logits, values, _ = self.policy(bg_dev)
        return logits, values.squeeze(-1), bg, bg_dev

    def _decode(self, logits, bg_dev, graphs):
        # Distribution argument validation costs a sync per Categorical (3 per action
        # slot); it only raises on non-finite logits, sampling is unaffected.
        prev = torch.distributions.Distribution._validate_args
        torch.distributions.Distribution.set_default_validate_args(False)
        try:
            return self.action_decoder.decode_actions(
                logits, bg_dev.batch, deterministic=False, return_log_probs=True,
                observations=graphs, batched_obs=bg_dev,
            )
        finally:
            torch.distributions.Distribution.set_default_validate_args(prev)

    def step(self):
        """One lockstep step: forward + decode for all envs, step every env, buffer the transition."""
        slots = self._slots
        K = self.action_budget
        B = 2 * self.num_envs
        graphs = [slot.obs[a] for slot in slots for a in AGENTS]
        step_map_idx = [slot.map_idx for slot in slots for _ in AGENTS]

        with torch.no_grad():
            logits, values, bg, bg_dev = self._forward(graphs)
            actions, log_probs = self._decode(logits, bg_dev, graphs)
        acts = actions.cpu().numpy()  # one device->host transfer for all envs

        rewards = np.zeros(B, dtype=np.float32)
        dones = np.zeros(B, dtype=bool)
        terminateds = np.zeros(B, dtype=bool)
        boot = np.zeros(B, dtype=np.float32)  # bootstrap for done columns: 0 terminated, V(final) truncated
        trunc_graphs, trunc_cols = [], []

        for i, slot in enumerate(slots):
            c = 2 * i
            action_dict = {}
            for j, agent in enumerate(AGENTS):
                padded = np.zeros((self.env_max_actions, 3), dtype=np.int64)
                padded[:K] = acts[c + j]
                action_dict[agent] = {'num_actions': K, 'actions': padded}
            next_obs, step_rewards, terms, truncs, _ = slot.env.step(action_dict)

            for j, agent in enumerate(AGENTS):
                r = float(step_rewards[agent])
                slot.episode_reward[agent] += r
                rewards[c + j] = r
            slot.episode_length += 1

            terminated = bool(terms.get('__all__', False))
            truncated = bool(truncs.get('__all__', False))
            if terminated or truncated:
                dones[c:c + 2] = True
                terminateds[c:c + 2] = terminated
                if truncated:
                    # The env returns the final observation on the truncating step.
                    trunc_graphs.extend(next_obs[a] for a in AGENTS)
                    trunc_cols.extend((c, c + 1))
                self.episode_rewards.append(slot.episode_reward['agent_0'])
                self.episode_lengths.append(slot.episode_length)
                self.episode_map_names.append(self.map_names[slot.map_idx])
                self._reset_slot(i)
            else:
                slot.obs = next_obs

        if trunc_graphs:
            with torch.no_grad():
                _, v_final, _, _ = self._forward(trunc_graphs)
            boot[trunc_cols] = v_final.cpu().numpy()

        self._x.append(bg.x)
        self._gf.append(bg.global_features)
        self._num_nodes.extend(self._n_territories[m] for m in step_map_idx)
        self._map_idx.extend(step_map_idx)
        self._actions.append(actions)
        self._log_probs.append(log_probs)
        self._values.append(values)
        self._rewards.append(rewards)
        self._dones.append(dones)
        self._terminateds.append(terminateds)
        self._boot.append(boot)

    def finish(self) -> Dict[str, object]:
        """
        Close the collection: bootstrap the last step and return compact CPU tensors.

        Keys: T, B; x [total_nodes, F], global_features [T*B, G], num_nodes [T*B],
        map_idx [T*B] (samples in (t, column) order); actions [T, B, K, 3],
        log_probs [T, B, K], values/rewards/next_values [T, B], dones/terminateds
        [T, B] bool; episode_rewards, episode_lengths, episode_map_names (lists,
        one entry per completed episode, agent_0's return).
        """
        T = len(self._rewards)
        B = 2 * self.num_envs
        with torch.no_grad():
            _, v_last, _, _ = self._forward([slot.obs[a] for slot in self._slots for a in AGENTS])
        values = torch.stack(self._values)  # [T, B] on the policy device
        values_next = torch.cat([values[1:], v_last.unsqueeze(0)], dim=0)  # V(s_{t+1}) of the same column
        dones = torch.from_numpy(np.stack(self._dones))
        boot = torch.from_numpy(np.stack(self._boot)).to(self.device)
        next_values = torch.where(dones.to(self.device), boot, values_next)
        out = {
            'T': T, 'B': B,
            'x': torch.cat(self._x),
            'global_features': torch.cat(self._gf),
            'num_nodes': torch.tensor(self._num_nodes, dtype=torch.long),
            'map_idx': torch.tensor(self._map_idx, dtype=torch.long),
            'actions': torch.stack(self._actions).cpu(),
            'log_probs': torch.stack(self._log_probs).cpu(),
            'values': values.cpu(),
            'next_values': next_values.cpu(),
            'rewards': torch.from_numpy(np.stack(self._rewards)),
            'dones': dones,
            'terminateds': torch.from_numpy(np.stack(self._terminateds)),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'episode_map_names': list(self.episode_map_names),
        }
        self._clear_buffers()
        return out

    def collect(self, num_steps: int) -> Dict[str, object]:
        """begin(); num_steps lockstep steps; finish(). num_steps * num_envs env steps in total."""
        self.begin()
        for _ in range(num_steps):
            self.step()
        return self.finish()


# ---------------------------------------------------------------------------
# Transport and rebuild helpers
# ---------------------------------------------------------------------------

def arrays_to_numpy(arrays: Dict[str, object]) -> Dict[str, object]:
    """Worker-side: tensors -> numpy so the result pickles compactly."""
    return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in arrays.items()}


def arrays_from_numpy(arrays: Dict[str, object]) -> Dict[str, object]:
    """Parent-side inverse of arrays_to_numpy."""
    return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}


def merge_arrays(parts: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Merge collections with the same T along the column axis (worker w's
    columns follow worker w-1's). Samples stay in (t, column) order.
    """
    if len(parts) == 1:
        return parts[0]
    T = parts[0]['T']
    if any(p['T'] != T for p in parts):
        raise ValueError(f"cannot merge collections with different T: {[p['T'] for p in parts]}")
    B = sum(p['B'] for p in parts)
    # x is node-major: split each part per step, then interleave the parts step by step.
    x_per_step = [p['x'].split(p['num_nodes'].view(T, -1).sum(dim=1).tolist()) for p in parts]
    merged = {
        'T': T, 'B': B,
        'x': torch.cat([chunks[t] for t in range(T) for chunks in x_per_step]),
        'global_features': torch.cat(
            [p['global_features'].view(T, p['B'], -1) for p in parts], dim=1).reshape(T * B, -1),
        'num_nodes': torch.cat([p['num_nodes'].view(T, -1) for p in parts], dim=1).reshape(-1),
        'map_idx': torch.cat([p['map_idx'].view(T, -1) for p in parts], dim=1).reshape(-1),
        'episode_rewards': [r for p in parts for r in p['episode_rewards']],
        'episode_lengths': [l for p in parts for l in p['episode_lengths']],
        'episode_map_names': [m for p in parts for m in p['episode_map_names']],
    }
    for key in _COLUMN_KEYS:
        merged[key] = torch.cat([p[key] for p in parts], dim=1)
    return merged


def build_rollout(arrays: Dict[str, object], edge_index_by_map: List[torch.Tensor],
                  device: torch.device) -> Dict[str, object]:
    """
    Rebuild the rollout dict consumed by PPOTrainer.update_policy on `device`.

    graph_lists[t][b] is a Data whose x / global_features are views into two
    device tensors and whose edge_index is the map's shared template, so moving
    a rollout to CUDA is two copies rather than one per graph.
    """
    T, B = arrays['T'], arrays['B']
    x = arrays['x'].to(device, non_blocking=True)
    gf = arrays['global_features'].to(device, non_blocking=True)
    num_nodes = arrays['num_nodes'].tolist()
    map_idx = arrays['map_idx'].tolist()
    graph_lists = []
    offset = 0
    i = 0
    for _ in range(T):
        step_graphs = []
        for _ in range(B):
            n = num_nodes[i]
            g = Data(x=x[offset:offset + n], edge_index=edge_index_by_map[map_idx[i]], num_nodes=n)
            g.global_features = gf[i:i + 1]
            step_graphs.append(g)
            offset += n
            i += 1
        graph_lists.append(step_graphs)
    rollout = {'graph_lists': graph_lists, 'map_names': list(arrays['episode_map_names'])}
    for key in _COLUMN_KEYS:
        rollout[key] = list(arrays[key].to(device, non_blocking=True).unbind(0))
    return rollout
