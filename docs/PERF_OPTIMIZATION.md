# Performance Optimization (branch `perf/parallel-gpu`)

Goal: make experiments run faster through parallelism and GPU use
**without changing any RL technique**. PPO+GNN, Decoupled-UCT MCTS
(PUCT, Dirichlet noise, temperature sampling), Expert Iteration and
AZ-distillation must compute the same quantities. Refactors, batching,
vectorization, caching and process layout are fair game. A change that
only alters floating-point summation order or the order in which random
numbers are drawn is acceptable but must be declared.

Companion: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (research results),
[RL_TRAINING_ROADMAP.md](RL_TRAINING_ROADMAP.md) "Engineering Roadmap"
(which already lists batched MCTS leaf evaluation and faster `env.step`).

## Validation protocol

Every optimization commit must pass, in this order:

1. `python run_tests.py` equivalent: all `tests/test_*.py` that passed on
   the base commit still pass (`test_rllib_wrapper.py` fails on this
   machine for an unrelated reason: importing Ray before torch breaks
   the torch DLL load; ignore it).
2. **Golden trajectories**:
   `python tests/golden_harness.py record --out /tmp/new.json` then
   `python tests/golden_harness.py compare --golden tests/golden/golden.json --new /tmp/new.json`.
   Sections: (a) masked-random games, (b) MCTS-uniform root stats,
   (c) MCTS+GNN root stats and priors, (d) a PPO rollout, (e) an AZ
   self-play episode. Integers and structure must match exactly; floats
   within 1e-5. A change that legitimately alters an RNG stream (for
   example a vectorized sampler) must say which section changed and why,
   and add statistical evidence instead (win-rate distribution over
   seeded games, or a learning curve).
3. **Benchmark**: `python experiments/benchmark_suite.py --label <name>`
   then `--compare experiments/benchmarks/baseline.json experiments/benchmarks/<name>.json`.
   Report the speedup table in the commit message.
4. **Learning evidence** for anything touching training loops: a short
   training run whose curve matches the baseline run under the same
   seeds and config.

Always launch Python as `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python ...`.

## Where the time goes (profiled 2026-09-09 on commit f2f3dd7)

Machine: 16 logical CPUs, RTX 5080. Shares are from cProfile and
microbenchmarks. Baseline absolute numbers
(`experiments/benchmarks/baseline.json`, quick suite, idle machine):

| Scenario | Baseline |
|---|---|
| env step, masked-random self-play, K=10 (any map size) | 0.32 ms |
| simulator turn (MCTS rollout primitive) | 0.30 ms |
| MCTS-uniform decision, budget 200, K=10, 12 to 30 territories | 1.8 to 2.4 s |
| MCTS+GNN decision, budget 200, K=10, CPU | 1.9 to 2.3 s |
| MCTS+GNN decision, budget 200, K=10, CUDA (batch-1 calls) | 6.3 to 6.9 s |
| PPO rollout of 1024 samples (3 maps, 1 CPU worker) | 2.9 s |
| PPO rollout of 1024 samples (6 maps, K=10, 8 workers) | 2.3 to 2.9 s |
| PPO update of 1024 samples on GPU | 0.35 to 0.6 s |
| ExIt self-play, 12 games, budget 40, 12 workers | 28 s |
| eval event, 3 maps x 2 games, budget 40 | 59 s |

### MCTS-uniform decision (budget 200, K=10)

| Component | Share | Notes |
|---|---|---|
| Random rollouts (up to 50 turns each) | 85 to 91% | on 30-territory maps every rollout hits the cap |
| of which masked-random sampler `get_action_raw` | 72 to 78% | 120 to 135 us per call: 3 `np.random.choice` + 3 `np.where` + 2 `np.zeros` per action slot |
| of which `RiskSimulator.step` | 13% | 35 to 40 us; python dict actions, `ActionValidator` per step |
| Tree selection, expansion, backprop | under 10% | |
| `clone_state` | 0.2% | already cheap (0.3 us); `deepcopy` is not used |

Measured, bit-identical fixes: `np.random.choice(a)` draws exactly what
`a[np.random.randint(len(a))]` draws, so the sampler can be rewritten
3.3 to 3.7x faster with the same RNG stream (verified over 1200
observation/seed pairs including the post-call RNG state); a list-based
`RiskSimulator.step` is 5x faster and `_check_terminal` 13x faster with
identical outputs. End to end, a decision runs 3.6x faster with identical
root visit counts and chosen action.

### MCTS+GNN decision (budget 200, K=10, CPU)

| Component | Share | Notes |
|---|---|---|
| Network calls | 94.5% | 789 forwards per decision: 589 `policy_fn` (one per sampled action) + 200 `value_fn` |
| of which `policy.forward` | 52% | 1.7 ms per single-graph call on CPU |
| of which `compute_log_probs` glue | 29% | `_BatchGeom(observations=)` 0.27 ms vs 0.075 ms with pre-batched geometry |
| Masked-random sampler | 7.5% | |
| `env_to_graph` | 2 to 5% | O(n^2) python edge loop is 67% of it on 30-node maps; edge_index is map-constant |

The same state is forwarded up to four times (value for agent_0, prior
for each sampled action of each agent). Caching one forward per
(state, agent) inside a search cuts forwards to 397 (bit-identical) or
199 when both agents' graphs share one batch of 2 (priors within 4e-6).
Prototype: 3.56x per decision. GPU forward latency is 4.5 ms at batch 1
but stays 4.5 ms at batch 128 (28k graphs/s), so batching leaves across
concurrent games is the route to the "network -> 0" ceiling of 18x.

### PPO rollout (batch 1024 samples)

| Layout | Time | Notes |
|---|---|---|
| 1 env per worker, CPU | 2.3 to 3.0 s | forward 43%, decoder 41%, `Batch.from_data_list` 14%, env 2% |
| same on GPU | 10.7 s | every step is a batch-of-2 forward plus `.item()` syncs |
| lockstep N=64 to 128 envs, CPU (prototype) | 0.19 to 0.25 s | one batched forward per step |
| lockstep N=128 envs, GPU (prototype) | 0.13 to 0.15 s | env stepping becomes 65% of the remainder |

Worker overheads: PyG rollout pickles are 5.5 MB per 1024 samples (0.7 s
to dump and load); the same data as numpy arrays is 0.23 MB and 0.1 ms.
Workers do not pin torch threads; 16 threads per worker is 1.7x slower
than 1 thread. The bootstrap value V(s') is recomputed with a second
forward every step although the next step forwards the same graph.

### Environment step

`ParallelRiskEnv.step` is 67 to 91 us (111 us with reward shaping); a
list-based implementation with precomputed adjacency sets and one region
scan per agent per phase is 2.3 to 2.6x faster with identical
observations, rewards, infos and state over 400 checked steps.
`env_to_graph` is 29 to 85 us; precomputing the static node features and
edge_index per map gives 18 us with identical tensors.

### Process harness

Spawn-pool start costs about 1 s per pool (torch import in each worker);
negligible against 400 s iterations. Eval tasks are per map, so the
30-territory maps set the tail of every eval event.

## Plan

Wave 1 (independent files, mostly bit-identical):

- **W1-A env core**: fast `ParallelRiskEnv.step` and `RiskSimulator`
  (`step`, `_check_terminal`, `state_to_obs`); `ActionValidator` stays the
  validation authority but works on list-mirrored state.
- **W1-B sampler + graph builder**: bit-identical masked-random sampler;
  per-map cached static graph features in `env_to_graph`.
- **W1-C PPO rollout**: lockstep vectorized rollout (N envs, one forward
  per step), V(s') reuse, compact worker transfer, pinned worker threads,
  optional single-process GPU rollout.
- **W1-D MCTS+GNN evaluator**: per-search neural cache with one forward
  per (state, agent), vectorized prior computation from cached logits.

Wave 2: cross-game batched leaf evaluation on the GPU for MCTS+GNN
self-play and eval. Design:

- **Search as a generator.** `DuctMCTS` keeps its algorithm verbatim, but
  the call chain that can need a network output (`make_root`,
  `apply_root_dirichlet`, `_select` via `_add_sampled_action`, and
  `_rollout` when a value function is set) is written as generators that
  `yield` an evaluation request `(game_state, agent_id)` whenever the
  evaluator cache misses, and continue once the entry exists. `run()`
  stays a plain method: it drives the generator and services each
  request immediately with a single-graph forward, so the synchronous
  path is bit-identical to today (golden sections c and e).
- **Lockstep driver.** A `GameBatch` holds G games (any mix of maps),
  each with its own env, search generator, and saved `np.random` state.
  One round = resume every game until it yields or finishes its move,
  collect the pending requests, deduplicate keys, run ONE batched
  forward for all of them, fill every game's evaluator cache, repeat.
  Before resuming a game the driver restores that game's RNG state and
  saves it afterwards (about 20 us per switch), so each game consumes
  exactly the random stream it would consume when played alone with the
  same seed. When a node is expanded, both agents' graphs are requested
  together so the child's value and both priors arrive in one round.
- **Process layout.** P worker processes (12 on this machine), each
  owning a CUDA context and G games (16 to 32); batch sizes of 16 to 64
  graphs per forward put the GPU in its flat-latency regime (4.5 ms per
  call at any batch size up to 128). Self-play (`collect_games_exit`,
  `collect_games`) and eval (`evaluate_all_maps`) both use the driver.
  Python bookkeeping (sampler, simulator, tree) is the CPU cost that
  remains, about 1 ms per simulation, so P stays close to the core
  count.
- **What changes numerically.** Batched forwards reduce in a different
  order than single-graph forwards (observed 8e-6 on log-probs at
  batch 2 on CPU; TF32 on the GPU is larger), so a PUCT argmax can flip
  and lockstep games are not trajectory-identical to sequential ones.
  Validation: per-request priors/values match the sequential evaluator
  within tolerance; win/draw/length distributions over seeded game sets
  match the sequential driver; ExIt learning curve matches the baseline
  run. The synchronous path (eval scripts, tests, golden harness) stays
  bit-identical.
- **Expected effect.** With the network cost amortized across G games,
  a K=10, budget-200 decision costs about 200 rounds of (4.5 ms GPU +
  G x python per simulation) per G games, roughly 0.25 s per decision
  per process against 2.2 s today, about 9x on ExIt self-play iterations
  and on the MCTS+GNN side of eval events. The MCTS-uniform opponent in
  eval is CPU-only and gets only the wave-1 gains.

## Results

### Wave 1 (commits 2c340e8, c979d5b, 801a0e9, 6ed58cc, 308cd40)

Clean benchmark on an idle machine, `experiments/benchmarks/baseline.json`
(e6991e7) versus `experiments/benchmarks/wave1.json` (308cd40), quick
suite. All four bit-identical changes pass the golden harness at 1e-9
tolerance; the PPO rollout change is declared non-identical (sample
composition and RNG order) and re-baselined golden section d.

| Workload | Before | After | Speedup |
|---|---|---|---|
| env step, K=10 (6 to 30 territories) | 0.32 ms | 0.10 ms | 3.3x |
| simulator turn | 0.30 ms | 0.09 ms | 3.3x |
| MCTS-uniform decision, budget 200, K=10 | 1.8 to 2.4 s | 0.54 to 0.74 s | 3.2 to 3.4x |
| MCTS+GNN decision, CPU | 1.9 to 2.3 s | 0.71 to 0.88 s | 2.6 to 2.8x |
| MCTS+GNN decision, CUDA (single-graph path) | 6.3 to 6.9 s | 2.7 to 3.1 s | 2.1 to 2.8x |
| PPO rollout, 1024 samples, 3 maps, 1 CPU worker | 2.91 s | 0.27 s | 10.7x |
| PPO rollout, 3 maps, 4 workers, GPU update | 2.20 s | 0.09 s | 24x |
| PPO rollout, 6 maps K=10, 8 workers | 2.3 to 2.9 s | 0.14 to 0.19 s | 15 to 16x |
| PPO iteration (rollout + update), GPU | 2.6 to 3.4 s | 0.41 to 0.63 s | 5.4 to 6.3x |
| ExIt self-play, 12 games, budget 40, 12 workers | 28.3 s | 13.8 s | 2.05x |
| eval event, 3 maps x 2 games, budget 40 | 59.2 s | 23.2 s | 2.55x |

The PPO update step is now the larger half of an iteration on the GPU
and dominates on CPU at K=10 (2.5 s of 2.65 s). MCTS+GNN decisions are
still about 400 single-graph forwards each; wave 2 batches them across
games.

### Wave 2: lockstep games, batched evaluation, and why the GPU lost

`DuctMCTS` can now be stepped as a generator that yields the
(game_state, agent_id) graphs its evaluator has not cached. `GameBatch`
in `parallel_risk/training/mcts_gnn/lockstep.py` holds G such games,
resumes each one (restoring and saving its private `random` and
`np.random` state so it consumes exactly the stream it would alone),
and serves all pending requests with one batched forward per round.
Self-play and evaluation both use it; `run()` and the other public
methods still drive the generators synchronously, so every existing
caller and the golden harness are unchanged.

Measured on an idle machine, 24 ExIt self-play games over the 21-map
roster, 12 workers:

| Layout | budget 40 | budget 200 |
|---|---|---|
| one game per worker (old) | 0.95 s/game | 3.89 s/game |
| 16 games per worker, CPU forwards | 0.69 s/game | 2.59 s/game |
| 16 games per worker, CUDA forwards | 2.64 s/game | 12.08 s/game |
| 4 workers x 16 games, CUDA | 1.52 s/game | 6.90 s/game |

**The GPU is the wrong device for this workload, at any batch size we
can reach.** Batches average 23 graphs per forward. Our graphs are tiny
(6 to 30 nodes, hidden 128, 3 layers) so such a forward costs a few
milliseconds on one core, while a CUDA forward costs about 4.5 ms of
fixed latency no matter the batch, and 12 worker processes' CUDA
contexts serialize on the single device. Cutting to 4 processes reduces
the contention but starves the tree search, which is python-bound and
wants every core. CUDA would only pay if one process held hundreds of
games, and then a single python thread would have to do all the
bookkeeping. So `device: cpu` is the default in both configs, and the
gain here comes from batching across games on the CPU, not from the GPU.

Equivalence (48 seeded games on simple_6, dense_12, hub_ring_30 at
budget 100, lockstep versus the same games played one at a time):
trajectory-identical in 47 of 48 games on CPU and 46 of 48 on CUDA;
outcome counts identical (chi-square p = 1.00) and game-length
distributions identical (Kolmogorov-Smirnov p = 1.00) in both. Per
request the batched forward differs from the single-graph forward by at
most 7.6e-6 on a summed log-prior and 4.8e-7 on a value, which is
occasionally enough to flip a PUCT argmax. `single_graph_forwards=True`
turns batching off and is exact; `tests/test_lockstep.py` uses it to
check the driver and the RNG swapping separately from float effects.

### PPO update path: decoder reads the mega-batch it already has

`_update_policy_impl` builds one PyG mega-batch of all T x B graphs per
update, but `compute_log_probs` and `compute_entropy` were re-collating
the same graph list twice per epoch to derive their masks. Passing
`batched_obs=mega_batch` removes both re-collations. Golden section d
(including the parameter sum after one update) is exact at 1e-9. In an
interleaved same-process A/B on a 1024-sample mega-batch of six maps at
K=10, log-probs plus entropy per epoch went from 30.4 ms to 13.2 ms on
CPU (2.3x) and from 69.6 ms to 11.8 ms on CUDA (5.9x); with 10 epochs per
iteration that is about 0.6 s saved per GPU iteration on the K-sweep
configuration.
