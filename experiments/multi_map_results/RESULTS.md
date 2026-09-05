# Multi-Map GNN Training — Results

Full 200-iteration run on all three maps (`simple_6`, `medium_8`, `large_10`).
Opponent for all evaluations below is **MCTS with 50 simulations per move**.

## 3-map model — final performance

After 200 training iterations, evaluated over 100 episodes per map:

| Map        | Win rate | W / L / D          |
|------------|---------:|--------------------|
| `simple_6` |    86%   | 86 / 4 / 10        |
| `medium_8` |   100%   | 100 / 0 / 0        |
| `large_10` |    99%   | 99 / 0 / 1         |

Learning curves per map: `learning_curves.png`.
Final bar chart: `final_performance.png`.

## Learning trajectory (3-map model vs MCTS-50, 50 episodes/map at each eval)

| Iter | simple_6 | medium_8 | large_10 |
|-----:|---------:|---------:|---------:|
|   25 |    4%    |    2%    |    6%    |
|   50 |   28%    |   76%    |   70%    |
|   75 |   48%    |   96%    |   96%    |
|  100 |   74%    |   84%    |   84%    |
|  125 |   86%    |   94%    |   92%    |
|  150 |   82%    |   98%    |   96%    |
|  175 |   88%    |  100%    |  100%    |
|  200 |   94%    |  100%    |   98%    |

`medium_8` and `large_10` saturate around iteration 75; `simple_6` shows
higher variance in the final third of training (bounces between 82–96%).

## Transfer test — 2-map vs 3-map on `large_10`

The 2-map model was trained on `simple_6` + `medium_8` only (200 iterations),
then evaluated **zero-shot on `large_10`** — a map it never saw during
training. Comparison bar chart: `transfer_comparison.png`.

| Model                        | Win rate on `large_10` (vs MCTS-50, 100 episodes) |
|------------------------------|--------------------------------------------------:|
| 2-map (zero-shot on large_10)|                                             90%   |
| 3-map (trained on large_10)  |                                             98%   |

The 8pp gap suggests the GNN generalizes to unseen maps that share the same
region schema, with a modest additional benefit from training on the target
map. Note: the three current maps share the same 3-region structure and
comparable action budgets — a stronger generalization test would use a map
with a different region count or degree distribution.

## Training configuration

- Architecture: GCN policy (3 layers, hidden dim 128) + PPO
- Action budget: 5 per turn per agent
- Batch size: 4096 environment steps per iteration, 10 epochs of SGD
- Reward: dense shaping (all components enabled)
- Rollout: 8 CPU workers (spawn), each running its own copy of the policy
- Update: on GPU (RTX 5080), single mega-batch across all T×B graphs
- Evaluation: parallel across the 3 maps (one process per map)

## Reproducing

Full 200-iteration run (GPU + 8 workers, includes transfer test):

```bash
PYTHONPATH=. python experiments/multi_map_training.py \
    --num-iterations 200 \
    --num-workers 8 \
    --output-dir experiments/multi_map_results
```

Wall-clock: well under 10 minutes end-to-end on RTX 5080 + high-core-count
CPU. See `../profile_baseline/benchmark_baseline.json` and
`../profile_after_phase3/benchmark_baseline.json` for the per-section
speedup that made this possible.

Full JSON of trajectory + config + final results: `multi_map_results.json`.
