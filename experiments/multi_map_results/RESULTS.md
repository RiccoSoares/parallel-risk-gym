## Multi-Map Training Experiment Results

### Quick mode (30 iterations) — smoke test only

These results are from `--quick` mode (30 iterations, batch_size=400) and are
intended only to verify the pipeline works end-to-end. They are NOT research-quality.

**Per-map win rates at end of quick training:**
| Map | Win rate (30 iters, quick) | Notes |
|-----|--------------------------|-------|
| simple_6 | 42.5% | Early learning visible |
| medium_8 | 0% | Bridge topology harder; needs more iterations |
| large_10 | 12.5% | Minimal learning |

**Transfer test:** 2-map model (simple_6 + medium_8) on large_10 zero-shot: 15%
vs 3-map model: 5% — both low; high draw rate reflects max_turns=50 cut-off.

### Run the full experiment

To obtain publication-quality results, run:

```bash
PYTHONPATH=. python experiments/multi_map_training.py \
    --num-iterations 200 \
    --output-dir experiments/multi_map_results_full
```

Expected outcome at 200 iterations (based on single-map training history):
- simple_6: >90% win rate (achieved 99.5% in Phase 2 revalidation)
- medium_8: ~70-90% win rate (bridge topology)
- large_10: ~70-90% win rate (corridor + flanks topology)
- Transfer test: 2-map zero-shot should show meaningful difference vs baseline

Full run will take approximately 1-2 hours on CPU.
