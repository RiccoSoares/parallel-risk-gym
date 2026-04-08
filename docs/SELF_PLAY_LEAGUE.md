# Self-Play League Experiment

This document describes the new self-play league experiment system for Parallel Risk.

## Overview

The self-play league experiment trains a PPO agent with sparse rewards using self-play, then evaluates its learning by testing it against:
1. **Random baseline** - measures general competence
2. **Historical snapshots** - measures improvement over past versions of itself

This provides stronger evidence of learning than evaluating against random alone.

## Components

### 1. CheckpointAgent (`parallel_risk/agents/checkpoint_agent.py`)
- Loads a trained RLlib checkpoint and generates actions
- Drop-in replacement for RandomAgent
- Lazy loading + explicit cleanup for memory efficiency

### 2. LeagueEvaluator (`parallel_risk/evaluation/league_evaluator.py`)
- Orchestrates evaluation of one policy against multiple opponents
- Memory-efficient: loads one opponent at a time
- Supports both random and checkpoint-based opponents

### 3. League Visualization (`parallel_risk/evaluation/league_visualize.py`)
- **Multi-opponent win rates**: Line plot showing win rate vs each opponent over training
- **Win rate heatmap**: Full matchup matrix visualization
- **Aggregate learning curve**: Mean/std win rate across all opponents
- **Episode length trends**: How game length changes by opponent type
- **Dashboard**: Combined view with all key plots

### 4. Main Experiment Script (`experiments/self_play_league.py`)
- Orchestrates complete workflow: training → snapshots → evaluation → visualization
- Follows same pattern as `validate_learning.py`

## Quick Start

### Test Run (10 iterations, ~10 minutes)
```bash
PYTHONPATH=. python experiments/self_play_league.py \
    --num-iterations 10 \
    --snapshot-interval 5 \
    --eval-interval 5 \
    --num-eval-episodes 20 \
    --num-workers 2 \
    --verbose
```

### Full Experiment (500 iterations, ~2-3 hours)
```bash
PYTHONPATH=. python experiments/self_play_league.py \
    --config parallel_risk/training/configs/ppo_sparse.yaml \
    --num-iterations 500 \
    --snapshot-interval 50 \
    --eval-interval 50 \
    --num-eval-episodes 100 \
    --num-workers 4 \
    --verbose
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-iterations` | 500 | Training iterations |
| `--snapshot-interval` | 50 | Save snapshot every N iterations |
| `--eval-interval` | 50 | Evaluate every N iterations |
| `--num-eval-episodes` | 100 | Episodes per opponent matchup |
| `--num-workers` | 4 | Rollout workers for training |
| `--config` | `ppo_sparse.yaml` | Training configuration file |
| `--results-dir` | `experiments/league_results` | Output directory |
| `--checkpoint-dir` | `checkpoints/league_training` | Training checkpoints |
| `--snapshot-dir` | `league_snapshots` | Historical policy snapshots |
| `--skip-training` | False | Skip training, only evaluate existing checkpoints |
| `--verbose` | False | Print detailed progress |

## Output Structure

After running the experiment, you'll get:

```
experiments/league_results/
├── league_results.json          # Complete evaluation data
├── league_summary.txt           # Text summary
├── league_win_rates.png         # Win rate curves for each opponent
├── league_heatmap.png           # Win rate heatmap
├── league_aggregate.png         # Aggregate learning curve
├── league_episode_lengths.png   # Episode length trends
└── league_dashboard.png         # Combined dashboard

checkpoints/league_training/
└── checkpoint_XXXXXX/           # Training checkpoints (every N iters)

league_snapshots/
├── iter_000050/                 # Policy snapshots (every snapshot_interval)
│   ├── metadata.json
│   └── [RLlib checkpoint files]
├── iter_000100/
└── ...
```

## Results JSON Format

```json
{
  "evaluations": {
    "50": {
      "random_baseline": {
        "win_rate": 0.52,
        "loss_rate": 0.45,
        "draw_rate": 0.03,
        "avg_episode_length": 48.5,
        "wins": 52,
        "losses": 45,
        "draws": 3
      },
      "snapshot_iter_50": {...}
    },
    "100": {
      "random_baseline": {...},
      "snapshot_iter_50": {...},
      "snapshot_iter_100": {...}
    }
  },
  "metadata": {
    "map_name": "simple_6",
    "action_budget": 5,
    "reward_shaping": "sparse",
    "snapshot_interval": 50,
    "num_iterations": 500
  }
}
```

## Success Criteria

**Technical Success:**
- Experiment runs without errors
- Memory stays bounded (doesn't grow with number of snapshots)
- All plots generated correctly

**Scientific Success:**
- Win rate vs random increases over training (learning vs baseline)
- Win rate vs early snapshots > 50% for later checkpoints (improvement over past self)
- Aggregate learning curve shows upward trend

## Evaluation-Only Mode

If you already have training checkpoints and snapshots, you can skip training:

```bash
PYTHONPATH=. python experiments/self_play_league.py \
    --skip-training \
    --checkpoint-dir checkpoints/league_training \
    --snapshot-dir league_snapshots \
    --eval-interval 50 \
    --num-eval-episodes 100 \
    --verbose
```

This is useful for:
- Re-evaluating with more episodes
- Generating new plots from existing data
- Testing evaluation code changes

## Reusing Components

The components are designed to be reusable:

### CheckpointAgent in Custom Scripts
```python
from parallel_risk.agents.checkpoint_agent import CheckpointAgent

agent = CheckpointAgent(checkpoint_path="checkpoints/iter_100")
action = agent.get_action(observation)
agent.unload()  # Free memory when done
```

### LeagueEvaluator for Custom Evaluations
```python
from parallel_risk.evaluation.league_evaluator import LeagueEvaluator

evaluator = LeagueEvaluator(env_config={...})

opponent_specs = [
    {"type": "random", "name": "random"},
    {"type": "checkpoint", "path": "checkpoint_1", "name": "opponent_1"},
    {"type": "checkpoint", "path": "checkpoint_2", "name": "opponent_2"},
]

results = evaluator.evaluate_league(
    main_policy_path="my_checkpoint",
    opponent_specs=opponent_specs,
    num_episodes=100
)
```

### League Visualization
```python
from parallel_risk.evaluation.league_visualize import plot_league_results

# Generate all plots from JSON
plot_league_results("league_results.json", output_dir="plots/")
```

## Dependencies

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

New dependency added: `seaborn>=0.12.0` (for heatmap visualization)

## Troubleshooting

**ImportError: No module named 'parallel_risk'**
- Solution: Run with `PYTHONPATH=.` prefix

**Memory issues during evaluation**
- Reduce `--num-eval-episodes`
- Reduce number of snapshots (increase `--snapshot-interval`)
- The system is designed to load one opponent at a time to minimize memory

**Training takes too long**
- Reduce `--num-iterations`
- Increase `--num-workers` (if you have CPU cores available)
- Use `--num-gpus` in config for GPU acceleration

**Plots not generated**
- Check matplotlib/seaborn installation: `pip install matplotlib seaborn`
- Check for errors in console output

## Next Steps

Once the basic league experiment works, you can extend it with:
- **Policy pool training**: Sample opponents from league during training (not just evaluation)
- **ELO ratings**: Compute ELO scores based on matchup results
- **Pruning strategies**: Keep only best/recent snapshots for long experiments
- **Cross-evaluation**: Compare different training runs in same league
- **Additional opponents**: Heuristic agents, human-designed strategies
