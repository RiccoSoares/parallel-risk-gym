# Visualization Guide

## Installation

Matplotlib is required for generating plots:
```bash
pip install matplotlib>=3.7.0
```

Or install from requirements.txt (includes all dependencies):
```bash
pip install -r requirements.txt
```

## Automatic Plot Generation

The validation experiment script automatically generates plots when it completes:

```bash
PYTHONPATH=. python experiments/validate_learning.py \
    --num-iterations 100 \
    --eval-interval 20
```

**Output:** Creates plots in `experiments/validate_learning_results/`:
- `win_rate_curve.png` - Win rate vs. training iteration
- `episode_length_curve.png` - Episode length progression
- `reward_distribution.png` - Histogram of episode lengths

## Manual Plot Generation

Generate plots from existing evaluation results:

```bash
PYTHONPATH=. python -m parallel_risk.evaluation.visualize \
    experiments/your_results/evaluation_results.json \
    experiments/your_results/
```

Or use Python:
```python
from parallel_risk.evaluation.visualize import plot_all

# Generate all plots
plot_all("experiments/sparse_validation_results/evaluation_results.json")

# Or individual plots
from parallel_risk.evaluation.visualize import (
    plot_win_rate_curve,
    plot_episode_length_curve,
    plot_reward_distribution
)

plot_win_rate_curve(
    "experiments/sparse_validation_results/evaluation_results.json",
    output_path="my_win_rate.png"
)
```

## Plot Types

### 1. Win Rate Curve
Shows learning progression over training iterations.

**Features:**
- Win rate vs. iteration
- Reference lines at 50% (random baseline) and 70% (target)
- X-axis: Training iteration
- Y-axis: Win rate (0-100%)

**Interpretation:**
- Flat line at 50%: No learning
- Increasing curve: Agent improving
- Plateau: Convergence

### 2. Episode Length Curve
Shows how game duration changes as agent improves.

**Features:**
- Average episode length with ±1 standard deviation shading
- X-axis: Training iteration
- Y-axis: Episode length (turns)

**Interpretation:**
- Decreasing length: Agent learning to win faster
- High variance: Inconsistent performance
- Stabilizing: Convergent strategy

### 3. Reward Distribution
Histogram of episode lengths for a specific checkpoint.

**Features:**
- Distribution of game durations
- Mean and standard deviation overlay
- Useful for understanding performance variance

**Interpretation:**
- Tight distribution: Consistent performance
- Bimodal: Different strategies for different scenarios
- Long tail: Occasional difficult games

## Example Results

From our 100-iteration sparse reward experiment:

**Win Rate Curve:**
- Started at ~50% (random)
- Progressed to 100% by iteration 100
- Clear learning signal

**Episode Length:**
- Started at ~71 turns (random baseline)
- Decreased to ~61 turns (trained agent)
- Agent wins faster (14% improvement)

## Troubleshooting

### No plots generated
**Issue:** Matplotlib not installed
**Solution:** `pip install matplotlib`

### Plots are empty
**Issue:** No evaluation data in JSON
**Solution:** Ensure evaluation ran successfully, check `evaluation_results.json`

### Import errors
**Issue:** Missing PYTHONPATH
**Solution:** Run with `PYTHONPATH=. python ...`

### Font cache warning
**Issue:** First-time matplotlib use builds font cache
**Solution:** Wait ~30 seconds, this only happens once

## Customization

Modify plot appearance in `parallel_risk/evaluation/visualize.py`:

- **Figure size:** Change `figsize=(10, 6)` parameter
- **Colors:** Modify line/marker colors
- **DPI:** Change `dpi=150` for higher/lower resolution
- **Style:** Add `plt.style.use('seaborn')` for different themes

## CI/CD Integration

For automated testing without display:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

This is already set in `visualize.py` for compatibility with headless environments.
