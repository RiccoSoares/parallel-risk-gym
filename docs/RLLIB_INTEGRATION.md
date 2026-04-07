# RLlib Integration Guide

**Status:** Implemented, ready for training  
**Last Updated:** 2026-04-07

## Overview

RLlib integration enables training RL agents on Parallel Risk using Ray's distributed training framework. This is Phase 1.2 of the RL training roadmap.

## Components

### 1. Environment Wrapper
**File:** `parallel_risk/training/rllib_wrapper.py`

**Key Features:**
- Converts PettingZoo ParallelEnv to RLlib MultiAgentEnv
- Flattens Dict observations to vectors for neural networks
- Simplifies variable-length actions to fixed budget
- Supports reward shaping configuration

**Action Space Simplification:**
Original Parallel Risk uses variable-length actions (0-10 actions per turn). RLlib needs fixed-size spaces.

**Solution:** Fixed budget approach
- Agent submits exactly N actions per turn (default: 5)
- Each action: `[source_territory, dest_territory, num_troops]`
- Action space: `Tuple of MultiDiscrete([n_territories, n_territories, 20])`

**Observation Space:**
Flattened vector containing:
- Territory ownership (n_territories floats)
- Territory troops (n_territories floats)  
- Adjacency matrix (n_territories² floats)
- Available income (1 float)
- Turn number (1 float)
- Region control (n_regions floats)

Total size: `2*n_territories + n_territories² + 2 + n_regions`

For simple_6 map: 6 territories, 2 regions → **50 dimensions**

### 2. Training Script
**File:** `parallel_risk/training/train_rllib.py`

**Features:**
- Load configuration from YAML
- Set up PPO algorithm with RLlib
- Training loop with checkpointing
- Progress logging
- Automatic stop conditions

**Usage:**
```bash
python -m parallel_risk.training.train_rllib \
    --config parallel_risk/training/configs/ppo_baseline.yaml \
    --num-iterations 100
```

### 3. Configuration
**File:** `parallel_risk/training/configs/ppo_baseline.yaml`

**Configurable Parameters:**
- Environment settings (map, action budget, reward shaping)
- PPO hyperparameters (learning rate, clip param, etc.)
- Training settings (workers, batch size, etc.)
- Network architecture (hidden layers, activation)
- Self-play settings (future: policy pool)
- Checkpointing and logging

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Existing Parallel Risk dependencies

### Install RLlib

**Option 1: Use install script**
```bash
./install_training_deps.sh
```

**Option 2: Manual installation**
```bash
pip install ray[rllib]==2.40.0 torch>=2.0.0 tensorboard>=2.14.0
```

**Verify installation:**
```bash
python -c "import ray; import ray.rllib; print('RLlib ready!')"
```

### Optional: Weights & Biases
For advanced experiment tracking:
```bash
pip install wandb
wandb login
```

Then set `wandb: true` in config YAML.

## Quick Start

### 1. Test the Wrapper

```bash
PYTHONPATH=. python tests/test_rllib_wrapper.py
```

Expected output: All tests pass ✅

### 2. Run Short Training Test (10 iterations)

```bash
python -m parallel_risk.training.train_rllib \
    --config parallel_risk/training/configs/ppo_baseline.yaml \
    --num-iterations 10 \
    --num-workers 2
```

This should complete in ~5-10 minutes and verify everything works.

### 3. Full Training Run

```bash
python -m parallel_risk.training.train_rllib \
    --config parallel_risk/training/configs/ppo_baseline.yaml
```

This will run for 1000 iterations or until convergence.

**Training Time Estimates:**
- CPU-only, 4 workers: ~12-24 hours
- GPU, 8 workers: ~6-12 hours
- Depends on hardware and configuration

## Configuration Guide

### Environment Configuration

```yaml
env:
  map_name: "simple_6"      # Map to train on
  max_turns: 100            # Max episode length
  action_budget: 5          # Actions per turn
  reward_shaping: "sparse"  # Reward shaping type
```

**Reward Shaping Options:**
- `"sparse"` - Only win/loss terminal rewards (baseline)
- `"dense"` - All reward components enabled
- `"territorial"` - Territory + region rewards only
- `"aggressive"` - Troop + strategic rewards only

**Recommendation:** Start with sparse, compare to dense if learning is slow.

### Training Configuration

```yaml
training:
  num_workers: 4            # Parallel rollout workers
  num_envs_per_worker: 1    # Envs per worker
  num_gpus: 0               # GPUs to use (0 or 1)
  train_batch_size: 4000    # Samples per training iteration
  sgd_minibatch_size: 128   # Minibatch size for SGD
  num_sgd_iter: 10          # SGD epochs per batch
```

**Tuning Guidelines:**
- **More workers** = faster data collection, more CPU needed
- **Larger batch** = more stable gradients, more memory needed
- **More SGD iters** = better sample efficiency, slower iterations

**Starting points:**
- Laptop (4 cores): `num_workers=2`, `train_batch_size=2000`
- Desktop (8+ cores): `num_workers=4-8`, `train_batch_size=4000`
- GPU available: Set `num_gpus=1` (doesn't help much for small models)

### PPO Hyperparameters

```yaml
ppo:
  gamma: 0.99           # Discount factor
  lambda: 0.95          # GAE lambda  
  clip_param: 0.2       # PPO clip parameter
  vf_clip_param: 10.0   # Value function clip
  entropy_coeff: 0.01   # Entropy bonus
  lr: 3.0e-4            # Learning rate
```

**Tuning Priority:**
1. **Learning rate** (`lr`) - Most important, try [1e-4, 3e-4, 1e-3]
2. **Entropy coefficient** (`entropy_coeff`) - Affects exploration, try [0.001, 0.01, 0.05]
3. **Clip param** (`clip_param`) - Try [0.1, 0.2, 0.3]

**Default values are good starting points** - only tune if training is unstable or not converging.

### Network Architecture

```yaml
model:
  fcnet_hiddens: [256, 256]   # Hidden layer sizes
  fcnet_activation: "relu"    # Activation function
  vf_share_layers: false      # Separate policy/value nets
```

**Architecture Experiments:**
- Smaller: `[128, 128]` - faster, might underfit
- Default: `[256, 256]` - good balance
- Larger: `[512, 512]` or `[256, 256, 256]` - more capacity, slower

**Other activations:** `"tanh"`, `"elu"`, `"swish"`

## Training Workflow

### Typical Training Session

1. **Start training:**
   ```bash
   python -m parallel_risk.training.train_rllib \
       --config configs/ppo_baseline.yaml \
       --checkpoint-dir checkpoints/run_001
   ```

2. **Monitor progress:**
   - Terminal output shows episode rewards, length, timesteps
   - TensorBoard: `tensorboard --logdir ~/ray_results`
   - Checkpoints saved every 10 iterations

3. **Check convergence:**
   - Episode reward should increase over time
   - Target: reward > 0 (winning more than losing)
   - Against random: should reach >0.8 win rate

4. **Interrupt if needed:**
   - Ctrl+C will save final checkpoint
   - Resume from checkpoint (future feature)

### Expected Learning Curve

**Phase 1: Random play (iterations 1-50)**
- Episode reward: ~0 (50% win rate against self)
- Episode length: varies widely
- Agent explores action space

**Phase 2: Learning basics (iterations 50-200)**
- Episode reward: increases to 0.2-0.5
- Episode length: stabilizes
- Agent learns valid actions, basic tactics

**Phase 3: Strategic play (iterations 200-500)**
- Episode reward: increases to 0.5-0.8
- Agent learns region completion, efficient attacks
- Self-play creates arms race

**Phase 4: Convergence (iterations 500+)**
- Episode reward: plateaus around 0.6-0.8
- Diminishing returns (both agents equally skilled)
- May need opponent variety (policy pool)

### Interpreting Metrics

**Episode Reward Mean:**
- Self-play: ~0 is expected (50/50 against equal opponent)
- Against random: should increase to >0.8
- Use separate evaluation to measure true skill

**Episode Length Mean:**
- Longer: indecisive play, stalemates
- Shorter: decisive victories
- Very short (<10): might be exploiting something

**Timesteps Total:**
- Total environment steps across all workers
- More = more experience = better learning

## Troubleshooting

### Training is too slow
- **Increase workers:** `--num-workers 8`
- **Decrease batch size:** Smaller batches train faster but less stable
- **Use GPU:** `--num-gpus 1` (limited benefit for small models)

### Agent not learning
- **Check reward shaping:** Try sparse vs. dense
- **Reduce learning rate:** Try `lr: 1.0e-4`
- **Increase exploration:** Try `entropy_coeff: 0.05`
- **Check logs:** Look for NaN or exploding values

### Training crashes
- **Out of memory:** Reduce `train_batch_size` or `num_workers`
- **Ray errors:** Update Ray: `pip install -U ray[rllib]`
- **Import errors:** Check PYTHONPATH, verify installation

### Rewards not changing
- **Sparse rewards issue:** Episodes might be too long to get feedback
  - Try shaped rewards: `reward_shaping: "dense"`
  - Reduce `max_turns: 50`
- **Exploration issue:** Increase `entropy_coeff`

## Advanced Features

### Self-Play (Future)

Current implementation: Both agents use same policy (pure self-play).

**Planned:**
- Policy pool: Keep last N checkpoints as opponents
- League training: Multiple agents with different strategies
- Opponent sampling: Mix of current, past, and random opponents

### Evaluation (Next Steps)

After training, evaluate against baselines:
1. Random agent (should beat >90%)
2. Heuristic agent (capture regions, efficient attacks)
3. Previous checkpoints (measure improvement)

See Phase 1.3 in `RL_TRAINING_ROADMAP.md`

### Hyperparameter Tuning

Use Ray Tune for automated search:
```python
# Future: parallel_risk/training/tune_hyperparams.py
from ray import tune

config = {
    "lr": tune.grid_search([1e-4, 3e-4, 1e-3]),
    "entropy_coeff": tune.grid_search([0.01, 0.05, 0.1]),
}
```

### Multi-Map Training (Phase 2)

Current: Train on single map size.

**Phase 2:** Train on multiple maps simultaneously with GNN architecture.

## Files Reference

```
parallel_risk/training/
├── __init__.py                    # Package init
├── rllib_wrapper.py               # Environment wrapper (250 lines)
├── train_rllib.py                 # Training script (300 lines)
└── configs/
    └── ppo_baseline.yaml          # Default configuration

tests/
└── test_rllib_wrapper.py          # Wrapper tests (150 lines)

install_training_deps.sh           # Installation script
```

## Next Steps

After successful training:

1. **Evaluation harness** (Phase 1.3)
   - Tournament system
   - Metrics: win rate, Elo, action analysis
   - Visualization

2. **Baseline experiments** (Phase 1.4)
   - Reward shaping ablation
   - Hyperparameter search
   - Architecture comparison

3. **Graph neural networks** (Phase 2)
   - PyTorch Geometric integration
   - Multi-map training
   - Transfer learning

See `docs/RL_TRAINING_ROADMAP.md` for full plan.

## Resources

**RLlib Documentation:**
- Quickstart: https://docs.ray.io/en/latest/rllib/index.html
- PPO: https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo
- Multi-agent: https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent

**Examples:**
- PettingZoo + RLlib: https://github.com/Farama-Foundation/PettingZoo/tree/master/tutorials/Ray
- Self-play: https://docs.ray.io/en/latest/rllib/rllib-examples.html#self-play

**Papers:**
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- Self-play: "Emergent Complexity via Multi-Agent Competition" (OpenAI, 2017)

## Changelog

**2026-04-07:** Initial RLlib integration
- Environment wrapper with action space simplification
- Training script with PPO
- Configuration system
- Test suite
- Documentation
