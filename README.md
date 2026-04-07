# Parallel Risk Gym

A PettingZoo-compatible multi-agent reinforcement learning environment for training agents in a Risk-like territorial conquest game with simultaneous turn resolution.

## Overview

**Parallel Risk** is a two-player strategy game designed specifically for training multi-agent RL algorithms. Both players submit actions simultaneously each turn, which are shuffled and resolved sequentially, creating strategic uncertainty and emergent gameplay.

**Key Features:**
- PettingZoo Parallel API compatible
- Ready for RLlib/PPO training with self-play
- Optional reward shaping to accelerate learning
- Deterministic combat mechanics (no dice)
- Agent-relative observations for symmetric self-play
- Customizable maps and game parameters

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# For RL training (includes Ray/RLlib, PyTorch, TensorBoard)
./install_training_deps.sh
```

## Training RL Agents

The primary use case: train competitive multi-agent RL policies.

### Validate Learning

First, verify agents can learn by running the validation experiment:

```bash
# Quick test (10 iterations, ~10 minutes)
python experiments/validate_learning.py \
    --num-iterations 10 \
    --eval-interval 5 \
    --verbose

# Full validation (500 iterations, ~2-3 hours)
python experiments/validate_learning.py \
    --num-iterations 500 \
    --eval-interval 50
```

**Success criteria:** Trained agent achieves >70% win rate vs. random opponent.

See [docs/VALIDATION_EXPERIMENT.md](docs/VALIDATION_EXPERIMENT.md) for complete guide.

### Quick Start

```bash
# Quick test (10 iterations, ~5 minutes)
python -m parallel_risk.training.train_rllib \
    --config parallel_risk/training/configs/ppo_baseline.yaml \
    --num-iterations 10 \
    --num-workers 2

# Full training run
python -m parallel_risk.training.train_rllib \
    --config parallel_risk/training/configs/ppo_baseline.yaml
```

### Training Features

**RLlib Integration:**
- Multi-agent PPO with self-play
- Configurable policy pool and opponent sampling
- Automatic checkpointing and TensorBoard logging
- GPU support

**Reward Shaping (Optional):**

Accelerate learning with dense reward signals:

```python
from parallel_risk import ParallelRiskEnv
from parallel_risk.env.reward_shaping import create_dense_config

# Enable all reward components
env = ParallelRiskEnv(reward_shaping_config=create_dense_config())

# Or use sparse rewards only (default)
env = ParallelRiskEnv()
```

Four configurable components:
- **Territory control** - Reward for map control percentage
- **Region completion** - Bonuses for completing regions
- **Troop advantage** - Reward for troop count superiority
- **Strategic position** - Reward for controlling key territories

**Configuration:**

Edit `parallel_risk/training/configs/ppo_baseline.yaml` to customize:
- Environment settings (map, action budget, reward shaping)
- PPO hyperparameters (learning rate, clip, entropy)
- Training settings (workers, batch size, GPUs)
- Network architecture

**Training Guides:**
- [docs/VALIDATION_EXPERIMENT.md](docs/VALIDATION_EXPERIMENT.md) - Validate that agents can learn
- [docs/RLLIB_INTEGRATION.md](docs/RLLIB_INTEGRATION.md) - Complete training setup and troubleshooting
- [docs/REWARD_SHAPING.md](docs/REWARD_SHAPING.md) - Reward component details and tuning
- [docs/RL_TRAINING_ROADMAP.md](docs/RL_TRAINING_ROADMAP.md) - Project roadmap and next steps

## Basic Usage

### Simple Example

```python
from parallel_risk import ParallelRiskEnv
import numpy as np

# Create environment
env = ParallelRiskEnv()
observations, infos = env.reset()

# Game loop
while env.agents:
    actions = {}
    for agent in env.agents:
        # Deploy 3 troops to territory 0
        actions[agent] = {
            'num_actions': 1,
            'actions': np.array([[0, 0, 3]] + [[0, 0, 0]] * 9, dtype=np.int32)
        }
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    if any(terminations.values()):
        print(f"Game over! Rewards: {rewards}")
        break
```

### Run Tests

```bash
# All tests
python run_tests.py

# Individual test suites
PYTHONPATH=. python tests/test_mechanics.py
PYTHONPATH=. python tests/test_rllib_wrapper.py  # Requires Ray/RLlib
```

## Game Mechanics

### Objective

Control all territories to win, or have the most territories when the turn limit (100 turns) is reached.

### Map

Default map (`simple_6`) - 6 territories in a 2×3 grid:

```
0 - 1 - 2  (North Region)
|   |   |
3 - 4 - 5  (South Region)
```

**Regions provide income bonuses:**
- North [0, 1, 2]: +4 troops/turn
- South [3, 4, 5]: +4 troops/turn
- Center [1, 4]: +2 troops/turn

### Actions

Each action is a triple: `[source, destination, num_troops]`

**Three action types:**
1. **Deploy** `[x, x, troops]` - Place income troops on owned territory
2. **Transfer** `[x, y, troops]` - Move troops between owned adjacent territories
3. **Attack** `[x, y, troops]` - Attack adjacent enemy territory

**Constraints:**
- Max 10 actions per turn per player
- Must leave ≥1 troop on source territory
- Income must be deployed in the same turn (doesn't accumulate)

### Combat

Deterministic percentage-based system:
- Defender loses: 60% of attacking troops (rounded down)
- Attacker loses: 70% of defending troops (rounded down)
- Attacker captures if defenders reduced to ≤0

**Example:** 10 attackers vs 6 defenders
- Defender: 6 - (10 × 0.6) = 0 (eliminated)
- Attacker: 10 - (6 × 0.7) = 6 survivors
- Result: Attacker captures with 6 troops

**Strategic note:** Need ~1.67× defender force to reliably capture.

### Turn Flow

1. **Income Phase** - Calculate income (base 5 + region bonuses)
2. **Action Submission** - Both players submit 0-10 actions simultaneously
3. **Resolution** - All actions shuffled and executed sequentially
4. **Victory Check** - Check win conditions

### Victory Conditions

- **Conquest:** Control all territories (+1.0 / -1.0 reward)
- **Elimination:** Opponent has 0 territories (+1.0 / -1.0)
- **Turn Limit:** Most territories at turn 100 (+0.5 / -0.5)

## API Reference

### Observation Space

Agent-relative `Dict` observations:

```python
{
    'territory_ownership': Box(shape=(6,), dtype=int8),    # 1=self, -1=enemy
    'territory_troops': Box(shape=(6,), dtype=int32),      # Troop counts
    'adjacency_matrix': Box(shape=(6,6), dtype=int8),      # Map structure
    'available_income': Box(shape=(1,), dtype=int32),      # Deployable troops
    'turn_number': Box(shape=(1,), dtype=int32),           # Current turn
    'region_control': Box(shape=(3,), dtype=int8),         # Binary region control
}
```

### Action Space

```python
{
    'num_actions': Discrete(11),                   # 0 to 10 actions
    'actions': Box(shape=(10, 3), dtype=int32),    # [source, dest, troops]
}
```

### Environment Configuration

```python
env = ParallelRiskEnv(
    map_name="simple_6",                   # Map to use
    max_actions_per_turn=10,               # Action budget
    income_per_turn=5,                     # Base income
    max_turns=100,                         # Turn limit
    initial_troops_per_territory=3,        # Starting troops
    seed=None,                             # Random seed
    reward_shaping_config=None,            # Optional reward shaping
)
```

**Reward shaping presets:**
- `create_sparse_config()` - No shaping (default)
- `create_dense_config()` - All components enabled
- `create_territorial_config()` - Territory + region rewards
- `create_aggressive_config()` - Troop + strategic rewards

### Info Dict

```python
infos[agent] = {
    'invalid_actions': int,                    # Count of invalid actions
    'controlled_regions': list[str],           # Controlled region names
    'income': int,                             # Next turn income
    'reward_components': dict,                 # Breakdown (if shaping enabled)
}
```

## Advanced Features

### Custom Maps

Add new maps in `parallel_risk/env/map_config.py`:

```python
def create_custom_8_map():
    adjacency_list = {...}
    regions = {...}
    region_bonuses = {...}
    # Build adjacency matrix, initial ownership
    
    return MapConfig(
        n_territories=8,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )

MapRegistry.register("custom_8", create_custom_8_map)
```

### Action Masking

For improved training efficiency, implement action masking using the observation space. See [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) for implementation examples.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project guide for Claude Code agents
- [docs/VALIDATION_EXPERIMENT.md](docs/VALIDATION_EXPERIMENT.md) - Learning validation experiment
- [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) - Design decisions and extension ideas
- [docs/COMBAT_SYSTEM.md](docs/COMBAT_SYSTEM.md) - Combat mechanics deep dive
- [docs/REWARD_SHAPING.md](docs/REWARD_SHAPING.md) - Reward shaping guide
- [docs/RLLIB_INTEGRATION.md](docs/RLLIB_INTEGRATION.md) - RLlib training guide
- [docs/RL_TRAINING_ROADMAP.md](docs/RL_TRAINING_ROADMAP.md) - Training roadmap

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with [PettingZoo](https://pettingzoo.farama.org/) and [Gymnasium](https://gymnasium.farama.org/).
