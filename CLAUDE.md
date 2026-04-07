# Claude Agent Guide: Parallel Risk Gym

This file provides context for Claude Code agents working on this project.

## Project Overview

**Parallel Risk** is a two-player simultaneous-turn strategy game built as a PettingZoo-compatible multi-agent RL environment. Players control territories on a map, deploy troops, transfer forces, and attack opponents. Actions from both players are collected each turn, shuffled randomly, and resolved sequentially.

**Purpose:** Training multi-agent reinforcement learning algorithms in a competitive territorial conquest game.

## Key Architecture Decisions

### 1. Action Space: Variable-Length with Fixed Size

We use a Dict space with explicit length indicator:
```python
{
    'num_actions': Discrete(11),           # 0 to 10 actions
    'actions': Box(shape=(10, 3), dtype=int32)  # Padded array
}
```

**Why:** RL algorithms need fixed-size tensors. Only first `num_actions` rows are processed, rest are padding.

**Action format:** `[source_territory, dest_territory, num_troops]`

### 2. Observation Space: Agent-Relative

Territory ownership is encoded as `1=self, -1=enemy` from each agent's perspective.

**Why:** Symmetric observations allow single policy to play both sides (critical for self-play training).

### 3. Combat: Deterministic Percentage-Based

- Defender casualties: 60% of attacking troops
- Attacker casualties: 70% of defending troops
- Attacker needs ~1.67× defender force to reliably capture

**Why deterministic:** Predictable outcomes, easier to learn optimal strategies, reduces variance in training.

### 4. Action Resolution: Random Shuffle

All actions from both players are collected, shuffled, then processed sequentially.

**Why:** Simple approach that creates strategic uncertainty. Alternative (weighted shuffle by troop counts) was rejected as too complex for initial version.

### 5. Modular Structure

Following PettingZoo conventions:
```
parallel_risk/
├── __init__.py               # Exports ParallelRiskEnv
├── parallel_risk_v0.py       # Entry point (PettingZoo convention)
└── env/
    ├── parallel_risk_env.py  # Core environment (330 lines)
    ├── map_config.py         # Map definitions (92 lines)
    ├── combat.py             # Combat resolver (37 lines)
    ├── validators.py         # Action validation (100 lines)
    └── reward_shaping.py     # RL reward shaping (320 lines)
```

**Why:** Extracted map definitions, combat logic, and validation into separate modules to make extensions easier without touching core environment.

### 6. Reward Shaping (Optional)

For RL training, dense reward signals are available via `reward_shaping_config`:

```python
from parallel_risk.env.reward_shaping import create_dense_config

env = ParallelRiskEnv(reward_shaping_config=create_dense_config())
```

Four reward components can be enabled independently:
- **Territory control:** Reward for % of map controlled
- **Region completion:** One-time bonus when completing regions
- **Troop advantage:** Reward for troop count ratio over opponent
- **Strategic position:** Reward for controlling well-connected territories

**Why optional:** Some RL algorithms handle sparse rewards well. Shaped rewards can accelerate learning but must be tuned carefully to avoid perverse incentives. All shaped rewards are scaled << 1.0 to keep terminal win/loss rewards dominant.

## Project Structure

- **parallel_risk/** - Main package
  - **env/** - Environment components (core, maps, combat, validation, reward shaping)
  - **training/** - RLlib training infrastructure (wrapper, training script, configs)
- **tests/** - Test suite (mechanics, combat, regions, run, reward_shaping, rllib_wrapper)
- **examples/** - Usage examples (reward_shaping_demo.py)
- **docs/** - Design documentation
  - DESIGN_NOTES.md - Deep dive into design decisions
  - COMBAT_SYSTEM.md - Complete combat mechanics
  - REWARD_SHAPING.md - RL reward shaping guide
  - RLLIB_INTEGRATION.md - RLlib training guide
  - RL_TRAINING_ROADMAP.md - Two-phase RL training plan
- **requirements.txt** - Dependencies (includes RLlib/Ray for training)
- **install_training_deps.sh** - Install training dependencies
- **run_tests.py** - Convenience script to run all tests

## Running Tests

```bash
# Run all tests
python run_tests.py

# Or individual tests
PYTHONPATH=. python tests/test_mechanics.py
PYTHONPATH=. python tests/test_reward_shaping.py

# Test RLlib wrapper (requires Ray installed)
PYTHONPATH=. python tests/test_rllib_wrapper.py
```

## Training RL Agents

### Installation

Install training dependencies (Ray/RLlib, PyTorch, TensorBoard):
```bash
./install_training_deps.sh
# or: pip install -r requirements.txt
```

### Quick Start

```bash
# Test training (10 iterations, ~5 minutes)
python -m parallel_risk.training.train_rllib \
    --config parallel_risk/training/configs/ppo_baseline.yaml \
    --num-iterations 10 \
    --num-workers 2

# Full training run
python -m parallel_risk.training.train_rllib \
    --config parallel_risk/training/configs/ppo_baseline.yaml
```

### Configuration

Edit `parallel_risk/training/configs/ppo_baseline.yaml` to customize:
- Environment (map, action budget, reward shaping)
- PPO hyperparameters (learning rate, clip param, etc.)
- Training settings (workers, batch size, GPUs)
- Network architecture

See `docs/RLLIB_INTEGRATION.md` for complete guide.

## Adding New Features

### Adding a New Map

Edit `parallel_risk/env/map_config.py` only:

```python
def create_large_10_map():
    adjacency_list = {...}
    regions = {...}
    region_bonuses = {...}
    # Build adjacency matrix, initial ownership

    return MapConfig(
        n_territories=10,
        adjacency_list=adjacency_list,
        adjacency_matrix=adjacency_matrix,
        initial_ownership=initial_ownership,
        regions=regions,
        region_bonuses=region_bonuses,
    )

MapRegistry.register("large_10", create_large_10_map)
```

Then use: `env = ParallelRiskEnv(map_name="large_10")`

### Modifying Combat Rules

Edit `parallel_risk/env/combat.py` only. The CombatResolver is isolated and independently testable.

### Adding/Changing Validation Rules

Edit `parallel_risk/env/validators.py`. All validation logic is centralized in the ActionValidator class.

### Using Reward Shaping for RL Training

See `docs/REWARD_SHAPING.md` for complete guide. Quick start:

```python
from parallel_risk.env.reward_shaping import create_dense_config

# Enable all reward components with default weights
env = ParallelRiskEnv(reward_shaping_config=create_dense_config())

# Or customize
from parallel_risk.env.reward_shaping import RewardShapingConfig

config = RewardShapingConfig(
    enable_territory_control=True,
    enable_region_completion=True,
    territory_control_weight=0.02,
    region_completion_weight=0.15,
)
env = ParallelRiskEnv(reward_shaping_config=config)
```

Reward component details available in `infos` for debugging:
```python
obs, rewards, terms, truncs, infos = env.step(actions)
print(infos['agent_0']['reward_components'])
# {'territory_control': 0.005, 'region_completion': 0.0, ...}
```

## Common Gotchas

1. **MapConfig is a dataclass, not a dict** - Use `map_config.n_territories`, not `map_config['n_territories']`

2. **Actions are agent-relative** - When processing actions, convert agent names to indices to check game state

3. **Income must be deployed in the same turn** - It doesn't accumulate between turns

4. **Action validation happens post-submission** - Invalid actions are counted but skipped during execution

5. **Tests need PYTHONPATH** - Use `run_tests.py` or set `PYTHONPATH=.` manually

## Documentation

- **docs/DESIGN_NOTES.md** - Deep dive into design decisions, alternative approaches considered, 10+ extension possibilities with code examples
- **docs/COMBAT_SYSTEM.md** - Complete combat mechanics with mathematical analysis
- **docs/REWARD_SHAPING.md** - RL reward shaping guide with component details, tuning guidelines, and validation checklist
- **docs/RLLIB_INTEGRATION.md** - Complete guide to training with RLlib: installation, configuration, troubleshooting
- **docs/RL_TRAINING_ROADMAP.md** - Two-phase plan for RL training: Phase 1 (RLlib baseline) complete, Phase 2 (GNN) planned
- **REFACTORING_SUMMARY.md** - Detailed refactoring history (monolithic → modular structure)
- **REWARD_SHAPING_SUMMARY.md** - Implementation summary for reward shaping


## Code Style Preferences

- Keep code pragmatic and focused - don't over-engineer
- No unnecessary abstractions for single-use code
- Clear variable names over terse ones
- Docstrings for public methods, inline comments only where logic isn't obvious
- Test after changes, verify all tests pass
