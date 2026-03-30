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
    ├── parallel_risk_env.py  # Core environment (317 lines)
    ├── map_config.py         # Map definitions (92 lines)
    ├── combat.py             # Combat resolver (37 lines)
    └── validators.py         # Action validation (100 lines)
```

**Why:** Extracted map definitions, combat logic, and validation into separate modules to make extensions easier without touching core environment.

## Project Structure

- **parallel_risk/** - Main package
- **tests/** - Test suite (mechanics, combat, regions, run)
- **docs/** - Design documentation (DESIGN_NOTES.md, COMBAT_SYSTEM.md)
- **requirements.txt** - Pinned dependencies (PettingZoo 1.25.0, Gymnasium 1.2.3, NumPy 2.3.5)
- **run_tests.py** - Convenience script to run all tests

## Running Tests

```bash
# Run all tests
python run_tests.py

# Or individual tests
PYTHONPATH=. python tests/test_mechanics.py
```

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

## Common Gotchas

1. **MapConfig is a dataclass, not a dict** - Use `map_config.n_territories`, not `map_config['n_territories']`

2. **Actions are agent-relative** - When processing actions, convert agent names to indices to check game state

3. **Income must be deployed in the same turn** - It doesn't accumulate between turns

4. **Action validation happens post-submission** - Invalid actions are counted but skipped during execution

5. **Tests need PYTHONPATH** - Use `run_tests.py` or set `PYTHONPATH=.` manually

## Documentation

- **docs/DESIGN_NOTES.md** - Deep dive into design decisions, alternative approaches considered, 10+ extension possibilities with code examples
- **docs/COMBAT_SYSTEM.md** - Complete combat mechanics with mathematical analysis
- **REFACTORING_SUMMARY.md** - Detailed refactoring history (monolithic → modular structure)


## Code Style Preferences

- Keep code pragmatic and focused - don't over-engineer
- No unnecessary abstractions for single-use code
- Clear variable names over terse ones
- Docstrings for public methods, inline comments only where logic isn't obvious
- Test after changes, verify all tests pass
