# Parallel Risk Gym

A PettingZoo-compatible multi-agent reinforcement learning environment implementing a Risk-like strategy game with parallel (simultaneous) turn resolution.

## Overview

**Parallel Risk** is a two-player territorial conquest game where both players submit their actions simultaneously each turn. Unlike traditional turn-based games, all actions are collected, shuffled randomly, and resolved sequentially.

The environment is designed for training multi-agent reinforcement learning algorithms and follows the [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/).

## Game Rules

### Objective

Control all territories to win, or have the most territories when the turn limit is reached.

### Map Structure

The default map (`simple_6`) consists of 6 territories arranged in a grid:

```
0 - 1 - 2  (North Region)
|   |   |
3 - 4 - 5  (South Region)
```

**Regions:**
- **North Region** [0, 1, 2]: +4 bonus troops/turn when fully controlled
- **South Region** [3, 4, 5]: +4 bonus troops/turn when fully controlled
- **Center Region** [1, 4]: +2 bonus troops/turn when fully controlled

**Initial Setup:**
- Agent 0 starts with territories [0, 1, 5]
- Agent 1 starts with territories [2, 3, 4]
- Each territory starts with 3 troops

### Turn Structure

Each turn follows this sequence:

1. **Income Phase**: Each player receives income troops based on:
   - Base income: 5 troops/turn
   - Region bonuses: Additional troops for controlling complete regions

2. **Action Submission**: Both players simultaneously submit 0-10 actions

3. **Action Resolution**:
   - All actions from both players are collected
   - Actions are shuffled randomly
   - Actions are processed sequentially
   - Invalid actions are skipped

4. **Victory Check**: Game ends if one player controls all territories or turn limit reached

### Action Types

Each action is a triple: `(source_territory, dest_territory, num_troops)`

The action type is determined by the territories involved:

#### 1. Deploy `(x, x, troops)`
- **When**: source == destination
- **Effect**: Place troops from income onto owned territory
- **Validation**: Must own territory, troops ≤ available income

#### 2. Transfer `(x, y, troops)`
- **When**: source ≠ destination, both owned by player
- **Effect**: Move troops between owned territories
- **Validation**: Territories adjacent, troops < source troops (must leave ≥1)

#### 3. Attack `(x, y, troops)`
- **When**: source owned by player, destination owned by opponent
- **Effect**: Attack enemy territory
- **Validation**: Territories adjacent, troops < source troops (must leave ≥1)

### Combat Resolution

Combat is **deterministic** and percentage-based:

Given `x` attacking troops and `y` defending troops:
- **Defender casualties** = 60% of x (rounded down)
- **Attacker casualties** = 70% of y (rounded down)

**Outcome:**
- If defenders_remaining ≤ 0: **Attacker captures territory**
  - Surviving attackers occupy the territory
- Otherwise: **Defender holds**
  - Defenders reduced but retain control
  - Attackers lose their troops

**Example:** 10 attackers vs 6 defenders
- Defender loses: 10 × 0.6 = 6 (eliminated)
- Attacker loses: 6 × 0.7 = 4 (rounded down)
- Result: Attacker wins with 6 surviving troops

**Strategic Note:** Attackers need ~1.67× the defending force to reliably capture territory.

### Victory Conditions

The game ends when:
1. **Conquest**: One player owns all territories (+1.0 reward for winner, -1.0 for loser)
2. **Elimination**: One player has 0 territories (+1.0 for survivor, -1.0 for eliminated)
3. **Turn Limit**: 100 turns reached (+0.5 for territory leader, -0.5 for other)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd parallel-risk-gym

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from parallel_risk import ParallelRiskEnv
import numpy as np

# Create environment
env = ParallelRiskEnv()

# Reset environment
observations, infos = env.reset()

# Game loop
while env.agents:
    # Generate actions for each agent
    actions = {}
    for agent in env.agents:
        # Example: Deploy 3 troops to territory 0
        actions[agent] = {
            'num_actions': 1,
            'actions': np.array([
                [0, 0, 3],  # Deploy 3 troops to territory 0
                [0, 0, 0],  # Padding (unused)
                # ... more padding to reach max_actions_per_turn
            ], dtype=np.int32)
        }

    # Step environment
    observations, rewards, terminations, truncations, infos = env.step(actions)

    # Render current state
    env.render()

    if any(terminations.values()):
        print(f"Game over! Rewards: {rewards}")
        break
```

### Random Policy Example

See `test_run.py` for a complete example with random policies:

```bash
python tests/test_run.py
```

## Observation Space

Each agent receives a `Dict` observation with:

```python
{
    'territory_ownership': Box(shape=(6,), dtype=int8),      # -1=enemy, 1=self
    'territory_troops': Box(shape=(6,), dtype=int32),        # Troop counts
    'adjacency_matrix': Box(shape=(6,6), dtype=int8),        # Map structure
    'available_income': Box(shape=(1,), dtype=int32),        # Deployable troops
    'turn_number': Box(shape=(1,), dtype=int32),             # Current turn
    'region_control': Box(shape=(3,), dtype=int8),           # Controlled regions
}
```

**Notes:**
- Observations are **agent-relative** (ownership from agent's perspective)
- All information is **fully observable** (no fog of war)
- Adjacency matrix is static but included for network convenience

## Action Space

Each agent submits a `Dict` action:

```python
{
    'num_actions': Discrete(11),                    # 0 to 10 actions
    'actions': Box(shape=(10, 3), dtype=int32),     # Action triples
}
```

Only the first `num_actions` rows of the `actions` array are processed.

**Action Triple Format:** `[source_territory, dest_territory, num_troops]`

## Environment Configuration

```python
env = ParallelRiskEnv(
    map_name="simple_6",              # Map configuration
    max_actions_per_turn=10,          # Max actions per player per turn
    income_per_turn=5,                # Base income
    max_turns=100,                    # Turn limit
    initial_troops_per_territory=3,   # Starting troops
    seed=None,                        # Random seed
)
```

## Testing

Run the test suites to verify functionality:

```bash
# Run all tests
python run_tests.py

# Or run individual tests with PYTHONPATH
PYTHONPATH=. python tests/test_mechanics.py
PYTHONPATH=. python tests/test_combat.py
PYTHONPATH=. python tests/test_regions.py
PYTHONPATH=. python tests/test_run.py
```

## Strategic Considerations

### Region Control
- Completing regions provides significant income advantage
- Breaking opponent's regions is often easier than capturing them
- Economic advantage compounds over time

### Combat Efficiency
- Small attacks usually fail (10 vs 10 → defender holds)
- Overwhelming force is efficient (20 vs 10 → keep 13 troops)
- Defenders have advantage (~17% fewer casualties)

### Action Order
- Actions are resolved randomly
- Can't predict exact order
- Multiple attacks on same territory possible
- Last successful attack determines ownership

### Income Management
- Base income: 5 troops/turn
- Region bonuses stack (can reach 11 troops/turn)
- Must deploy during same turn (doesn't accumulate)

## Advanced Features

### Custom Maps

Add new maps by extending `_initialize_map()`:

```python
def _initialize_map(self):
    if self.map_name == "custom_8":
        adjacency_list = {...}
        regions = {...}
        region_bonuses = {...}
        # ... return map_config
```

### Action Masking

The `infos` dict includes useful debugging information:

```python
infos[agent] = {
    'invalid_actions': int,           # Count of invalid actions
    'controlled_regions': list[str],  # Region names controlled
    'income': int,                    # Next turn's income
}
```

For better RL training, implement action masking using the observation space (see `docs/DESIGN_NOTES.md` for examples).

## Documentation

- **docs/DESIGN_NOTES.md** - Detailed design decisions, alternative approaches, and extension possibilities
- **docs/COMBAT_SYSTEM.md** - Complete combat mechanics documentation with examples

## Training RL Agents

The environment follows the PettingZoo Parallel API standard, which is designed to work with multi-agent RL frameworks.

**Note:** Integration with specific RL frameworks (Stable-Baselines3, RLlib, etc.) has not yet been tested. See PettingZoo's [training documentation](https://pettingzoo.farama.org/tutorials/sb3/) for guidance on integrating PettingZoo environments with your framework of choice.


## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with [PettingZoo](https://pettingzoo.farama.org/) and [Gymnasium](https://gymnasium.farama.org/).
