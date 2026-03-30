# Parallel Risk Environment - Design Notes

This document contains research findings and design decisions for modeling the Parallel Risk game in PettingZoo/Gymnasium. Use this as reference for future extensions or alternative modeling approaches.

---

## Table of Contents
1. [Game Requirements](#game-requirements)
2. [Action Space Design](#action-space-design)
3. [Observation Space Design](#observation-space-design)
4. [Alternative Approaches](#alternative-approaches)
5. [Extension Possibilities](#extension-possibilities)

---

## Game Requirements

### Core Mechanics
- **Players**: 2 players taking parallel/simultaneous turns
- **Map**: Graph of n territories (nodes) with adjacency edges
- **Actions per turn**: Variable number of action triples per player
- **Action format**: `(source_territory, dest_territory, num_troops)`
- **Action types**:
  - **Deploy** `(x, x, troops)`: Place income troops on owned territory
  - **Transfer** `(x, y, troops)`: Move troops between owned territories
  - **Attack** `(x, y, troops)`: Attack enemy-owned territory
- **Income**: Fixed per turn (e.g., 5 troops), can be split across deployments
- **Action resolution**: Shuffle all actions randomly, process sequentially, discard invalid ones
- **Combat**: Deterministic ratio-based (defender gets 1.5x multiplier)
- **Victory**: Last player with territories wins

### Key Challenges
1. Variable-length action lists per turn
2. Complex action validation (ownership, adjacency, resources)
3. Simultaneous action submission with conflict resolution
4. Rich observation space (map topology, ownership, troop counts)

---

## Action Space Design

### Research Summary: Variable-Length Actions in RL

There are three main approaches for handling variable-length action sequences:

### **Approach A: Dict Space with Length Indicator** ✅ CHOSEN

**Structure:**
```python
spaces.Dict({
    'num_actions': spaces.Discrete(max_actions + 1),  # 0 to max_actions
    'actions': spaces.Box(
        low=0,
        high=n_territories,
        shape=(max_actions, 3),
        dtype=np.int32
    )
})
```

**Pros:**
- ✅ Fixed-size space (RL-friendly)
- ✅ Works with all major RL frameworks (Stable-Baselines3, RLlib, CleanRL)
- ✅ Explicit length helps neural networks learn
- ✅ Simple to implement and debug
- ✅ ~124 bytes per action (reasonable memory)
- ✅ Universal framework compatibility

**Cons:**
- ❌ Wastes space when few actions taken
- ❌ Padding in unused action slots

**Why Chosen:**
This is the industry-standard approach for variable-length actions in RL. It provides the best trade-off between flexibility and compatibility.

---

### **Approach B: Sequence Space** (Alternative)

**Structure:**
```python
spaces.Sequence(
    spaces.Box(low=0, high=n_territories, shape=(3,), dtype=np.int32)
)
```

**Pros:**
- ✅ True variable-length (no padding waste)
- ✅ Clean semantics
- ✅ Built-in masking support: `sample(mask=(length_mask, action_mask))`
- ✅ Memory efficient for sparse actions

**Cons:**
- ❌ Limited RL framework support
- ❌ Complex batching for parallel environments
- ❌ Harder to integrate with standard neural network architectures
- ❌ Medium-high implementation complexity

**When to Use:**
- When memory is critical concern
- When actions are very sparse (typically 0-2 actions per turn)
- When using custom RL implementations

---

### **Approach C: Fixed-Size Box with Padding** (Simplest)

**Structure:**
```python
spaces.Box(
    low=0,
    high=n_territories,
    shape=(max_actions, 3),
    dtype=np.int32
)
# Requires external convention: padding with zeros or sentinel values
```

**Pros:**
- ✅ Simplest implementation
- ✅ Universal compatibility
- ✅ Efficient batching

**Cons:**
- ❌ No explicit length indicator (network must learn padding)
- ❌ Wasteful for sparse actions
- ❌ Requires external validation logic
- ❌ Ambiguous semantics (is [0,0,0] a valid action or padding?)

**When to Use:**
- Quick prototypes only
- When max_actions is very small (≤3)

---

### **Approach D: Sequential Single-Action Pattern** (Turn-Based)

**Structure:**
```python
spaces.Discrete(n_territories * n_territories * max_troops)
# Or use MultiDiscrete([n_territories, n_territories, max_troops])
```

**Pros:**
- ✅ Maximum control over action masking
- ✅ Simplest action space (single action per step)
- ✅ Best for action masking frameworks
- ✅ Easy for neural networks

**Cons:**
- ❌ Requires fundamental environment restructure
- ❌ Many micro-steps per game turn
- ❌ Doesn't match parallel game semantics
- ❌ More complex state management

**When to Use:**
- When action masking is critical
- When training with frameworks that struggle with Dict spaces
- When each action has complex dependencies

---

## Action Space Trade-Off Matrix

| Factor | Dict + Length | Sequence | Fixed Box | Sequential |
|--------|--------------|----------|-----------|------------|
| **Framework Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Memory Efficiency** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Semantic Clarity** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Implementation Ease** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **RL Training** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Batching** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Action Masking** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Action Validation Strategies

### **Strategy 1: Post-hoc Validation** ✅ CHOSEN

**Implementation:**
- Validate each action in `step()` against current game state
- Silently discard invalid actions
- Track count in `info` dict for debugging

**Pros:**
- ✅ Works everywhere
- ✅ Simple to implement
- ✅ No framework dependencies

**Cons:**
- ❌ Agent wastes actions on invalid moves
- ❌ Slower learning (network doesn't get masking signal)

**Code Pattern:**
```python
for action in all_actions:
    if self._validate_action(action):
        self._execute_action(action)
    else:
        infos[action['agent']]['invalid_actions'] += 1
```

---

### **Strategy 2: Info Dict Masking** (Recommended for Training)

**Implementation:**
- Return `info['action_mask']` or `info['valid_actions']`
- RL algorithm samples only from legal subset
- Requires framework support (RLlib, SB3 with custom policy)

**Pros:**
- ✅ Better training efficiency
- ✅ Network learns constraints faster
- ✅ Fewer wasted actions

**Cons:**
- ❌ Requires framework integration
- ❌ More complex implementation

**Code Pattern:**
```python
def _get_action_mask(self, agent):
    """Generate boolean mask of valid actions"""
    mask = np.zeros((max_actions, n_territories, n_territories, max_troops), dtype=bool)
    for source in owned_territories:
        for dest in all_territories:
            if self._is_valid_pair(source, dest):
                max_troops_available = self.game_state['territory_troops'][source] - 1
                mask[:, source, dest, :max_troops_available] = True
    return mask

# In step() return:
infos[agent]['action_mask'] = self._get_action_mask(agent)
```

---

### **Strategy 3: Observation-Space Masking** (Advanced)

**Implementation:**
- Include action mask as part of observation Dict
- Network learns to read and apply mask
- Most flexible but highest complexity

**Pros:**
- ✅ Most flexible
- ✅ Network can reason about validity
- ✅ No framework requirements

**Cons:**
- ❌ Very complex
- ❌ Larger observation space
- ❌ Harder to train

---

## Observation Space Design

### Current Implementation: Full Information Dict

```python
spaces.Dict({
    'territory_ownership': spaces.Box(low=-1, high=1, shape=(n_territories,), dtype=np.int8),
        # -1 = enemy, 1 = self (agent-relative perspective)

    'territory_troops': spaces.Box(low=0, high=100, shape=(n_territories,), dtype=np.int32),
        # Troop counts for all territories (full visibility)

    'adjacency_matrix': spaces.Box(low=0, high=1, shape=(n_territories, n_territories), dtype=np.int8),
        # Static map structure, 1 if territories are adjacent

    'available_income': spaces.Box(low=0, high=income_per_turn, shape=(1,), dtype=np.int32),
        # Remaining income for deployment this turn

    'turn_number': spaces.Box(low=0, high=max_turns, shape=(1,), dtype=np.int32),
        # Current turn for temporal reasoning
})
```

**Design Rationale:**
- **Agent-relative ownership** (-1 vs 1): Makes observations symmetric for both players
- **Full troop visibility**: Simplifies learning, avoids fog-of-war complexity
- **Static adjacency matrix**: Included in obs for network convenience (could be external)
- **Available income**: Critical for deployment planning
- **Turn number**: Enables time-based strategy

---

### Alternative Observation Designs

#### **Option 1: Flattened Vector** (Simpler Networks)

```python
# Total size: n_territories * 2 + n_territories^2 + 2 = 6*2 + 36 + 2 = 50
spaces.Box(
    low=-100,
    high=100,
    shape=(n_territories * 2 + n_territories**2 + 2,),
    dtype=np.float32
)
# Layout: [ownership vector, troops vector, adjacency flattened, income, turn]
```

**Pros:**
- ✅ Simple MLPs can process directly
- ✅ Smaller observation object
- ✅ Easier batching

**Cons:**
- ❌ Less semantic structure
- ❌ Harder to debug
- ❌ Can't leverage CNN/GNN architectures

---

#### **Option 2: Graph-Based** (For GNNs)

```python
spaces.Dict({
    'nodes': spaces.Box(shape=(n_territories, node_features), ...),
        # node_features: [ownership, troops, income_deployment_this_turn]

    'edges': spaces.Box(shape=(n_edges, 2), dtype=np.int32),
        # Edge list representation

    'global': spaces.Box(shape=(global_features,), ...),
        # [available_income, turn_number]
})
```

**Pros:**
- ✅ Natural representation for graph problems
- ✅ Scales to different map sizes
- ✅ Can use Graph Neural Networks (GNNs)
- ✅ Inductive transfer to new maps

**Cons:**
- ❌ Requires GNN architectures
- ❌ More complex preprocessing
- ❌ Less support in standard RL libraries

**When to Use:**
- When training on multiple map topologies
- When map size varies significantly
- When using GNN-based policies

---

#### **Option 3: Image-Like Representation** (For CNNs)

```python
# For spatial 2D maps
spaces.Box(
    low=-100,
    high=100,
    shape=(height, width, channels),
    dtype=np.float32
)
# Channels: ownership, troops, adjacency (multiple channels), income map
```

**Pros:**
- ✅ Can use CNNs (Atari-style)
- ✅ Spatial relationships implicit
- ✅ Well-studied architectures

**Cons:**
- ❌ Only works for grid-like maps
- ❌ Wastes space for non-grid topologies
- ❌ Current map is not regular grid

---

#### **Option 4: Partial Observability / Fog of War**

```python
spaces.Dict({
    'territory_ownership': spaces.Box(low=-1, high=1, shape=(n_territories,), dtype=np.int8),
        # Known ownership (-1, 0, 1) where 0 = unknown

    'territory_troops': spaces.Box(low=-1, high=100, shape=(n_territories,), dtype=np.int32),
        # -1 = unknown, else troop count

    'visible_mask': spaces.Box(low=0, high=1, shape=(n_territories,), dtype=np.int8),
        # 1 if territory is adjacent to owned territory

    # ... rest same as full info
})
```

**Pros:**
- ✅ More realistic/challenging
- ✅ Encourages reconnaissance strategies
- ✅ Closer to real-world scenarios

**Cons:**
- ❌ Much harder to train
- ❌ Requires belief state reasoning
- ❌ Longer training time

---

## Observation Space Trade-Offs

| Factor | Dict (Current) | Flattened Vector | Graph-Based | Image-Like | Partial Obs |
|--------|----------------|------------------|-------------|------------|-------------|
| **Semantic Clarity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Network Compatibility** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Memory Efficiency** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Scalability** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Training Ease** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Debug-ability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## Alternative Approaches Not Chosen

### **Approach: Sequential Environment (AEC API)**

**Structure:**
Convert to turn-based where each agent submits one action at a time.

```python
# Use PettingZoo AEC API instead of Parallel
# Each step processes one action from one agent
# Agents alternate: agent_0 action_1, agent_1 action_1, agent_0 action_2, ...
```

**Pros:**
- ✅ Maximum control over action ordering
- ✅ Natural action masking support
- ✅ Simpler state management
- ✅ Can implement detailed turn phases (deploy phase, attack phase, etc.)

**Cons:**
- ❌ Doesn't match parallel game semantics
- ❌ More steps per game turn
- ❌ Loses simultaneous action aspect
- ❌ Agents can see each other's actions mid-turn

**When to Use:**
- If parallel semantics not important
- If you want turn-based like chess
- If action masking is critical

---

### **Approach: Hierarchical Actions**

**Structure:**
Break actions into two levels: high-level strategy, low-level execution.

```python
# Step 1: Select action type
action_type = spaces.Discrete(3)  # deploy, transfer, attack

# Step 2: Select parameters based on type
action_params = spaces.Dict({
    'deploy': spaces.MultiDiscrete([n_territories, max_troops]),
    'transfer': spaces.MultiDiscrete([n_territories, n_territories, max_troops]),
    'attack': spaces.MultiDiscrete([n_territories, n_territories, max_troops]),
})
```

**Pros:**
- ✅ Structured action space
- ✅ Can mask parameters based on type
- ✅ Easier for hierarchical RL

**Cons:**
- ❌ Requires multiple sub-steps per action
- ❌ Complex to implement
- ❌ Still need variable-length wrapper

---

### **Approach: Macro Actions**

**Structure:**
Define high-level strategic actions instead of low-level troop movements.

```python
spaces.Discrete(n_macro_actions)
# Examples: "fortify_border", "all_out_attack_territory_X", "balanced_defense"
```

**Pros:**
- ✅ Simpler action space
- ✅ Faster training
- ✅ More interpretable

**Cons:**
- ❌ Loss of fine-grained control
- ❌ Requires manual strategy definition
- ❌ Less flexible
- ❌ May not find optimal strategies

---

## Extension Possibilities

### **1. Enhanced Action Masking**

Add action masks to info dict for better training:

```python
def _get_valid_actions(self, agent):
    """Return list of all valid action triples"""
    valid = []
    agent_idx = self.possible_agents.index(agent)

    # Deploy actions
    for territory in np.where(self.game_state['territory_ownership'] == agent_idx)[0]:
        for troops in range(1, self.game_state['available_income'][agent] + 1):
            valid.append((territory, territory, troops))

    # Transfer/Attack actions
    for source in np.where(self.game_state['territory_ownership'] == agent_idx)[0]:
        available_troops = self.game_state['territory_troops'][source] - 1
        if available_troops <= 0:
            continue

        for dest in np.where(self.map_config['adjacency_matrix'][source] == 1)[0]:
            for troops in range(1, available_troops + 1):
                valid.append((source, dest, troops))

    return valid

# In step():
infos[agent]['valid_actions'] = self._get_valid_actions(agent)
```

**Use with:**
- RLlib: `"model": {"custom_action_dist": MaskedActionDistribution}`
- Stable-Baselines3: Custom policy with action masking

---

### **2. Variable Income Based on Territory Count**

```python
def _calculate_income(self, agent):
    """Income based on territories owned"""
    agent_idx = self.possible_agents.index(agent)
    territory_count = np.sum(self.game_state['territory_ownership'] == agent_idx)

    base_income = 3
    territory_bonus = territory_count // 2  # 1 troop per 2 territories

    return base_income + territory_bonus
```

**Impact on observation space:**
- Remove fixed `available_income` from init
- Calculate dynamically each turn

---

### **3. Territory Bonuses / Regions**

```python
# Add to map_config
'regions': {
    'north': [0, 1, 2],
    'south': [3, 4, 5],
},
'region_bonuses': {
    'north': 2,
    'south': 2,
}

def _calculate_income(self, agent):
    income = base_income

    # Check region control
    for region, territories in self.map_config['regions'].items():
        if all(self.game_state['territory_ownership'][t] == agent_idx
               for t in territories):
            income += self.map_config['region_bonuses'][region]

    return income
```

**Impact on observation space:**
- Add `region_control` indicator
- Or keep implicit (network learns from ownership)

---

### **4. Multiple Map Support**

```python
# Add map loader
def _initialize_map(self):
    if self.map_name == "simple_6":
        return self._load_simple_6()
    elif self.map_name == "classic_42":
        return self._load_from_json(f"maps/{self.map_name}.json")

def _load_from_json(self, path):
    with open(path) as f:
        config = json.load(f)
    # config format: {territories, adjacency, initial_ownership, regions, ...}
    return self._build_map_config(config)
```

**JSON Format:**
```json
{
  "n_territories": 6,
  "adjacency": [[0,1], [1,2], [0,3], [1,4], [2,5], [3,4], [4,5]],
  "initial_ownership": {
    "agent_0": [0, 1, 5],
    "agent_1": [2, 3, 4]
  },
  "initial_troops": 3
}
```

---

### **5. Probabilistic Combat**

Replace deterministic combat with dice-based:

```python
def _resolve_combat(self, attacking_troops, defending_troops):
    """Probabilistic combat with binomial distribution"""
    attacker_hit_rate = 0.6
    defender_hit_rate = 0.7

    attacker_hits = np.random.binomial(attacking_troops, attacker_hit_rate)
    defender_hits = np.random.binomial(defending_troops, defender_hit_rate)

    attacker_casualties = min(attacking_troops, defender_hits)
    defender_casualties = min(defending_troops, attacker_hits)

    attacker_remaining = attacking_troops - attacker_casualties
    defender_remaining = defending_troops - defender_casualties

    if attacker_remaining > defender_remaining:
        return 'attacker_wins', attacker_remaining
    else:
        return 'defender_holds', defender_remaining
```

**Considerations:**
- Makes training harder (stochastic outcomes)
- Requires more samples for convergence
- More realistic/engaging for humans

---

### **6. Reward Shaping**

Current: +1 for win, -1 for loss, 0 during game

Enhanced:
```python
def _calculate_rewards(self):
    """Shaped rewards for better training"""
    rewards = {agent: 0.0 for agent in self.possible_agents}

    # Territory control reward (small, frequent)
    for agent_idx, agent in enumerate(self.possible_agents):
        territory_count = np.sum(self.game_state['territory_ownership'] == agent_idx)
        territory_proportion = territory_count / self.map_config['n_territories']
        rewards[agent] += territory_proportion * 0.01  # Small reward

    # Troop count reward (military strength)
    for agent_idx, agent in enumerate(self.possible_agents):
        owned_territories = np.where(self.game_state['territory_ownership'] == agent_idx)[0]
        total_troops = np.sum(self.game_state['territory_troops'][owned_territories])
        rewards[agent] += total_troops * 0.001

    # Penalty for invalid actions
    for agent in self.possible_agents:
        if self.infos[agent]['invalid_actions'] > 0:
            rewards[agent] -= 0.05 * self.infos[agent]['invalid_actions']

    return rewards
```

**Trade-offs:**
- ✅ Faster learning
- ✅ More granular feedback
- ❌ Can bias strategy
- ❌ Requires tuning

---

### **7. Multi-Agent (3-4 Players)**

```python
def __init__(self, ..., n_players=2):
    self.possible_agents = [f"agent_{i}" for i in range(n_players)]
    # ... rest of init

# Update ownership to support n_players
# Update termination to handle multiple eliminations
# Update initial territory distribution
```

**Challenges:**
- More complex alliances/diplomacy
- Harder to train (more agents)
- Reward structure more complex

---

### **8. Render / Visualization**

```python
def render(self, mode='human'):
    if mode == 'human':
        self._render_ascii()
    elif mode == 'rgb_array':
        return self._render_matplotlib()

def _render_matplotlib(self):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from([(s, d) for s in self.map_config['adjacency_list']
                      for d in self.map_config['adjacency_list'][s]])

    node_colors = [['blue', 'red'][self.game_state['territory_ownership'][i]]
                   for i in range(self.map_config['n_territories'])]

    node_labels = {i: f"{i}\n({self.game_state['territory_troops'][i]})"
                   for i in range(self.map_config['n_territories'])}

    nx.draw(G, node_color=node_colors, labels=node_labels, with_labels=True)

    # Convert to RGB array
    fig = plt.gcf()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return data
```

---

### **9. Action History / Replay Buffer Integration**

Add to observation:
```python
'action_history': spaces.Box(
    low=-1,
    high=n_territories,
    shape=(history_length, 3),  # Last N actions
    dtype=np.int32
)
```

Helps network learn:
- Opponent patterns
- Action sequences
- Timing strategies

---

### **10. Curriculum Learning**

Progressive difficulty:
```python
def __init__(self, ..., difficulty='easy'):
    if difficulty == 'easy':
        self.income_per_turn = 10
        self.initial_troops = 5
        self.combat_advantage = 1.2  # Easier to attack
    elif difficulty == 'hard':
        self.income_per_turn = 3
        self.initial_troops = 2
        self.combat_advantage = 2.0  # Harder to attack
```

Training schedule:
1. Start with easy (learn basics)
2. Increase difficulty over time
3. Transfer policy to harder settings

---

## Common Pitfalls & Solutions

### **Pitfall 1: Not Validating at Every Step**

**Problem:** Validating only at init, not during game.

**Solution:**
```python
# BAD: Only validate format
assert action['num_actions'] <= self.max_actions

# GOOD: Validate against current game state
if not self._validate_action(action_info):
    continue  # Skip invalid action
```

---

### **Pitfall 2: Not Reshaping Flattened Actions**

**Problem:** Treating flat Box array as individual values.

**Solution:**
```python
# BAD: Using actions[i] directly
for i in range(num_actions * 3):
    value = action_dict['actions'][i]

# GOOD: Reshape to (num_actions, 3)
actions_reshaped = action_dict['actions'].reshape(max_actions, 3)
for i in range(num_actions):
    source, dest, troops = actions_reshaped[i]
```

---

### **Pitfall 3: Missing Edge Cases**

**Problem:** Not testing with 0 actions, max actions, or all invalid.

**Solution:**
```python
# Test cases:
# - Agent submits 0 actions (should be valid)
# - Agent submits max actions (should work)
# - All actions invalid (should skip all)
# - Simultaneous attacks on same territory
# - Deploying more than income
# - Attacking with all troops (should fail - must leave 1)
```

---

### **Pitfall 4: Silent Failures**

**Problem:** Invalid actions silently ignored during development.

**Solution:**
```python
# During development
if not self._validate_action(action_info):
    if os.getenv('DEBUG'):
        print(f"Invalid action: {action_info} - Reason: {self._get_invalid_reason(action_info)}")
    infos[agent]['invalid_actions'] += 1
```

---

### **Pitfall 5: Ignoring Batch Training**

**Problem:** Not testing with vectorized environments.

**Solution:**
```python
# Test with Stable-Baselines3 SubprocVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

envs = SubprocVecEnv([lambda: ParallelRiskEnv() for _ in range(4)])
# Ensure observations/actions batch correctly
```

---

## Recommended Reading

### PettingZoo
- API Documentation: https://pettingzoo.farama.org/api/
- Custom Environments: https://pettingzoo.farama.org/tutorials/custom_environment/
- Parallel API: https://pettingzoo.farama.org/api/parallel/
- AEC API: https://pettingzoo.farama.org/api/aec/

### Gymnasium
- Spaces: https://gymnasium.farama.org/api/spaces/
- Custom Environments: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

### Multi-Agent RL Frameworks
- RLlib: https://docs.ray.io/en/latest/rllib/index.html
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- PettingZoo + SB3: https://pettingzoo.farama.org/tutorials/sb3/

### Research Papers
- "Multi-Agent Reinforcement Learning: A Selective Overview" (2019)
- "The StarCraft Multi-Agent Challenge" (Samvelyan et al., 2019)
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Schrittwieser et al., 2020)

---

## Summary

### Current Design
- **Action Space**: Dict with length indicator (10 max actions)
- **Observation Space**: Dict with full game info
- **Validation**: Post-hoc (discard invalid)
- **Combat**: Deterministic ratio-based
- **Map**: Hard-coded 6-territory grid

### Best for Extensions
1. **Better training**: Add action masking to info dict
2. **More maps**: Implement JSON map loader
3. **Scalability**: Convert observation to graph-based
4. **Realism**: Add fog of war, probabilistic combat
5. **Complexity**: Variable income, territory bonuses, regions

### Key Takeaway
The current design optimizes for **simplicity and compatibility**. It uses well-established patterns (Dict spaces, post-hoc validation) that work with all major RL frameworks. When extending, maintain backward compatibility by adding optional features rather than changing core structure.
