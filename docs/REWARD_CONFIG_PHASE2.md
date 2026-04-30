# Phase 2 Reward Configuration

## Problem: Defensive Equilibrium

Phase 2 GNN training showed agents learning a **defensive strategy**:
- All episodes went to 100 turns (turn limit)
- Agents avoided attacking, just building up defensively
- Win rate oscillated wildly (6% to 94%) based on who had slightly more at turn limit
- Old turn limit rewards (+0.5 winner, -0.5 loser) made this a viable strategy

## Solution: Reward Changes

### 1. Turn Limit Penalty (Core Environment)

Changed turn limit rewards to discourage defensive play:

| Outcome | Old | New | Rationale |
|---------|-----|-----|-----------|
| Turn limit winner | +0.5 | **+0.2** | Small consolation, not a real win |
| Turn limit loser | -0.5 | **-0.8** | Almost as bad as elimination |
| Expected value | 0.0 | **-0.3** | Both agents should prefer decisive games |

### 2. Reward Shaping Components (All Enabled)

| Component | Weight | Description | Purpose |
|-----------|--------|-------------|---------|
| **territory_control** | 0.01 | % of map controlled | Baseline progress signal |
| **region_completion** | 0.1 | One-time bonus per region | Encourage regional domination |
| **troop_advantage** | 0.01 | Ratio of troops vs opponent | Encourage building strength |
| **strategic_position** | 0.005 | Connectivity score of territories | Prefer well-connected positions |
| **territory_conquest** | 0.1 | Immediate reward per capture | **Direct incentive for aggression** |

### 3. Key New Component: Territory Conquest

The `territory_conquest` reward provides **immediate positive feedback** when capturing enemy territories:

- +0.1 per territory captured (same turn)
- Captured 3 territories? Get +0.3 that step
- Creates clear learning signal: attacking = good

## Expected Behavior

With these changes, agents should:

1. **Attack more often** - conquest gives immediate reward
2. **End games decisively** - turn limit is now a net negative
3. **Take calculated risks** - +0.1 conquest vs risk of losing troops
4. **Reduce episode length** - games should end before turn 100

## Comparison: Phase 1 vs Phase 2

| Metric | Phase 1 (Success) | Phase 2 (Old) | Phase 2 (New) |
|--------|-------------------|---------------|---------------|
| Final Win Rate | 100% | 46% | TBD |
| Avg Episode Length | 14-24 turns | 100 turns | TBD |
| Strategy | Aggressive | Defensive | Expected: Aggressive |

## Configuration Reference

Default config (all enabled):
```python
RewardShapingConfig(
    enable_territory_control=True,      # 0.01 weight
    enable_region_completion=True,      # 0.1 weight
    enable_troop_advantage=True,        # 0.01 weight
    enable_strategic_position=True,     # 0.005 weight
    enable_territory_conquest=True,     # 0.1 weight
)
```

For conquest-focused training:
```python
from parallel_risk.env.reward_shaping import create_conquest_config
env = ParallelRiskEnv(reward_shaping_config=create_conquest_config())
```
