# Combat System Documentation

## Current Implementation: Percentage-Based Deterministic Combat

### Rules

Given `x` attacking troops and `y` defending troops:

1. **Defender casualties** = 70% of x (rounded down)
2. **Attacker casualties** = 60% of y (rounded down)
3. **Outcome**:
   - If `defenders_remaining = y - defender_casualties <= 0`: **Attacker wins**
     - Territory ownership transfers to attacker
     - Surviving attackers occupy the territory: `max(1, x - attacker_casualties)`
   - Otherwise: **Defender holds**
     - Territory remains with defender
     - Defenders reduced to `y - defender_casualties`
     - Surviving attackers return to source territory: `max(0, x - attacker_casualties)`

### Mathematical Properties

**Break-even ratio:**
To capture a territory, attacker needs:
- `x * 0.7 > y` (must inflict more than y casualties)
- Solving: `x > y / 0.7 = y * 1.43`
- **Attacker needs ~1.43× defending troops to win**

**Casualty comparison:**
- Attackers lose fewer per troop: 60% vs 70%
- Attacker advantage: ~17% fewer casualties
- This creates defensive advantage (like real Risk)

### Combat Examples

| Attackers | Defenders | Def. Casualties | Att. Casualties | Def. Remaining | Result | Survivors |
|-----------|-----------|-----------------|-----------------|----------------|--------|-----------|
| 10 | 5 | 7 | 3 | -2 | ✅ Attacker wins | 7 at dest |
| 10 | 10 | 7 | 6 | 3 | ❌ Defender holds | 3 defenders, 4 attackers return |
| 10 | 15 | 7 | 9 | 8 | ❌ Defender holds | 8 defenders, 1 attacker returns |
| 20 | 10 | 14 | 6 | -4 | ✅ Attacker wins | 14 at dest |
| 5 | 8 | 3 | 4 | 5 | ❌ Defender holds | 5 defenders, 1 attacker returns |
| 17 | 10 | 11 | 6 | -1 | ✅ Attacker wins | 11 at dest |

### Strategic Implications

**For Attackers:**
- Need ~1.43× forces for reliable capture (down from 1.67×)
- Lower attack threshold encourages aggressive play
- Failed attacks less punishing - more survivors return
- Can effectively probe with smaller forces
- Attacking is now economically favorable

**For Defenders:**
- Must maintain stronger defensive positions
- Territory control more volatile
- Fortification still important but less dominant

**For Both:**
- Combat is predictable (deterministic)
- Can calculate exact outcomes
- Encourages tactical planning over random aggression

### Testing

**Test file:** `tests/test_combat.py`

Covers:
- ✅ Attacker wins scenario (10 vs 5)
- ✅ Defender holds scenario (10 vs 15)
- ✅ Overwhelming attack (20 vs 10)
- ✅ Equal forces edge case (10 vs 10)
- ✅ All calculations verified


### Balance Notes

Current ratios (70% / 60%) with survivor return create:
- **Moderate attacker advantage**: Attackers lose fewer troops proportionally (60% vs 70%)
- **Attacker threshold**: Needs ~1.43× defender troops to successfully capture (down from 1.67×)
- **Encourages aggression**: Lower threshold and survivor return make attacking more viable
- **Dynamic gameplay**: Territory control is more fluid and volatile
- **Rewards offensive strategies**: Attacking is now economically favorable
- **Reduced attack penalty**: Failed attacks return survivors, making aggressive play more viable
- **Economic warfare**: Both sides can inflict casualties without captures, but failed attacks now less punishing
- **Encourages probing**: Attackers can test defenses with smaller forces since survivors return

