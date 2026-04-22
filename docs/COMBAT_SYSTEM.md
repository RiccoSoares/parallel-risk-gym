# Combat System Documentation

## Current Implementation: Percentage-Based Deterministic Combat

### Rules

Given `x` attacking troops and `y` defending troops:

1. **Defender casualties** = 60% of x (rounded down)
2. **Attacker casualties** = 70% of y (rounded down)
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
- `x * 0.6 > y` (must inflict more than y casualties)
- Solving: `x > y / 0.6 = y * 1.67`
- **Attacker needs ~1.67× defending troops to win**

**Casualty comparison:**
- Attackers lose more per troop: 70% vs 60%
- Defender advantage: ~17% fewer casualties
- This creates defensive advantage (like real Risk)

### Combat Examples

| Attackers | Defenders | Def. Casualties | Att. Casualties | Def. Remaining | Result | Survivors |
|-----------|-----------|-----------------|-----------------|----------------|--------|-----------|
| 10 | 5 | 6 | 3 | -1 | ✅ Attacker wins | 7 at dest |
| 10 | 10 | 6 | 7 | 4 | ❌ Defender holds | 4 defenders, 3 attackers return |
| 10 | 15 | 6 | 10 | 9 | ❌ Defender holds | 9 defenders, 0 attackers return |
| 20 | 10 | 12 | 7 | -2 | ✅ Attacker wins | 13 at dest |
| 5 | 8 | 3 | 5 | 5 | ❌ Defender holds | 5 defenders, 0 attackers return |
| 17 | 10 | 10 | 7 | 0 | ✅ Attacker wins | 10 at dest |

### Strategic Implications

**For Attackers:**
- Need ~2× forces for reliable capture (accounting for casualties)
- Overwhelming force is efficient (20 vs 10 → keep 13)
- Failed attacks now less punishing - surviving troops return home
- Can probe defenses with smaller attacks to test strength

**For Defenders:**
- Even small forces can hold (5 defenders survive 10 attackers)
- Fortifying key territories is effective
- Breaking enemy regions doesn't require full conquest

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

Current ratios (60% / 70%) with survivor return create:
- **Moderate defender advantage**: Defenders lose fewer troops proportionally (60% vs 70%)
- **Attacker threshold**: Needs ~1.67× defender troops to successfully capture
- **Reduced attack penalty**: Failed attacks return survivors, making aggressive play more viable
- **Economic warfare**: Both sides can inflict casualties without captures, but failed attacks now less punishing
- **Encourages probing**: Attackers can test defenses with smaller forces since survivors return

