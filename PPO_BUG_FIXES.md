# PPO Bug Fixes - Phase 2 TorchRL Training

**Branch:** `fix-ppo-bugs`  
**Date:** 2026-04-22  
**Status:** ✅ Complete

## Summary

Fixed two critical bugs in the custom PPO implementation that were contributing to catastrophic forgetting and training instability in Phase 2 (TorchRL + GNN).

---

## Bug #1: PPO Gradient Leak ✅ FIXED (Previous Fix)

**Location:** `parallel_risk/training/torchrl/train.py:281`

**Issue:** Old log probabilities weren't detached, causing gradients to flow through both new AND old policy.

**Fix:** Added `.detach()` to old log probs (already fixed on 2026-04-21)

```python
# CORRECT:
old_log_probs_flat = torch.cat([lp.sum(dim=1) for lp in rollout['log_probs']]).detach()
```

**Impact:** Primary cause of catastrophic forgetting - should be resolved.

---

## Bug #2: GAE Bootstrap Error ✅ FIXED (This PR)

**Location:** `parallel_risk/training/torchrl/train.py:214-256` (compute_gae method)

**Issue:** GAE computation didn't properly bootstrap from next state value for non-terminal states. Always initialized `next_value` to zeros, causing biased advantage estimates.

**Changes:**

1. **Modified `collect_rollout()` to store next observation:**
```python
rollout = {
    # ... existing keys
    'next_obs': None,  # NEW: Next observation for GAE bootstrapping
}

# At end of rollout collection:
if len(obs) > 0:
    graphs = [obs[agent] for agent in sorted(obs.keys())]
    rollout['next_obs'] = Batch.from_data_list(graphs)
```

2. **Modified `compute_gae()` to accept and use next_obs:**
```python
def compute_gae(self, rewards, values, dones, next_obs=None):  # NEW parameter
    # Bootstrap from next state value for non-terminal states
    if next_obs is not None:
        with torch.no_grad():
            _, next_value, _ = self.policy(next_obs)
            next_value = next_value.squeeze(-1)  # [batch_size]
    else:
        next_value = torch.zeros(batch_size, device=self.device)
    
    # ... rest of GAE computation uses bootstrapped next_value
```

3. **Updated `update_policy()` to pass next_obs:**
```python
advantages, returns = self.compute_gae(
    rollout['rewards'],
    rollout['values'],
    rollout['dones'],
    rollout['next_obs']  # Pass next observation for bootstrapping
)
```

**Impact:** 
- Correct advantage estimation, especially for long episodes
- Better credit assignment for multi-step decisions
- Reduces training instability

---

## Bug #3: Value Normalization Inconsistency ✅ FIXED (This PR)

**Location:** `parallel_risk/training/torchrl/train.py:321` (update_policy method)

**Issue:** Value loss used raw returns while advantages were normalized, creating scale mismatch between policy and value training.

**Changes:**

1. **Added RunningMeanStd class** (lines 29-84):
```python
class RunningMeanStd:
    """
    Track running mean and standard deviation for normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x: torch.Tensor):
        # Update running statistics with new batch
        ...
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (torch.sqrt(torch.tensor(self.var)) + self.epsilon)
```

2. **Added return normalization to PPOTrainer.__init__():**
```python
# Running statistics for value normalization (Bug #3 fix)
self.return_rms = RunningMeanStd()
```

3. **Updated value loss computation in update_policy():**
```python
# Update return statistics and normalize returns (Bug #3 fix)
returns_flat = returns.view(-1)
self.return_rms.update(returns_flat)
returns_normalized = self.return_rms.normalize(returns_flat)

# ... later in training loop:

# Value loss - now uses normalized returns (Bug #3 fix)
value_loss = nn.functional.mse_loss(new_values_flat, returns_normalized)
```

4. **Added monitoring of return statistics to TensorBoard:**
```python
# Log return statistics for monitoring (Bug #3 related)
self.writer.add_scalar('Stats/return_mean', self.return_rms.mean, self.global_step)
self.writer.add_scalar('Stats/return_std', torch.sqrt(torch.tensor(self.return_rms.var)).item(), self.global_step)
```

**Impact:**
- Matches RLlib's (Phase 1) normalization behavior
- Better value function convergence
- Reduced training instability
- Consistent scaling between policy and value updates

---

## Comparison to Phase 1 (RLlib)

| Aspect | Phase 1 (RLlib) | Phase 2 (Before) | Phase 2 (After) |
|--------|----------------|------------------|-----------------|
| Old log probs detach | ✅ Correct | ✅ Fixed | ✅ Correct |
| GAE bootstrap | ✅ Correct | ❌ Missing | ✅ Fixed |
| Value normalization | ✅ Running stats | ❌ Raw returns | ✅ Fixed |

Phase 2 now matches Phase 1's correct PPO implementation.

---

## Testing Plan

### 1. Validation Experiment
Run new training with all fixes:
```bash
python -m parallel_risk.training.torchrl.train \
    --config parallel_risk/training/torchrl/configs/gnn_gcn.yaml \
    --num-iterations 200
```

### 2. Evaluation at Checkpoints
Evaluate at iterations: 50, 100, 150, 200 using:
```bash
# TODO: Add evaluation script
```

### 3. Success Criteria
- ✅ No catastrophic forgetting (win rate drop < 10%)
- ✅ Monotonic or stable learning curve
- ✅ Win rate > 70% by iteration 200
- ✅ Value loss converges smoothly

### 4. Monitor TensorBoard
Check for:
- Stable policy loss
- Converging value loss
- Return mean/std statistics (new)
- No batch size errors

---

## Expected Results

### Before Fixes
- **Iteration 50**: 98% win rate
- **Iteration 100**: 4% win rate ❌ (catastrophic forgetting)
- **Iteration 150**: 83% win rate
- **Iteration 200**: 100% win rate

### After Fixes (Expected)
- **Iteration 50**: ~70-80% win rate
- **Iteration 100**: ~80-90% win rate
- **Iteration 150**: ~85-95% win rate
- **Iteration 200**: ~90-100% win rate

**Key improvement:** Monotonic or stable learning, no forgetting.

---

## Files Modified

1. **`parallel_risk/training/torchrl/train.py`**
   - Added `RunningMeanStd` class (lines 29-84)
   - Modified `PPOTrainer.__init__()` to add `self.return_rms`
   - Modified `collect_rollout()` to store `next_obs`
   - Modified `compute_gae()` to accept and use `next_obs` for bootstrapping
   - Modified `update_policy()` to normalize returns and pass `next_obs`
   - Added TensorBoard logging for return statistics

---

## Notes

### Why These Bugs Matter

**Bug #2 (GAE Bootstrap):**
- Without bootstrapping, advantages for non-terminal states are underestimated
- This creates bias in policy gradients
- Especially problematic for long episodes (100 turns in Parallel Risk)

**Bug #3 (Value Normalization):**
- Returns in Parallel Risk range from -1 to +1 (win/loss) but can be scaled by discounting
- Without normalization, value function struggles with varying magnitudes
- Policy sees normalized advantages but value sees raw returns → inconsistent training

### Action Decoder Issue (Not a Bug)

The 97% invalid action rate is a design issue, not a bug. Phase 1 proved it's learnable without action masking. However, action masking would improve learning efficiency for both phases.

**Optional enhancement (future work):**
- Implement action masking in `action_decoder.py`
- Mask invalid sources (not owned)
- Mask invalid destinations (not adjacent)
- Mask invalid troop counts (> available)

---

## References

- **Bug Analysis:** `phase2_bug_analysis.md`
- **RLlib Implementation:** `parallel_risk/training/rllib/train.py`
- **GAE Paper:** Schulman et al. (2016) - High-Dimensional Continuous Control Using Generalized Advantage Estimation
- **PPO Paper:** Schulman et al. (2017) - Proximal Policy Optimization Algorithms
