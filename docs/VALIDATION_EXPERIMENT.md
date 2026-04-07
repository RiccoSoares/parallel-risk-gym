# Learning Validation Experiment

**Status:** Ready to run  
**Last Updated:** 2026-04-07

## Purpose

This experiment validates that Parallel Risk is **learnable** by reinforcement learning agents. Specifically, it demonstrates that PPO-trained agents can learn to consistently beat random baseline opponents.

**Success Criteria:**
- ✅ Trained agent achieves >70% win rate vs. random policy
- ✅ Win rate improves over training iterations
- ✅ Training completes without crashes or divergence

## Methodology

### Experiment Design

1. **Baseline Measurement**
   - Random vs. Random = 50% win rate (theoretical)
   - Establishes reference point

2. **Training Phase**
   - Train PPO agent for 300-500 iterations (~2-3 hours)
   - Use sparse rewards (harder test of learning)
   - Save checkpoints every 10 iterations

3. **Evaluation Phase**
   - Evaluate checkpoints every 50 iterations
   - Run 100 episodes per evaluation
   - Measure win rate vs. random opponent

4. **Analysis**
   - Plot win rate progression
   - Generate summary statistics
   - Visualize learning curves

### Why This Validates Learning

- **Win rate improvement:** If agents learn, win rate should increase from ~50% to >70%
- **Reproducibility:** Same config + seed should produce similar results
- **Statistical significance:** 100 episodes provides 95% confidence interval of ±10%

## Running the Experiment

### Prerequisites

Ensure training dependencies are installed:
```bash
./install_training_deps.sh
# or: pip install ray[rllib]==2.40.0 torch tensorboard matplotlib
```

### Quick Test (10 iterations, ~10 minutes)

Verify everything works before full run:
```bash
python experiments/validate_learning.py \
    --num-iterations 10 \
    --eval-interval 5 \
    --num-eval-episodes 50 \
    --verbose
```

Expected: Script completes without errors, generates plots.

### Full Validation Run (500 iterations, ~2-3 hours)

```bash
python experiments/validate_learning.py \
    --num-iterations 500 \
    --eval-interval 50 \
    --num-eval-episodes 100 \
    --num-workers 4 \
    --verbose
```

**Parameters:**
- `--num-iterations`: Training iterations (default: 500)
- `--eval-interval`: Evaluate every N iterations (default: 50)
- `--num-eval-episodes`: Episodes per evaluation (default: 100)
- `--num-workers`: Parallel rollout workers (default: 4)
- `--results-dir`: Output directory (default: `experiments/validate_learning_results/`)
- `--checkpoint-dir`: Checkpoint directory (default: `checkpoints/validation_run/`)
- `--verbose`: Print detailed progress

### Monitor Progress

Training progress is printed to console:
```
Iteration 50/500 - Episode Reward: 0.12 - Episode Length: 45.3
Iteration 100/500 - Episode Reward: 0.25 - Episode Length: 42.1
...
```

Evaluation results are printed as they complete:
```
Evaluating checkpoint at iteration 50...
  Win rate: 55.00%
Evaluating checkpoint at iteration 100...
  Win rate: 62.00%
...
```

### Output Artifacts

All results are saved to `experiments/validate_learning_results/`:

- **`training_log.txt`** - Training metrics per iteration
- **`evaluation_results.json`** - Win rates at each checkpoint
- **`eval_<iteration>.json`** - Detailed results for each evaluation
- **`win_rate_curve.png`** - Win rate vs. training iteration plot
- **`episode_length_curve.png`** - Episode length over training
- **`reward_distribution.png`** - Histogram of episode rewards
- **`final_summary.txt`** - Human-readable summary

## Interpreting Results

### Success (>70% win rate)

```
Final win rate (iteration 500): 75.00%

✓ SUCCESS: Agent achieved >70% win rate vs. random
  Learning has been validated!
```

**Interpretation:** The environment is learnable, PPO is working correctly, and reward signals are sufficient. Ready to proceed to Phase 1.4 (baseline experiments).

**Next Steps:**
- Run reward shaping ablations
- Try different hyperparameters
- Implement tournament evaluation system

### Partial Success (60-70% win rate)

```
Final win rate (iteration 500): 65.00%

⚠ PARTIAL SUCCESS: Agent achieved >60% win rate
  Learning is happening but may need more training
```

**Interpretation:** Learning is occurring but not converged. Agent is clearly better than random but hasn't plateaued yet.

**Actions:**
- Extend training to 1000 iterations
- Try dense reward shaping to accelerate learning
- Check if win rate is still increasing (hasn't plateaued)

### Failure (<60% win rate)

```
Final win rate (iteration 500): 52.00%

✗ FAILURE: Agent did not achieve >60% win rate
  Learning may not be working - investigate environment/hyperparams
```

**Interpretation:** Something is wrong. Learning is not happening effectively.

**Diagnosis Checklist:**
1. **Check training metrics:**
   - Is episode reward increasing?
   - Are there NaN or exploding values?
   - Is policy loss decreasing?

2. **Check action validity:**
   - Are most actions invalid? (check `invalid_actions` in logs)
   - Is action space properly normalized?

3. **Check environment rewards:**
   - Are terminal rewards (+1/-1) being received?
   - Are shaped rewards (if enabled) reasonable?

4. **Try fixes:**
   - Switch to dense reward shaping: `reward_shaping: "dense"` in config
   - Reduce learning rate: `lr: 1.0e-4`
   - Increase exploration: `entropy_coeff: 0.05`
   - Increase training batch size: `train_batch_size: 8000`

## Expected Learning Curve

Based on similar turn-based strategy games:

```
Iteration    Win Rate    Interpretation
---------    --------    --------------
0-50         50-55%      Random exploration, barely above baseline
50-150       55-65%      Learning basic tactics (valid moves, region capture)
150-300      65-75%      Refining strategy, efficient combat
300-500      70-80%      Converged to strong policy, diminishing returns
```

**Typical progression:**
- First 100 iterations: Slow initial learning
- 100-300: Rapid improvement as agent discovers winning strategies
- 300+: Plateau as agent reaches skill ceiling against random opponent

## Troubleshooting

### Training is too slow

**Symptoms:** Each iteration takes >2 minutes

**Solutions:**
- Reduce workers: `--num-workers 2` (if limited CPU)
- Reduce batch size in config: `train_batch_size: 2000`
- Use GPU if available: `num_gpus: 1` in config (limited benefit for small models)

### Training crashes

**Symptoms:** Out of memory, Ray errors, import errors

**Solutions:**
- **Out of memory:** Reduce `train_batch_size` or `num_workers`
- **Ray errors:** Update Ray: `pip install -U ray[rllib]`
- **Import errors:** Check PYTHONPATH, reinstall: `pip install -e .`

### Evaluation fails

**Symptoms:** Cannot load checkpoint, environment errors

**Solutions:**
- Check checkpoint exists: `ls checkpoints/validation_run/`
- Verify Ray version matches training: `pip list | grep ray`
- Check environment config matches training config

### High variance in win rates

**Symptoms:** Win rate jumps around (60% → 75% → 55%)

**Solutions:**
- Increase `--num-eval-episodes 200` for more stable estimates
- Run multiple seeds and average results
- Check if agent is still learning (variance decreases after convergence)

## Advanced Usage

### Evaluate Existing Checkpoints Only

Skip training, just evaluate checkpoints that already exist:
```bash
python experiments/validate_learning.py \
    --skip-training \
    --checkpoint-dir checkpoints/my_existing_run \
    --eval-interval 50
```

### Custom Configuration

Use a different training config:
```bash
python experiments/validate_learning.py \
    --config parallel_risk/training/configs/my_custom_config.yaml \
    --num-iterations 500
```

### Separate Evaluation

Evaluate a single checkpoint manually:
```bash
python -m parallel_risk.evaluation.evaluate_agent \
    --checkpoint checkpoints/validation_run/checkpoint_000500 \
    --opponent random \
    --num-episodes 200 \
    --output results/eval_500.json \
    --verbose
```

### Generate Plots from Results

Regenerate plots from existing evaluation data:
```bash
python -m parallel_risk.evaluation.visualize \
    experiments/validate_learning_results/evaluation_results.json \
    experiments/validate_learning_results/
```

## Statistical Notes

### Confidence Intervals

With 100 evaluation episodes:
- 50% win rate: ±10% (95% CI)
- 70% win rate: ±9% (95% CI)
- 90% win rate: ±6% (95% CI)

Formula: `CI = 1.96 * sqrt(p*(1-p)/n)` where p = win rate, n = episodes

### Sample Size Guidelines

- **50 episodes:** Fast but noisy (±14% CI at 50% win rate)
- **100 episodes:** Balanced (±10% CI) - recommended default
- **200 episodes:** More stable (±7% CI) - use for final validation

### Multiple Comparisons

When evaluating multiple checkpoints (e.g., 10 evaluations at different iterations), consider:
- Bonferroni correction for significance testing
- Focus on overall trend rather than individual fluctuations
- Final win rate is most important metric

## Next Steps After Validation

### If Learning Validated (>70% win rate)

**Phase 1.3: Evaluation Harness**
- Implement tournament system
- Add Elo rating tracking
- Create policy pool for self-play

**Phase 1.4: Baseline Experiments**
- Reward shaping ablation studies
- Hyperparameter tuning
- Architecture comparisons

**Phase 2: Graph Neural Networks**
- Multi-map training
- Transfer learning experiments

### If Learning Not Validated

**Debug Environment:**
- Review reward function logic
- Check action space validity
- Verify combat resolution correctness

**Try Different RL Algorithm:**
- DQN (value-based, might handle sparse rewards better)
- SAC (off-policy, more sample efficient)
- A3C (asynchronous, different exploration)

**Simplify Problem:**
- Start with 4-territory map instead of 6
- Reduce action budget to 3
- Enable dense reward shaping

## References

### Papers
- **PPO:** "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **Self-Play:** "Emergent Complexity via Multi-Agent Competition" (OpenAI, 2017)
- **Reward Shaping:** "Policy Invariance Under Reward Shaping" (Ng et al., 1999)

### Parallel Risk Documentation
- `docs/RL_TRAINING_ROADMAP.md` - Full training plan
- `docs/RLLIB_INTEGRATION.md` - RLlib integration guide
- `docs/REWARD_SHAPING.md` - Reward shaping details

### External Resources
- RLlib documentation: https://docs.ray.io/en/latest/rllib/
- PettingZoo: https://pettingzoo.farama.org/
- Ray Tune: https://docs.ray.io/en/latest/tune/

## Changelog

**2026-04-07:** Initial validation experiment infrastructure
- Created evaluation module with agent evaluation
- Implemented RandomAgent baseline
- Added validation experiment orchestration script
- Generated documentation
