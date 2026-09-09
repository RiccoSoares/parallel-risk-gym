---
name: research-experiment
description: End-to-end workflow for running an RL experiment on parallel-risk-gym — familiarize → plan plots → launch → monitor → fix → interpret → document. Invoke via /research-experiment when the user wants to run a new experiment, a proposal from docs/EXPERIMENT_PROPOSALS.md, or a rerun of an existing script with new parameters.
when_to_use: The user asks to run an experiment, validate a technique, test a hypothesis, sweep a hyperparameter, or execute an entry from EXPERIMENT_PROPOSALS.md. Do NOT invoke for one-off code edits, doc writes, or tests — only for research runs that generate metrics/plots.
allowed-tools: [Read, Write, Edit, Glob, Grep, Bash, PowerShell, Monitor, TodoWrite, SendUserFile, Agent, ScheduleWakeup, CronCreate]
---

# Research Experiment Workflow

You are running an RL experiment on `parallel-risk-gym`. Follow the
phases below in order. Do not skip Phase 1 — every experiment starts by
grounding in what's already been done.

At the start, spawn a `TodoWrite` list mirroring the phases so the user
can see progress. Update it after each phase.

---

## Phase 1 — Familiarize

Before writing any code, read these in order:

1. **[docs/EXPERIMENT_LOG.md](../../../docs/EXPERIMENT_LOG.md)** — full
   ledger of prior experiments (Setup → Result → What it changes).
   Check if the current question is answered or partially answered.
2. **[docs/EXPERIMENT_PROPOSALS.md](../../../docs/EXPERIMENT_PROPOSALS.md)**
   — if the user's request matches a listed entry, use its Setup /
   Files-touched fields directly.
3. **[docs/RL_TRAINING_ROADMAP.md](../../../docs/RL_TRAINING_ROADMAP.md)**
   — where the experiment sits in the plan.
4. **[docs/ARCHITECTURAL_IMPROVEMENTS.md](../../../docs/ARCHITECTURAL_IMPROVEMENTS.md)**
   — if the request is an architecture upgrade, check its tier.
5. **[CLAUDE.md](../../../CLAUDE.md)** — env conventions, common
   gotchas, running conventions.
6. Relevant existing scripts under `experiments/` and modules under
   `parallel_risk/` for the technique you're about to run.

**Before proceeding, state in one sentence what you learned that
changes your plan.** If the experiment is a duplicate of something in
EXPERIMENT_LOG.md, stop and ask the user whether they want a rerun or
a variant.

## Phase 2 — Feature implementation (only if needed)

Most experiments are variants of existing scripts and skip this phase.
Only implement new code when:
- The experiment needs a new agent variant, loss, or measurement.
- The requested plot type doesn't exist yet.
- A missing plumbing change blocks the run (e.g., `max_actions_per_turn`
  for K > 10, region-padded checkpoint loading for max_regions mismatch).

Rules:
- Keep changes atomic — if you touch an agent class signature, update
  every callsite in the same edit. Broken worker imports mid-experiment
  waste hours.
- Add a smoke test under `tests/` alongside any new module. Follow
  existing `tests/test_*.py` patterns; use `[OK]/[FAIL]` prints, not
  unicode ticks (Windows cp1252 breaks on ✓).
- Verify: run `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONPATH=. python
  tests/test_your_new_file.py` and any adjacent tests before firing
  the full experiment.

## Phase 3 — Plan and code the plots

**Plots must be research-quality**, not throwaway. Decide the plot
shape *before* running the experiment so the metrics collector emits
the right fields.

Conventions established on this repo:
- Categorical palette: `matplotlib.get_cmap('tab10')` for ≤ 10
  series, `'tab20'` beyond. Sort by name for stable colors across
  runs. See `experiments/multi_map_training.py::_map_colors`.
- No emoji, no yellow (unreadable), no red-green pairing (colorblind
  hostile).
- Every reported score includes a **95% Wilson score CI** on error
  bars — see `summarize()` in
  `experiments/diagnostic_mcts_budget.py`.
- Bar charts sorted by map size (`MapRegistry.get(m).n_territories`)
  so small→large reads left→right.
- Save both a full dashboard (2-4 panel) AND, when comparing multiple
  runs, a **combined comparison plot** (see
  `experiments/diagnostic_mcts_budget/_plot_combined.py`).
- Font sizes: title 11-12, labels 9-10, legend 9. Rotate x-labels 25°
  when many maps.
- Use `matplotlib.use('Agg')` at module top — every experiment runs
  headless on Windows.
- `plt.tight_layout(rect=[0, 0, 1, 0.93])` when using `suptitle`.

Plot outputs go under `experiments/<experiment_name>/`. JSON metrics
too. **Never let a plot fail silently** — the experiment script must
generate the plot at the end of its run, not in a follow-up step.

## Phase 4 — Launch efficiently

### The loop (Phases 4-6 iterate)

```
launch → monitor progress → notified on completion → verify → (bug? fix, go to launch)
                                                            ↓
                                                       results ok → Phase 5
```

### Launch settings

- **GPU:** enable with `use_gpu: true` in the trainer config for
  training runs. Diagnostic matchups (Q3/Q4) run on CPU — MCTS is
  Python-bound.
- **Workers:** cap `ProcessPoolExecutor` at
  `min(len(maps), 8)`. **Do not** use `max_workers=len(maps)`
  unless the host has that many free cores — the 21-map K-sweep
  hung when 21 eval workers spawned alongside 8 rollout workers
  (over-subscription deadlock).
- **Windows spawn mode:** every worker-pool experiment sets
  `mp.set_start_method('spawn', force=True)` in `__main__`. See
  `commit 62f6783` for the forkserver Windows bug.
- **`max_actions_per_turn`:** when `action_budget > 10`, pass
  `max_actions_per_turn=max(10, action_budget)` to `ParallelRiskEnv`.
  The K > 10 shape bug (`np.zeros((10 - K, 3))` = negative) was
  fixed in commit `5ed7dae` — mirror that pattern in every new
  script.
- **Env vars:** always launch with `PYTHONIOENCODING=utf-8
  PYTHONUTF8=1 PYTHONPATH=.` prefixed (see CLAUDE.md).

### Backgrounding

Long runs go to background: `Bash(run_in_background=True)`. Redirect
stdout to a `.log` file next to the output directory.

Sanity-check first: 1 map, 4 games (or 5 iterations for training)
with 4 workers, ~1 min. Bugs surface at cheap sanity size.

## Phase 5 — Verify running & get notified

**Arm a Monitor immediately after backgrounding** — the harness's
completion notification only fires on process exit, not on silent
hangs. Pattern:

```
Monitor(
  command='tail -f experiments/<name>.log | grep -E --line-buffered "\\[[0-9]+/[0-9]+\\]|score=|Aggregate|Traceback|Error|FAILED|Killed"',
  timeout_ms=3600000,
  persistent=False,
  description='<experiment>: per-<unit> progress + errors',
)
```

The grep alternation MUST cover both progress markers AND failure
signatures (Traceback/Error/Killed/OOM/FAILED). A monitor that only
matches success looks identical to "still running" during a
crashloop.

For runs > 30 min the user requires a periodic health check (see
memory: feedback_experiment_health_checks). Use ScheduleWakeup or a
30-min CronCreate to check that the log's last progress marker
timestamp isn't stale.

If a bug fires mid-run:
1. Stop the background task via TaskStop.
2. Fix root cause (do NOT patch by disabling the check that fired).
3. Rerun sanity on the fixed code.
4. Relaunch full.
5. Update EXPERIMENT_LOG.md's entry to note the bug + fix.

## Phase 6 — Search prior context (optional)

Only run when the results are unclear or contradict expectations.
Sources, in order:

1. **This repo's own history** — `git log --oneline -30`, `git blame`
   on the relevant files, older
   `experiments/*_results/results.json`. Prior authors (colleague on
   maps, older sessions) may have run something related.
2. **EXPERIMENT_LOG.md cross-cutting conclusions** — often there's
   already a stated pattern that explains the anomaly.
3. **External literature**: the Bauer paper (paywalled — see thesis
   at IEEE Xplore v.16 n.2 2024), the AlphaZero / MuZero papers, and
   Lanctot 2013 for decoupled UCT are the primary references. Use
   WebSearch when specifically prompted by the anomaly.
4. **claude-code-guide agent** — for questions about Claude Code /
   SDK / tool behavior.

Skip this phase if results match expectations.

## Phase 7 — Interpret

Interpretation is the deliverable, not the plot. Structure:

- **Headline number** — the aggregate the reader will remember.
- **Per-map / per-condition table** — sorted meaningfully (by size,
  by score, by regime).
- **Regimes** — identify 2-4 distinct behavior clusters (e.g., in
  Q3: crushed / dominant / draw-heavy / stalemate).
- **What it changes** — what earlier conclusion (if any) is now
  wrong, sharpened, or supported.
- **What it doesn't answer** — be explicit about the interpretation's
  scope so follow-ups are obvious.

If the result is null / noisy / inconclusive, say so plainly.
Publishing a false positive is worse than an honest "no signal
detected".

## Phase 8 — Document the conclusion

Two touches, both required:

1. **Append an entry to
   [docs/EXPERIMENT_LOG.md](../../../docs/EXPERIMENT_LOG.md)** at the
   bottom, following the Setup → Result → What it changes template.
   Cross-link scripts, result paths, and the commit hash.
2. **Update
   [docs/EXPERIMENT_PROPOSALS.md](../../../docs/EXPERIMENT_PROPOSALS.md)**
   — mark the proposal as done, add any new open threads the results
   suggest. Prune stale entries.

If the finding is significant enough to change the plan, also update
`docs/RL_TRAINING_ROADMAP.md` (rare — usually a paragraph edit).

Send the final plot to the user via SendUserFile (they may be viewing
from another device).

## Phase 9 — Commit

- Stage explicitly by filename — never `git add -A` or `git add .`.
  Committing scratch outputs was a repeated issue in prior sessions.
- Exclude: `.log` files, `_sanity/` dirs, `results_partial.json`,
  raw TensorBoard `runs/`. Keep: scripts, `results.json`, final
  plots.
- Commit message: one-line summary + body describing what was learned.
  Attribute Claude per the conversation's active attribution rule.
- After commit, `git status` to confirm clean working tree, and
  report the commit hash + line count in the final user-facing
  message.

---

## Repo-specific gotchas (do not repeat these)

- **Windows unicode:** always `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`
  when launching Python. Test scripts must use `[OK]/[FAIL]`, never
  ✓/✗.
- **ProcessPoolExecutor sizing:** `min(len(maps), 8)` maximum.
- **K > 10:** pass `max_actions_per_turn=max(10, K)`.
- **max_regions mismatch:** use
  `parallel_risk/training/mcts_gnn/checkpoint_utils.py::load_state_dict_with_region_padding`
  to load checkpoints trained on smaller region sets.
- **Per-slot log_prob clamp:** must be per-slot (-10 each), not
  summed (-20 total) — clamping the sum blows up at K > 7.
- **MCTS baseline is a moving target across K** — always fix a
  common opponent (MCTS-K5 or MaskedRandom) when comparing across K
  values.
- **medium_8 at K=5, max_turns=40 is a defensive equilibrium** —
  100% draws, not a training failure. Consider excluding from
  cold-start experiments.
- **Cold-start AZ/ExIt does not work** on the 9-map roster without
  warm-start weights or higher search budget (draws collapse the
  signal).
- **Don't touch validated code** — if HEAD works, don't rewrite it
  based on stale older-revision comments. See
  memory: feedback_dont_touch_validated_code.

## Tools you can invoke inline

- `Agent(subagent_type='Explore')` for cross-file searches you'd
  otherwise do in > 3 grep queries.
- `Agent(subagent_type='claude-code-guide')` for skill / SDK / CLI
  questions (as this skill did to look up its own format).
- `SendUserFile` to deliver final plots — the user may be on a
  different device.
- `Monitor` for streaming progress from long-running scripts.

## Deliverables checklist

At the end of a `/research-experiment` invocation, verify:
- [ ] `results.json` in the experiment output directory
- [ ] Final dashboard `.png` (and combined plot if multi-run)
- [ ] Entry in `EXPERIMENT_LOG.md`
- [ ] Proposal entry updated in `EXPERIMENT_PROPOSALS.md`
- [ ] Commit created, working tree clean
- [ ] Final plot sent to user via SendUserFile
- [ ] Todos all marked completed
