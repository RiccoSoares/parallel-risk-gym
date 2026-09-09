# Architectural Improvements (Optional)

Menu of potential architecture upgrades for the Parallel Risk MCTS+GNN
pipeline, distilled from comparing our current implementation against
Bauer's GG-net (IEEE ToG 2024, `github.com/andenrx/py-risk`). Ordered by
effort × confidence-in-benefit, so cheaper high-leverage changes come
first.

None of these are on the critical path for the current research plan —
they're a backlog to draw from when we decide "the current architecture
isn't the bottleneck; let's try something different".

## Context

Comparison performed with an incomplete picture: we read Bauer's public
`py-risk` code (18+ GNN model variants + configs) but not the paper
itself (paywalled on IEEE Xplore, thesis PDF 403). Architecture claims
below reflect the code; motivation/ablation claims would require the
paper.

Our current implementation (as of `mcts_gnn_selfplay` branch):
- **GNN**: `GCNConv` (Kipf-Welling), 3 layers, hidden_dim=128, mean pooling
- **Node features**: troops (log-normalized), ownership (agent-relative), in-degree, region one-hot
- **Global features**: income, turn_number, region_control
- **Policy head**: K autoregressive action slots (`src` softmax → `dst|src` softmax → `troops|src,dst` softmax)
- **Value head**: MLP on pooled graph + globals → tanh scalar
- **Training tracks**: PPO (Phase 2), AZ-distillation (Phase 3), Expert Iteration (Phase 3)

Bauer's GG-net (from `py-risk/risk/nn.py`):
- Explored GCN, GATv2, TransformerConv variants (Model1 → Model20+)
- `GlobalAttention` pooling for the value head
- Explicit hierarchical bonus-region nodes with dedicated message-passing (Model15/18/19/20)
- Per-order joint scoring policy head (score full `(src, dst, troops)` triple with one MLP; softmax over enumerated legal orders)
- Trained via genetic algorithm (PyGAD, `pop_size=30`, 25 generations)
- MCTS at `iter=200`, `moves-consider=20`, `exploration=1.0`

## Real architectural differences vs superficial ones

| Category | Difference | Notes |
|---|---|---|
| **Superficial / implementation only** | Unified `(src,dst,troops)` tuple vs separate `AttackTransferOrder`/`DeployOrder` classes | Both encode the same underlying moves. Object model choice. |
| | Order resolution priority (random shuffle vs deploys-first) | Execution detail, doesn't change strategic content. |
| **RL-framing artifact (not game rule)** | Fixed K slots per turn vs variable orders per turn | Our K is a PettingZoo compatibility constraint (fixed-shape action tensors); Warzone allows variable per-turn move counts subject to income/army limits. Consequence for the action-space encoding but not a "game is different" point. |
| **Real architectural differences** | GCN vs GATv2/TransformerConv | Layer choice — affects representational power. |
| | Mean pooling vs `GlobalAttention` | Aggregation choice — attention learns weighted contribution per node. |
| | Flat region one-hot vs explicit bonus-region nodes | Bauer models regions as first-class objects with their own message-passing and learned importance. We fold them into node features. |
| | Factorized softmax vs joint scoring | Ours: `log P(src) + log P(dst\|src) + log P(troops\|src,dst)`. Bauer: one MLP scores full `(src,dst,troops)` triples jointly. Both can represent any distribution in principle; joint scoring may learn triple-wise interactions in fewer gradient steps. |
| | Slots see static masks vs sequential intra-turn state updates | Ours: all K slots decoded from initial-observation masks; slot 2 doesn't know slot 1 conquered a territory. Bauer: each order picked given the post-prior-orders state. |
| **Different games / rules** | Combat determinism | Ours 70%/60% deterministic; Warzone default is dice. |
| | Neutral territories | Ours: none. Warzone: often present. |

## Prioritized improvement menu

Effort tiers:
- **XS**: minutes-to-hours (config change)
- **S**: 1-2 days
- **M**: 3-5 days
- **L**: 1-2 weeks (invasive; touches env or breaks existing checkpoints)

Confidence in benefit:
- **High**: strong evidence (diagnostic, literature, or direct Bauer analogue)
- **Medium**: plausible but unmeasured
- **Uncertain**: could go either way

### Tier 1 — cheap wins, do these first

**1.1 MCTS budget: 40 → 200 (XS, High confidence)**
- Config-only change (`mcts_budget=200` in the eval + self-play configs)
- Bauer runs MCTS at 200 sims/move as default
- Our diagnostic already showed budget=40 caps a good policy (94% raw PPO → 31% wrapped MCTS+GNN at budget=40 vs MCTS-uniform-40)
- Wall-clock: ~5× per-iteration eval time; still viable overnight
- Files: any of the `experiments/mcts_gnn_*_training_run.py` configs

**1.2 GCN → GATv2 or TransformerConv (S, Medium-high confidence)**
- Drop-in swap in `parallel_risk/models/gnn_gcn.py` — implement `GATv2Policy` or `TransformerConvPolicy` as siblings
- Bauer's best variants (Model12-14, Model19-20) use TransformerConv; his Model5-8 use GATv2
- Standard graph-learning literature says attention-based message passing outperforms plain GCN on many tasks
- Existing training scripts stay unchanged
- New file: `parallel_risk/models/gnn_gat.py` or `gnn_transformer.py`; add config-driven backbone selection
- Test: rerun a warm-start ExIt run with the new backbone; compare eval win-rate curves

**1.3 Attention pooling for the value head (XS-S, Medium confidence)**
- Replace `torch_geometric.nn.global_mean_pool` with `GlobalAttention` (learned gating MLP)
- Small architectural touch on `GCNPolicy.forward`
- Standard upgrade with no downside; small quality boost is likely
- Independent of the backbone swap — could combine

### Tier 2 — medium effort, uncertain-but-plausible benefit

**2.1 Explicit region-bonus modeling (M, Medium confidence)**
- Bauer's Model15/18/19/20 style: regions get their own node-set with dedicated `TransformerConv` layers and a learned per-region importance score, then feed into the value head
- Our regions map cleanly to Bauer's "bonus regions"; the concept ports directly
- Requires: (a) a new graph-construction path that adds region nodes with `region ↔ member-territory` edges; (b) a hierarchical forward pass; (c) refactoring the graph-wrapper to emit this richer structure
- Files: `parallel_risk/training/torchrl/graph_wrapper.py`, `parallel_risk/models/gnn_gcn.py` (or a new hierarchical variant)
- Benefit depends on how much strategic value the network gets from reasoning at region-granularity — likely helps more on maps with 4+ regions

**2.2 Intra-turn mask-only conditioning (S-M, Medium confidence)**
- Between slots within a single turn, update just the ownership mask based on the previous slot's decoded action's likely effect (deterministic combat lets us predict conquest outcomes)
- Slot 2 then sees the updated mask; can chain attacks (attack A→B, use B's new ownership for B→C)
- Does not require simulating the full env in the forward pass — just track predicted ownership delta
- Requires: refactoring `ActionDecoder.decode_actions` and `.compute_log_probs` to be truly autoregressive across slots
- File: `parallel_risk/models/action_decoder.py`
- Benefit likely larger on maps with corridor/chain topology than on dense meshes

**2.3 Joint (dst, troops) scoring conditioned on source (S, Low-medium confidence)**
- Keep per-slot source softmax, but replace the separate dest and troops heads with a single MLP that scores `(dst, troops)` pairs jointly conditioned on the chosen source
- Captures dst×troops interactions in one gradient step instead of two separately-trained softmaxes
- Partial fix for the factorization difference vs Bauer
- File: `parallel_risk/models/action_decoder.py`
- Low confidence because factorization is not fundamentally limiting

### Tier 3 — invasive, save for after Tier 1/2 aren't enough

**3.1 Full joint-triple scoring (M, Uncertain confidence)**
- Enumerate all legal `(src, dst, troops)` triples per state, score each with a joint MLP, softmax over the set
- Directly mirrors Bauer's per-order scoring approach
- Requires an "enumerate legal triples" API on the env (currently we don't have one — we sample via masks)
- Enumeration could be slow at K=20 with 30 territories (~10⁴+ triples per slot)
- Files: `parallel_risk/env/parallel_risk_env.py` (new enumerator), `parallel_risk/models/action_decoder.py`
- Uncertain confidence — the factorization vs joint-scoring debate isn't well-resolved in the literature for RL

**3.2 Full sequential per-slot forward passes (M, Medium confidence)**
- Instead of one forward pass producing K slots at once, do K sequential forward passes with a simulated state update between each (using deterministic combat to predict conquest outcomes)
- Cleanest semantics for intra-turn conditioning
- ~K× the forward-pass cost per action decision — significantly slower at inference
- Files: `parallel_risk/agents/mcts_gnn_agent.py`, `parallel_risk/agents/gnn_agent.py`, `parallel_risk/models/action_decoder.py`

**3.3 Variable-length action per turn (L, Uncertain confidence)**
- Rewrite the env's action space from `Dict{num_actions, Box(K, 3)}` to a proper variable-length representation
- Would break every existing checkpoint (different action-head architecture)
- Matches Warzone/Bauer's game model more faithfully but adds significant plumbing (RL frameworks want fixed action tensors)
- Not clearly worth it unless we want direct Bauer-comparability as a research contribution

**3.4 Full env-in-the-loop decoding (L, Uncertain confidence)**
- Run the actual env's `execute_action` logic inside the policy forward pass to update state between slots
- Ties the policy net to the env implementation
- Breaks the observation abstraction; complicates gradients
- Not recommended unless there's a strong specific reason

## Recommended sequence

If we decide to invest in architecture upgrades, go **Tier 1 first** — three cheap changes stacked (`mcts_budget=200` + `TransformerConv` + attention pooling). That's ~2-3 days of work total, and gives three signal-carrying data points on "what was actually holding us back?"

If Tier 1 moves the eval curves meaningfully, we know the architecture matters and Tier 2 (region modeling, intra-turn conditioning) becomes attractive.

If Tier 1 doesn't move the curves much, the bottleneck is likely elsewhere (compute scale, game complexity, or training-signal quality) and Tier 2/3 architecture changes probably won't fix it. In that case the more productive direction is scaling compute, adding coevolution (Phase 4), or working on the diagnostic setup itself (better opponent-strength calibration, cleaner metrics).

## What we specifically don't recommend

- **Changing the game rules to match Warzone exactly** (dice combat, neutrals, variable orders per turn). These are orthogonal to any interesting RL research question we could run. Only worth doing if we want to publish a direct benchmark vs GG-net.
- **Cargo-culting all 18+ of Bauer's model variants**. He iterated through many designs during development; the recent ones (Model14-20) are the ones that made it into his best results. Reproducing the earlier variants isn't useful.

## Related documents

- [docs/RL_TRAINING_ROADMAP.md](RL_TRAINING_ROADMAP.md) — overall project roadmap; this doc supplements Phase 3.5+ engineering priorities
- Bauer citation: **BAUER, A.** "Artificial intelligence with graph neural networks applied to a risk-like board game." *IEEE Transactions on Games*, v. 16, n. 2, p. 342-351, 2024
- Bauer code: <https://github.com/andenrx/py-risk>
