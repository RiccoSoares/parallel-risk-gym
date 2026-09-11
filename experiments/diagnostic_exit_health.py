"""Diagnostic: is the ExIt cold-start failure hiding a bigger bug?

Loads a checkpoint from the failed run and inspects three things that
should each rule out (or confirm) a specific class of hidden bug:

  (1) Mask-clamp rate — % of MCTS-chosen actions the decoder scores at
      log_prob <= -20 (my numerical floor). If this is >5%, the decoder's
      static-mask log_prob computation disagrees with MCTS's action
      validation. That would be a real bug hiding behind the clamp.

  (2) Advantage distribution — after per-batch normalization, are
      advantages varied (unit variance) or degenerate (all near zero)?
      Degeneracy would explain the zero learning signal.

  (3) Value head signal — for a batch of examples with known z outcomes,
      does the trained value head predict positive for z=+1 states and
      negative for z=-1? If not, the value head hasn't learned to
      distinguish winning from losing.
"""

from __future__ import annotations

# Load torch first (Windows RLlib/torch import ordering).
import torch  # noqa: F401

import argparse
import statistics
from pathlib import Path

import numpy as np


def _build_agent_and_env(ckpt_path, map_name, max_turns):
    from parallel_risk import ParallelRiskEnv
    from parallel_risk.agents.mcts_gnn_agent import MCTSGNNAgent
    env = ParallelRiskEnv(map_name=map_name, max_turns=max_turns,
                          reward_shaping_config=None)
    if ckpt_path is not None:
        agent = MCTSGNNAgent.from_checkpoint(
            ckpt_path, env.map_config,
            simulation_budget=40, c_puct=1.4, device='cpu',
        )
    else:
        # Random-init GCNPolicy the same way MCTSGNNAgent would build one
        from parallel_risk.models.action_decoder import ActionDecoder
        from parallel_risk.models.gnn_gcn import GCNPolicy
        n_regions = len(env.map_config.regions)
        policy = GCNPolicy(
            node_features_dim=3 + n_regions,
            global_features_dim=2 + n_regions,
            hidden_dim=128, num_layers=3, action_budget=5, max_troops=20,
            dropout=0.0,
        )
        policy.eval()
        decoder = ActionDecoder(action_budget=5, max_troops=20)
        agent = MCTSGNNAgent(policy=policy, decoder=decoder,
                             map_config=env.map_config,
                             simulation_budget=40, c_puct=1.4, action_budget=5,
                             max_turns=max_turns, device='cpu')
    return agent, env


def collect_one_game(ckpt_path, map_name, max_turns, seed):
    """Run one self-play game with the given checkpoint; return raw examples."""
    from parallel_risk.training.mcts_gnn.exit_self_play import play_episode_exit
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent, env = _build_agent_and_env(ckpt_path, map_name, max_turns)
    examples = play_episode_exit(
        agent, env,
        {'dirichlet_alpha': 0.3, 'noise_frac': 0.25, 'dirichlet_min_actions': 8,
         'temperature_turns': 10, 'max_temperature': 1.0},
        map_name=map_name,
    )
    return examples, agent


def diagnose_mask_clamp_rate(examples, agent):
    """(1) How often does the decoder score an MCTS-chosen action at ~-1e10?"""
    from torch_geometric.data import Batch
    if not examples:
        return None
    graphs = [ex['graph'] for ex in examples]
    action_rows = [[list(row) for row in ex['action_key']] for ex in examples]
    actions = torch.tensor(action_rows, dtype=torch.long)
    mega = Batch.from_data_list(graphs)
    with torch.no_grad():
        action_logits, _, _ = agent.policy(mega)
        raw_lp = agent.decoder.compute_log_probs(
            action_logits, actions, mega.batch, observations=graphs,
        )  # [N, action_budget]
    # Per-slot flags
    slot_lp = raw_lp.view(-1)
    hits = (slot_lp < -1e5).float().sum().item()   # -1e10 sentinel range
    below_20 = (slot_lp < -20.0).float().sum().item()
    total_slots = slot_lp.numel()
    per_example_lp = raw_lp.sum(dim=1)
    return {
        'total_examples': len(examples),
        'total_slot_evals': total_slots,
        'slots_hitting_1e10_sentinel': int(hits),
        'slots_below_-20_(clamp_floor)': int(below_20),
        'clamp_rate': below_20 / total_slots,
        'per_example_lp_min': float(per_example_lp.min().item()),
        'per_example_lp_max': float(per_example_lp.max().item()),
        'per_example_lp_median': float(per_example_lp.median().item()),
    }


def diagnose_advantage_distribution(examples, agent):
    """(2) Are advantages varied after per-batch normalization?"""
    from torch_geometric.data import Batch
    graphs = [ex['graph'] for ex in examples]
    z = torch.tensor([ex['z'] for ex in examples], dtype=torch.float32)
    mega = Batch.from_data_list(graphs)
    with torch.no_grad():
        _, values, _ = agent.policy(mega)
    v = values.squeeze(-1)
    A_raw = z - v
    A_mean = A_raw.mean().item()
    A_std = A_raw.std(unbiased=False).item()
    A_norm = (A_raw - A_mean) / (A_std + 1e-8)
    return {
        'z_distribution': {
            'unique_values': sorted(set(z.tolist())),
            'counts': dict((round(v_, 3), int((z == v_).sum())) for v_ in set(z.tolist())),
        },
        'value_pred_stats': {
            'min': float(v.min()), 'max': float(v.max()),
            'mean': float(v.mean()), 'std': float(v.std(unbiased=False)),
        },
        'raw_advantage_stats': {
            'min': float(A_raw.min()), 'max': float(A_raw.max()),
            'mean': A_mean, 'std': A_std,
        },
        'normalized_advantage_stats': {
            'min': float(A_norm.min()), 'max': float(A_norm.max()),
            'mean': float(A_norm.mean()), 'std': float(A_norm.std(unbiased=False)),
        },
    }


def diagnose_value_signal(examples, agent):
    """(3) Does v(state) correlate with z(outcome)?"""
    from torch_geometric.data import Batch
    graphs = [ex['graph'] for ex in examples]
    z = np.array([ex['z'] for ex in examples], dtype=np.float32)
    mega = Batch.from_data_list(graphs)
    with torch.no_grad():
        _, values, _ = agent.policy(mega)
    v = values.squeeze(-1).cpu().numpy()

    # Group by z
    v_win = v[z > 0]
    v_loss = v[z < 0]
    v_draw = v[z == 0]

    return {
        'n_win': int(len(v_win)),  'v_win_mean': float(v_win.mean()) if len(v_win) else None,
        'n_loss': int(len(v_loss)), 'v_loss_mean': float(v_loss.mean()) if len(v_loss) else None,
        'n_draw': int(len(v_draw)), 'v_draw_mean': float(v_draw.mean()) if len(v_draw) else None,
        'v_win_minus_v_loss': (float(v_win.mean() - v_loss.mean())
                               if len(v_win) and len(v_loss) else None),
        # Correlation between v and z (Pearson-ish)
        'corr_v_z': (float(np.corrcoef(v, z)[0, 1])
                     if len(v) > 1 and z.std() > 0 else None),
    }


def run_map(ckpt_path, map_name, max_turns, seed, num_games=5):
    """Aggregate a diagnosis across multiple games on one map."""
    from parallel_risk.training.mcts_gnn.exit_self_play import play_episode_exit

    all_examples = []
    for g in range(num_games):
        examples, agent = collect_one_game(ckpt_path, map_name, max_turns, seed + g)
        all_examples.extend(examples)
    # Use the last agent (they're all the same weights)
    print(f"\n--- MAP: {map_name}  ({num_games} games, {len(all_examples)} examples) ---")

    m1 = diagnose_mask_clamp_rate(all_examples, agent)
    print(f"(1) MASK CLAMP RATE")
    print(f"    total slot evals: {m1['total_slot_evals']}")
    print(f"    hit -1e10 sentinel: {m1['slots_hitting_1e10_sentinel']} "
          f"({100*m1['slots_hitting_1e10_sentinel']/m1['total_slot_evals']:.1f}%)")
    print(f"    below -20 clamp:   {m1['slots_below_-20_(clamp_floor)']} "
          f"({100*m1['clamp_rate']:.1f}%)")
    print(f"    per-example log_prob: min={m1['per_example_lp_min']:.2f}  "
          f"med={m1['per_example_lp_median']:.2f}  max={m1['per_example_lp_max']:.2f}")

    m2 = diagnose_advantage_distribution(all_examples, agent)
    print(f"(2) ADVANTAGE DISTRIBUTION")
    print(f"    z counts: {m2['z_distribution']['counts']}")
    print(f"    v(s) stats: {m2['value_pred_stats']}")
    print(f"    raw advantage: mean={m2['raw_advantage_stats']['mean']:+.4f}  "
          f"std={m2['raw_advantage_stats']['std']:.4f}  "
          f"range=[{m2['raw_advantage_stats']['min']:+.3f}, {m2['raw_advantage_stats']['max']:+.3f}]")
    print(f"    normalized advantage std (should be ~1.0): "
          f"{m2['normalized_advantage_stats']['std']:.4f}")

    m3 = diagnose_value_signal(all_examples, agent)
    print(f"(3) VALUE HEAD SIGNAL")
    print(f"    n_win={m3['n_win']:3d}  v_win_mean={m3['v_win_mean']}")
    print(f"    n_loss={m3['n_loss']:3d} v_loss_mean={m3['v_loss_mean']}")
    print(f"    n_draw={m3['n_draw']:3d} v_draw_mean={m3['v_draw_mean']}")
    print(f"    v(win)-v(loss) = {m3['v_win_minus_v_loss']}  "
          f"(positive = value head distinguishes win from loss)")
    print(f"    corr(v, z) = {m3['corr_v_z']}  (positive = value tracks outcome)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='Path to a saved .pt checkpoint. If omitted, use random init.')
    ap.add_argument('--maps', type=str, default='simple_6,large_10',
                    help='Comma-separated maps to diagnose.')
    ap.add_argument('--max-turns', type=int, default=40)
    ap.add_argument('--num-games', type=int, default=5)
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()

    print("=" * 70)
    print(f"ExIt cold-start diagnostic")
    print(f"  checkpoint: {args.checkpoint or 'RANDOM INIT'}")
    print(f"  maps: {args.maps}   max_turns: {args.max_turns}   games: {args.num_games}")
    print("=" * 70)

    for m in args.maps.split(','):
        run_map(args.checkpoint, m.strip(), args.max_turns, args.seed,
                num_games=args.num_games)

    print("\n" + "=" * 70)
    print("Interpretation cheat-sheet:")
    print("  (1) clamp rate > 5% => decoder/validator mismatch — REAL BUG")
    print("  (2) normalized advantage std ≈ 1 => healthy; << 1 => variance collapse")
    print("  (3) v(win)-v(loss) > 0.1 => value head is learning; ≈ 0 => not learning")


if __name__ == "__main__":
    main()
