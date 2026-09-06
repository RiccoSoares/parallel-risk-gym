"""
Numerical parity test: vectorized ActionDecoder must match legacy per-graph
implementation on deterministic decoding, log_probs, and entropy.

Non-deterministic sampling uses different RNG streams (batched vs per-graph
Categorical), so we don't compare sampled actions bit-for-bit — we compare
distributions via log_probs and entropy.

If this test fails, DO NOT ship the vectorized decoder — a masking bug will
silently degrade PPO training.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torch_geometric.data import Batch

from parallel_risk import ParallelRiskEnv
from parallel_risk.training.torchrl.graph_wrapper import env_to_graph
from parallel_risk.models.action_decoder import ActionDecoder
from parallel_risk.models._action_decoder_legacy import _ActionDecoderLegacy


def build_mixed_batch(seed=0, action_budget=5, max_troops=20):
    """Build a real Batch of observations from the 3 maps, one per map."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs = []
    for map_name in ['simple_6', 'medium_8', 'large_10']:
        env = ParallelRiskEnv(map_name=map_name, seed=seed)
        obs, _ = env.reset(seed=seed)
        # Take agent_0's observation
        g = env_to_graph(obs['agent_0'], env.map_config, torch.device('cpu'))
        graphs.append(g)
    batched = Batch.from_data_list(graphs)
    return graphs, batched


def fake_logits(batched, action_budget=5, max_troops=20, seed=0):
    """Random per-slot logits with the shapes the real GCN produces."""
    torch.manual_seed(seed)
    B = int(batched.batch.max().item()) + 1
    total_nodes = batched.x.size(0)
    logits = []
    for _ in range(action_budget):
        logits.append({
            'source': torch.randn(total_nodes),
            'dest':   torch.randn(total_nodes),
            'troops': torch.randn(B, max_troops),
        })
    return logits


def test_deterministic_decode_matches_legacy():
    graphs, batched = build_mixed_batch(seed=42)
    logits = fake_logits(batched, seed=42)

    legacy = _ActionDecoderLegacy(action_budget=5, max_troops=20)
    new = ActionDecoder(action_budget=5, max_troops=20)

    a_leg, lp_leg = legacy.decode_actions(
        logits, batched.batch, deterministic=True, return_log_probs=True,
        observations=graphs,
    )
    a_new, lp_new = new.decode_actions(
        logits, batched.batch, deterministic=True, return_log_probs=True,
        observations=graphs,
    )

    assert torch.equal(a_leg, a_new), (
        "deterministic decode actions differ:\n"
        f"legacy:\n{a_leg}\nnew:\n{a_new}"
    )
    assert torch.allclose(lp_leg, lp_new, atol=1e-5), (
        f"log_probs differ: max abs diff = {(lp_leg - lp_new).abs().max().item()}"
    )
    print("  deterministic decode + log_probs: OK")


def test_compute_log_probs_matches_legacy():
    graphs, batched = build_mixed_batch(seed=7)
    logits = fake_logits(batched, seed=7)

    legacy = _ActionDecoderLegacy(action_budget=5, max_troops=20)
    new = ActionDecoder(action_budget=5, max_troops=20)

    # Get a set of actions from legacy deterministic decode (guaranteed valid)
    actions, _ = legacy.decode_actions(
        logits, batched.batch, deterministic=True, return_log_probs=False,
        observations=graphs,
    )

    lp_leg = legacy.compute_log_probs(logits, actions, batched.batch, observations=graphs)
    lp_new = new.compute_log_probs(logits, actions, batched.batch, observations=graphs)

    max_diff = (lp_leg - lp_new).abs().max().item()
    assert torch.allclose(lp_leg, lp_new, atol=1e-5), (
        f"compute_log_probs mismatch, max abs diff = {max_diff}"
    )
    print(f"  compute_log_probs on legacy-argmax actions: OK (max abs diff {max_diff:.2e})")


def test_compute_entropy_matches_legacy():
    graphs, batched = build_mixed_batch(seed=13)
    logits = fake_logits(batched, seed=13)

    legacy = _ActionDecoderLegacy(action_budget=5, max_troops=20)
    new = ActionDecoder(action_budget=5, max_troops=20)

    ent_leg = legacy.compute_entropy(logits, batched.batch, observations=graphs)
    ent_new = new.compute_entropy(logits, batched.batch, observations=graphs)

    max_diff = (ent_leg - ent_new).abs().max().item()
    assert torch.allclose(ent_leg, ent_new, atol=1e-5), (
        f"compute_entropy mismatch, max abs diff = {max_diff}\n"
        f"legacy:\n{ent_leg}\nnew:\n{ent_new}"
    )
    print(f"  compute_entropy: OK (max abs diff {max_diff:.2e})")


def test_sampled_actions_are_valid():
    """Sanity check: sampled actions should be valid indices (respect masks)."""
    graphs, batched = build_mixed_batch(seed=99)
    logits = fake_logits(batched, seed=99)
    new = ActionDecoder(action_budget=5, max_troops=20)

    torch.manual_seed(999)
    actions, log_probs = new.decode_actions(
        logits, batched.batch, deterministic=False, return_log_probs=True,
        observations=graphs,
    )

    B = len(graphs)
    for b, g in enumerate(graphs):
        n = g.num_nodes
        for slot in range(5):
            s, d, t = actions[b, slot].tolist()
            assert 0 <= s < n, f"source {s} out of range for graph {b} (n={n})"
            assert 0 <= d < n, f"dest {d} out of range for graph {b} (n={n})"
            # Ownership: source must be owned by agent
            ownership = g.x[s, 1].item()
            assert ownership == 1, f"source {s} not owned in graph {b} (ownership={ownership})"
            # Dest must equal source OR be adjacent
            if d != s:
                ei = g.edge_index
                neighbors = ei[1, ei[0] == s].tolist()
                assert d in neighbors, f"dest {d} not adjacent to source {s} in graph {b}"
    print(f"  sampled actions all valid: OK (checked {B * 5} actions)")


def test_single_graph_case():
    """The eval script passes batch_size=1. Should work identically."""
    graphs, _ = build_mixed_batch(seed=5)
    single_graph = graphs[0]
    batched = Batch.from_data_list([single_graph])
    logits = fake_logits(batched, seed=5)

    legacy = _ActionDecoderLegacy(action_budget=5, max_troops=20)
    new = ActionDecoder(action_budget=5, max_troops=20)

    a_leg, lp_leg = legacy.decode_actions(
        logits, batched.batch, deterministic=True, return_log_probs=True,
        observations=[single_graph],
    )
    a_new, lp_new = new.decode_actions(
        logits, batched.batch, deterministic=True, return_log_probs=True,
        observations=[single_graph],
    )
    assert torch.equal(a_leg, a_new), f"single-graph action mismatch:\n{a_leg}\nvs\n{a_new}"
    assert torch.allclose(lp_leg, lp_new, atol=1e-5), \
        f"single-graph log_prob mismatch: max diff = {(lp_leg - lp_new).abs().max().item()}"
    print("  single-graph case: OK")


if __name__ == "__main__":
    print("Testing ActionDecoder vectorized vs legacy parity...")
    test_deterministic_decode_matches_legacy()
    test_compute_log_probs_matches_legacy()
    test_compute_entropy_matches_legacy()
    test_sampled_actions_are_valid()
    test_single_graph_case()
    print("\nAll parity tests passed!")
