"""Checkpoint loading utilities shared by MCTSGNNTrainer and ExitTrainer.

Warm-starting across different `max_regions` values (e.g. a PPO checkpoint
trained on max_regions=3 loaded into a trainer with max_regions=4 because
the new map set includes hex_grid_10) requires zero-padding the
input-projection and global-projection weights on load. The rest of the
network is dimension-independent (GCN backbone + action heads share their
shapes across max_regions).

Zero-padding means the extra channels are initially a no-op — on states
where those channels are zero, the network's outputs are byte-identical to
the checkpoint. Gradients teach the padded weights during training.
"""

from typing import Dict

import torch


def load_state_dict_with_region_padding(
    module: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Load a state_dict into a module, zero-padding region-related weights.

    Handles the specific case where the source checkpoint was trained with
    fewer region features than the target module expects — pads:
      - `input_proj.weight`: `[hidden_dim, cur_node_features]`  where
        `cur_node_features = 3 + max_regions` (troops, ownership, in_degree,
        + region one-hot). Extra columns beyond source width are zeroed.
      - `global_proj.weight`: `[hidden_dim, cur_global_features]` where
        `cur_global_features = 2 + max_regions`. Same padding.

    Biases and all other weights (GCN backbone, action heads, value head)
    have shapes independent of max_regions — they load unchanged. Any
    remaining shape mismatch is re-raised (not silently patched).
    """
    target_sd = module.state_dict()
    padded = dict(state_dict)  # shallow copy so we can rewrite entries

    for key in ('input_proj.weight', 'global_proj.weight'):
        if key not in padded or key not in target_sd:
            continue
        src = padded[key]
        tgt = target_sd[key]
        if src.shape == tgt.shape:
            continue
        # We only pad when source is narrower (or equal) than target on
        # the input-dim axis (dim 1 for a Linear weight). Other mismatches
        # are unexpected and should fail loudly.
        if src.dim() != 2 or tgt.dim() != 2 or src.shape[0] != tgt.shape[0] \
                or src.shape[1] > tgt.shape[1]:
            raise RuntimeError(
                f"Cannot pad {key}: src {tuple(src.shape)} vs tgt {tuple(tgt.shape)}. "
                "Only supports src input-dim <= tgt input-dim with matching "
                "hidden_dim."
            )
        extra = tgt.shape[1] - src.shape[1]
        pad = torch.zeros(src.shape[0], extra, dtype=src.dtype, device=src.device)
        padded[key] = torch.cat([src, pad], dim=1)

    # Now load — any remaining mismatch (e.g. hidden_dim change) still raises.
    module.load_state_dict(padded)
