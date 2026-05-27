"""
Key node selection (Section 4.2 / Fig. 2a).

Two-stage process:
  1. Per-frame 3D voxel sampling: at each frame independently, partition the
     scene into a voxel grid, discard voxels that contain only high-uncertainty
     Gaussians, randomly select one low-uncertainty Gaussian per remaining voxel.
     Union the per-frame candidate sets.
  2. Significant Period Threshold (SPT) filter: keep only candidates whose
     uncertainty stays below the threshold for at least `spt` *consecutive* frames.
  3. Top-k selection: rank by mean uncertainty (ascending), keep top `key_ratio`
     fraction → is_key boolean mask of shape (G,).
"""

from __future__ import annotations

import torch
from torch import Tensor


def select_key_nodes(
    u: Tensor,            # (G, T)  scalar uncertainties
    positions: Tensor,    # (G, T, 3)  per-frame Gaussian positions
    scene_scale: float = 1.0,
    key_ratio: float = 0.02,
    spt: int = 5,
    voxel_resolution: int = 50,
    seed: int = 0,
) -> Tensor:
    """Select key nodes and return a boolean mask.

    Parameters
    ----------
    u               : (G, T) scalar uncertainties
    positions       : (G, T, 3) per-frame Gaussian positions
    scene_scale     : SoM scene_scale for spatial normalization
    key_ratio       : fraction of Gaussians to keep as key nodes (default 0.02)
    spt             : Significant Period Threshold in frames (default 5)
    voxel_resolution: number of voxels per spatial axis (default 50)
    seed            : random seed for reproducibility

    Returns
    -------
    is_key : (G,) bool tensor
    """
    torch.manual_seed(seed)

    G, T = u.shape
    device = u.device

    # Global uncertainty threshold: the value at the (1 - key_ratio) quantile,
    # so the top key_ratio fraction of Gaussians have u < u_thresh.
    u_thresh = torch.quantile(u, 1.0 - key_ratio)

    # ── Stage 1: per-frame voxel sampling ─────────────────────────────────────
    # Normalize positions by scene_scale to get a uniform spatial volume.
    pos_norm = positions / (scene_scale + 1e-8)  # (G, T, 3)

    candidate_mask = torch.zeros(G, dtype=torch.bool, device=device)

    for t in range(T):
        pos_t = pos_norm[:, t, :]   # (G, 3)
        u_t   = u[:, t]             # (G,)

        # Only consider Gaussians with uncertainty below threshold.
        low_u_mask = u_t < u_thresh  # (G,)
        if low_u_mask.sum() == 0:
            continue

        # Assign each Gaussian to a voxel.
        voxel_ids = _assign_voxels(pos_t, voxel_resolution)  # (G,) int64

        # For voxels that have at least one low-uncertainty Gaussian,
        # randomly select one.
        low_u_voxels = voxel_ids[low_u_mask]        # subset voxel IDs
        low_u_indices = torch.where(low_u_mask)[0]  # Gaussian indices

        # Group by voxel: for each unique voxel select one random member.
        unique_voxels = low_u_voxels.unique()
        for vox in unique_voxels:
            members = low_u_indices[low_u_voxels == vox]
            chosen  = members[torch.randint(0, len(members), (1,), device=device)]
            candidate_mask[chosen] = True

    # ── Stage 2: Significant Period Threshold (SPT) filter ───────────────────
    low_u_seq = (u < u_thresh).float()  # (G, T) binary

    # For each Gaussian, find the maximum run of consecutive low-uncertainty frames.
    max_run = _max_consecutive_run(low_u_seq)  # (G,)

    # Keep candidates that pass the SPT filter.
    candidate_mask = candidate_mask & (max_run >= spt)

    if candidate_mask.sum() == 0:
        # Fallback: relax SPT to 1 to avoid empty key set.
        candidate_mask = (max_run >= 1) & (u < u_thresh).any(dim=1)

    # ── Stage 3: rank by mean uncertainty, keep top key_ratio * G ────────────
    n_key = max(1, int(key_ratio * G))

    mean_u = u[candidate_mask].mean(dim=1)            # (n_candidates,)
    cand_indices = torch.where(candidate_mask)[0]     # (n_candidates,)

    n_keep = min(n_key, len(cand_indices))
    _, top_idx = mean_u.topk(n_keep, largest=False)   # lowest uncertainty first

    is_key = torch.zeros(G, dtype=torch.bool, device=device)
    is_key[cand_indices[top_idx]] = True

    return is_key


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _assign_voxels(pos: Tensor, resolution: int) -> Tensor:
    """Map (G, 3) positions to integer voxel IDs in [0, resolution^3).

    Positions outside the range of the scene are clamped into the grid.
    """
    p_min = pos.min(dim=0).values  # (3,)
    p_max = pos.max(dim=0).values  # (3,)
    span  = (p_max - p_min).clamp(min=1e-6)

    # Normalized coordinates in [0, 1].
    p_norm = (pos - p_min) / span  # (G, 3)
    # Discretize to [0, resolution-1].
    idx = (p_norm * (resolution - 1)).long().clamp(0, resolution - 1)  # (G, 3)
    # Flatten to scalar voxel ID.
    voxel_ids = idx[:, 0] * (resolution ** 2) + idx[:, 1] * resolution + idx[:, 2]
    return voxel_ids  # (G,)


def _max_consecutive_run(seq: Tensor) -> Tensor:
    """Compute the maximum run of consecutive 1s for each row of (G, T).

    Vectorized: O(G*T) with no Python loop over Gaussians.
    Returns a (G,) int64 tensor.
    """
    G, T = seq.shape
    device = seq.device

    s = seq.long()  # (G, T)  0/1

    # Running count: count[g, t] = number of consecutive 1s ending at t.
    # count[:, 0] = s[:, 0]
    # count[:, t] = (count[:, t-1] + 1) * s[:, t]
    count = torch.zeros_like(s)
    count[:, 0] = s[:, 0]
    for t in range(1, T):
        count[:, t] = (count[:, t - 1] + 1) * s[:, t]

    return count.max(dim=1).values  # (G,)
