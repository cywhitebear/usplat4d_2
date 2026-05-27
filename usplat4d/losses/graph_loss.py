"""
Graph-based losses (Section 4.3, eqs. 9 and 11).

Key node loss (eq. 9):
    L^key = Σ_t Σ_{i ∈ V_k}  ‖p_{i,t} - p°_{i,t}‖_{U^{-1}_{w,t,i}}  +  L^motion,key

Non-key node loss (eq. 11):
    L^non-key = Σ_t Σ_{i ∈ V_n}  ‖p_{i,t} - p°_{i,t}‖_{U^{-1}_{w,i}}
              + Σ_t Σ_{i ∈ V_n}  ‖p_{i,t} - p^DQB_{i,t}‖_{U^{-1}_{w,i}}
              + L^motion,non-key

where ‖v‖_A = v^T A v is the Mahalanobis norm.

For non-key nodes, U_{w,i} has no time index — it is the time-mean of U_{i,t}.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .motion_loss import compute_motion_loss


# ──────────────────────────────────────────────────────────────────────────────

def key_node_loss(
    positions_cur: Tensor,   # (K, T, 3)  current (trainable) key node positions
    positions_ref: Tensor,   # (K, T, 3)  pretrained anchor p°
    quats_cur: Tensor,       # (K, T, 4)
    transforms_cur: Tensor,  # (K, T, 3, 4)  from model.compute_transforms
    U: Tensor,               # (K, T, 3, 3)  per-frame uncertainty matrices
    key_edges: Tensor,       # (K, E)  neighbor local indices for key nodes
    key_weights: Tensor,     # (K, E)
    lambda_iso: float   = 1.0,
    lambda_rigid: float = 1.0,
    lambda_rot: float   = 0.01,
    lambda_vel: float   = 0.01,
    lambda_acc: float   = 0.01,
    delta: int          = 1,
) -> Tensor:
    """Key node loss (eq. 9).

    Returns scalar loss.
    """
    # Mahalanobis distance from current to pretrained positions.
    diff = positions_cur - positions_ref   # (K, T, 3)
    U_inv = torch.linalg.inv(U)            # (K, T, 3, 3)
    mahal = _mahalanobis_sum(diff, U_inv)  # scalar

    # Motion locality loss over key nodes.
    L_motion = compute_motion_loss(
        positions   = positions_cur,
        quats       = quats_cur,
        transforms  = transforms_cur,
        edges       = key_edges,
        weights     = key_weights,
        lambda_iso  = lambda_iso,
        lambda_rigid= lambda_rigid,
        lambda_rot  = lambda_rot,
        lambda_vel  = lambda_vel,
        lambda_acc  = lambda_acc,
        delta       = delta,
    )

    return mahal + L_motion


def nonkey_node_loss(
    positions_cur: Tensor,   # (N_nk, T, 3)
    positions_ref: Tensor,   # (N_nk, T, 3)  pretrained anchor p°
    positions_dqb: Tensor,   # (N_nk, T, 3)  DQB-interpolated
    quats_cur: Tensor,       # (N_nk, T, 4)
    transforms_cur: Tensor,  # (N_nk, T, 3, 4)
    U_agg: Tensor,           # (N_nk, 3, 3)  time-averaged uncertainty (no time index)
    nk_edges: Tensor,        # (N_nk, E)
    nk_weights: Tensor,      # (N_nk, E)
    lambda_iso: float   = 1.0,
    lambda_rigid: float = 1.0,
    lambda_rot: float   = 0.01,
    lambda_vel: float   = 0.01,
    lambda_acc: float   = 0.01,
    delta: int          = 1,
) -> Tensor:
    """Non-key node loss (eq. 11).

    Returns scalar loss.
    """
    N_nk, T, _ = positions_cur.shape
    U_inv = torch.linalg.inv(U_agg)  # (N_nk, 3, 3)

    # Broadcast U_inv over time: (N_nk, T, 3, 3)
    U_inv_t = U_inv.unsqueeze(1).expand(-1, T, -1, -1)

    # Term 1: distance to pretrained positions.
    diff_ref = positions_cur - positions_ref   # (N_nk, T, 3)
    mahal_ref = _mahalanobis_sum(diff_ref, U_inv_t)

    # Term 2: distance to DQB-interpolated positions.
    diff_dqb = positions_cur - positions_dqb   # (N_nk, T, 3)
    mahal_dqb = _mahalanobis_sum(diff_dqb, U_inv_t)

    # Motion locality loss.
    L_motion = compute_motion_loss(
        positions   = positions_cur,
        quats       = quats_cur,
        transforms  = transforms_cur,
        edges       = nk_edges,
        weights     = nk_weights,
        lambda_iso  = lambda_iso,
        lambda_rigid= lambda_rigid,
        lambda_rot  = lambda_rot,
        lambda_vel  = lambda_vel,
        lambda_acc  = lambda_acc,
        delta       = delta,
    )

    return mahal_ref + mahal_dqb + L_motion


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _mahalanobis_sum(diff: Tensor, A: Tensor) -> Tensor:
    """Compute sum of Mahalanobis norms: Σ diff^T A diff.

    diff : (..., 3)
    A    : (..., 3, 3)
    returns scalar
    """
    # v = A @ diff (last two dims)
    v = (A @ diff.unsqueeze(-1)).squeeze(-1)  # (..., 3)
    return (diff * v).sum()
