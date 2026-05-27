"""
Motion locality losses (Appendix A.2.1, eqs. S8–S13).

Applied separately to key nodes (with key_graph edges) and non-key nodes
(with nonkey_graph edges).

L^motion = λ_iso * L^iso  +  λ_rigid * L^rigid  +  λ_rot * L^rot
         + λ_vel * L^vel  +  λ_acc * L^acc

Notation:
  positions : (N, T, 3)  — N nodes, T frames
  quats     : (N, T, 4)  — wxyz
  edges     : (N, E) int — indices into N for each node's neighbors
  weights   : (N, E)     — edge weights w_{i,j}, normalized
  transforms: (N, T, 3, 4) — SE(3) transforms from model.compute_transforms
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_motion_loss(
    positions: Tensor,   # (N, T, 3)
    quats: Tensor,       # (N, T, 4)  wxyz
    transforms: Tensor,  # (N, T, 3, 4)  SE(3) per-node per-frame
    edges: Tensor,       # (N, E) int  — neighbor local indices
    weights: Tensor,     # (N, E)
    lambda_iso: float   = 1.0,
    lambda_rigid: float = 1.0,
    lambda_rot: float   = 0.01,
    lambda_vel: float   = 0.01,
    lambda_acc: float   = 0.01,
    delta: int          = 1,
) -> Tensor:
    """Total motion locality loss for a set of nodes.

    Parameters
    ----------
    positions  : (N, T, 3)
    quats      : (N, T, 4)  wxyz, unit-normalized
    transforms : (N, T, 3, 4)  SE(3) from compute_transforms (used in L_rigid)
    edges      : (N, E) indices into the N dimension (for each node, its neighbors)
    weights    : (N, E) edge weights
    delta      : time interval Δ for rigidity and rotation losses

    Returns
    -------
    Scalar loss tensor.
    """
    L_iso   = isometry_loss(positions, edges, weights)
    L_rigid = rigidity_loss(positions, transforms, edges, weights, delta)
    L_rot   = rotation_loss(quats, edges, weights, delta)
    L_vel   = velocity_loss(positions, quats)
    L_acc   = acceleration_loss(positions, quats)

    return (lambda_iso   * L_iso
          + lambda_rigid * L_rigid
          + lambda_rot   * L_rot
          + lambda_vel   * L_vel
          + lambda_acc   * L_acc)


# ──────────────────────────────────────────────────────────────────────────────
# Individual loss terms
# ──────────────────────────────────────────────────────────────────────────────

def isometry_loss(
    positions: Tensor,  # (N, T, 3)
    edges: Tensor,      # (N, E)
    weights: Tensor,    # (N, E)
) -> Tensor:
    """L^iso (eq. S8): preserve edge lengths relative to canonical (t=0).

    L^iso = (1/k|V|) Σ_t Σ_i Σ_{j ∈ knn_i} w_{ij} * |‖p_{i,o} - p_{j,o}‖ - ‖p_{i,t} - p_{j,t}‖|
    """
    N, T, _ = positions.shape
    N, E    = edges.shape

    # Canonical positions (frame 0).
    p0 = positions[:, 0, :]   # (N, 3)
    p0_nbr = p0[edges]         # (N, E, 3)

    # Per-edge canonical distances.
    d0 = (p0.unsqueeze(1) - p0_nbr).norm(dim=-1)  # (N, E)

    # Per-edge distances at each frame.
    p_t   = positions                             # (N, T, 3)
    p_nbr = positions[edges.reshape(-1)].reshape(N, E, T, 3)  # (N, E, T, 3)

    d_t = (p_t.unsqueeze(1) - p_nbr).norm(dim=-1)  # (N, E, T)

    # |d0 - d_t| weighted by w_{ij}.
    diff = (d0.unsqueeze(-1) - d_t).abs()  # (N, E, T)
    loss = (weights.unsqueeze(-1) * diff).sum(dim=(0, 1, 2))
    return loss / (E * N * T + 1e-8)


def rigidity_loss(
    positions: Tensor,   # (N, T, 3)
    transforms: Tensor,  # (N, T, 3, 4)  SE(3) per-node per-frame
    edges: Tensor,       # (N, E)
    weights: Tensor,     # (N, E)
    delta: int = 1,
) -> Tensor:
    """L^rigid,Δ (eq. S9): rigid transform consistency between neighbor pairs.

    For each pair (i, j ∈ E_i) and time t ≥ Δ:
      ‖p_{j,t} - T_{i,t} T_{i,t-Δ}^{-1} p_{j,t-Δ}‖_2

    T_{i,t} T_{i,t-Δ}^{-1} is the relative SE(3) transform of node i
    between frames t and t-Δ.
    """
    N, T, _ = positions.shape
    N, E    = edges.shape

    if T <= delta:
        return positions.new_tensor(0.0)

    T_range = range(delta, T)
    n_t = len(list(T_range))

    loss = positions.new_tensor(0.0)

    # Compute relative transforms: R_rel, t_rel such that
    # p(t) = R_rel @ p(t-Δ) + t_rel
    # R_rel = R_t @ R_{t-Δ}^T,  t_rel = t_t - R_rel @ t_{t-Δ}
    for t in T_range:
        R_t     = transforms[:, t,       :3, :3]   # (N, 3, 3)
        t_t     = transforms[:, t,       :3, 3]    # (N, 3)
        R_tm    = transforms[:, t-delta, :3, :3]   # (N, 3, 3)
        t_tm    = transforms[:, t-delta, :3, 3]    # (N, 3)

        # Relative rotation: R_rel = R_t @ R_{t-Δ}^T
        R_rel = R_t @ R_tm.transpose(-1, -2)      # (N, 3, 3)
        # Relative translation: t_rel = t_t - R_rel @ t_{t-Δ}
        t_rel = t_t - (R_rel @ t_tm.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # Apply relative transform to neighbor positions at t-Δ.
        p_nbr_tm = positions[edges.reshape(-1), t-delta].reshape(N, E, 3)  # (N, E, 3)
        p_nbr_t  = positions[edges.reshape(-1), t      ].reshape(N, E, 3)

        # Predicted: R_rel_i @ p_{j,t-Δ} + t_rel_i  for each j ∈ E_i
        # (N, E, 3) = (N, 3, 3) @ (N, E, 3, 1) + (N, 3)
        pred = (R_rel.unsqueeze(1) @ p_nbr_tm.unsqueeze(-1)).squeeze(-1) + t_rel.unsqueeze(1)

        err = (pred - p_nbr_t).norm(dim=-1)  # (N, E)
        loss = loss + (weights * err).sum()

    return loss / (E * N * n_t + 1e-8)


def rotation_loss(
    quats: Tensor,   # (N, T, 4)  wxyz
    edges: Tensor,   # (N, E)
    weights: Tensor, # (N, E)
    delta: int = 1,
) -> Tensor:
    """L^rot,Δ (eq. S10): relative rotation consistency between neighbor pairs.

    ‖q_{j,t-Δ}^{-1} q_{i,t-Δ}^{-1} · q_{i,t} q_{j,t}^{-1} - I‖

    Simplified to: for each (i,j), the relative quat difference
    q_rel_i = q_{i,t}^{-1} q_{i,t-Δ} should match q_rel_j = q_{j,t}^{-1} q_{j,t-Δ}.
    We penalize: ‖q_rel_i · q_rel_j^{-1} - identity‖
    """
    N, T, _ = quats.shape
    N, E    = edges.shape

    if T <= delta:
        return quats.new_tensor(0.0)

    T_range = range(delta, T)
    n_t = len(list(T_range))

    loss = quats.new_tensor(0.0)

    for t in T_range:
        q_i_t  = quats[:, t,       :]  # (N, 4)
        q_i_tm = quats[:, t-delta, :]  # (N, 4)
        q_j_t  = quats[edges.reshape(-1), t      ].reshape(N, E, 4)
        q_j_tm = quats[edges.reshape(-1), t-delta].reshape(N, E, 4)

        # Relative quat for node i: q_rel_i = q_i(t-Δ)^{-1} * q_i(t)
        q_rel_i = _quat_mult(_quat_inv(q_i_tm), q_i_t)   # (N, 4)
        # Relative quat for neighbor j: q_rel_j = q_j(t-Δ)^{-1} * q_j(t)
        q_rel_j = _quat_mult(
            _quat_inv(q_j_tm),                           # (N, E, 4)
            q_j_t,
        )  # (N, E, 4)

        # Discrepancy: q_rel_i * q_rel_j^{-1} should be identity [1,0,0,0].
        # Use L1 on the difference from identity.
        q_rel_i_exp = q_rel_i.unsqueeze(1).expand_as(q_rel_j)  # (N, E, 4)
        diff_q = _quat_mult(q_rel_i_exp, _quat_inv(q_rel_j))   # (N, E, 4)
        # Distance from identity: 1 - |dot(diff_q, [1,0,0,0])| = 1 - |w component|
        err = (1.0 - diff_q[..., 0].abs()).abs()  # (N, E)

        loss = loss + (weights * err).sum()

    return loss / (E * N * n_t + 1e-8)


def velocity_loss(
    positions: Tensor,  # (N, T, 3)
    quats: Tensor,      # (N, T, 4)  wxyz
) -> Tensor:
    """L^vel (eq. S11): L1 penalty on position and rotation changes.

    L^vel = Σ_{t=1}^{T-1} Σ_i ( ‖p_{i,t} - p_{i,t-1}‖_1 + ‖q_{i,t}^{-1} q_{i,t-1}‖_1 )
    """
    if positions.shape[1] < 2:
        return positions.new_tensor(0.0)

    dp = (positions[:, 1:] - positions[:, :-1]).abs().sum(dim=-1).mean()  # scalar

    q_t  = quats[:, 1:,  :]  # (N, T-1, 4)
    q_tm = quats[:, :-1, :]  # (N, T-1, 4)
    dq = _quat_mult(_quat_inv(q_tm), q_t)  # (N, T-1, 4)
    # L1 deviation from identity: penalize xyz components (should be near 0).
    dq_vel = dq[..., 1:].abs().sum(dim=-1).mean()

    return dp + dq_vel


def acceleration_loss(
    positions: Tensor,  # (N, T, 3)
    quats: Tensor,      # (N, T, 4)  wxyz
) -> Tensor:
    """L^acc (eq. S12): L1 penalty on second-order finite differences.

    L^acc = Σ_{t=2}^{T-1} Σ_i ( ‖p_{i,t-2} - 2p_{i,t-1} + p_{i,t}‖_1
                                + ‖q_{i,t-2}^{-1} q_{i,t-1} · (q_{i,t-1}^{-1} q_{i,t})^{-1}‖_1 )
    """
    if positions.shape[1] < 3:
        return positions.new_tensor(0.0)

    # Position acceleration.
    acc_p = (positions[:, :-2] - 2.0 * positions[:, 1:-1] + positions[:, 2:])  # (N, T-2, 3)
    loss_p = acc_p.abs().sum(dim=-1).mean()

    # Rotation acceleration: second-order finite difference of relative rotations.
    q0 = quats[:, :-2, :]   # p_{t-2}
    q1 = quats[:, 1:-1, :]  # p_{t-1}
    q2 = quats[:, 2:, :]    # p_{t}

    dq01 = _quat_mult(_quat_inv(q0), q1)   # relative rotation t-2 → t-1
    dq12 = _quat_mult(_quat_inv(q1), q2)   # relative rotation t-1 → t
    acc_q = _quat_mult(_quat_inv(dq01), dq12)  # second difference
    # L1 on xyz components (should be near 0).
    loss_q = acc_q[..., 1:].abs().sum(dim=-1).mean()

    return loss_p + loss_q


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion helpers (wxyz convention)
# ──────────────────────────────────────────────────────────────────────────────

def _quat_mult(q: Tensor, r: Tensor) -> Tensor:
    """Hamilton product (wxyz). q, r: (..., 4)."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_inv(q: Tensor) -> Tensor:
    """Quaternion inverse = conjugate for unit quaternions (wxyz)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
