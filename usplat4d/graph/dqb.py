"""
Dual Quaternion Blending (DQB) — Kavan et al., 2007 (eq. 10).

Given key node SE(3) transforms {T_{j,t}} and normalized edge weights {w_{ij}},
the interpolated pose of non-key node i at time t is:

    (p^DQB_{i,t}, q^DQB_{i,t}) = DQB{ (w_{ij}, T_{j,t}) }_{j ∈ E_i}

A dual quaternion encodes a rigid SE(3) transform as a pair (q_r, q_d) where:
  q_r  — rotation quaternion (wxyz, unit norm)
  q_d  = 0.5 * t_quat * q_r   where t_quat = [0, tx, ty, tz]

Blending procedure:
  1. Flip sign of q_r to ensure all are in the same hemisphere (antipodal fix).
  2. Linearly blend both parts: q̄_r = Σ w_j * q_r_j,  q̄_d = Σ w_j * q_d_j.
  3. Normalize: q_r = q̄_r / ‖q̄_r‖,  q_d = q̄_d / ‖q̄_r‖.
  4. Extract translation: t = 2 * q_d * q_r*.

All quaternions use the wxyz convention (matching SoM / roma).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def dqb_interpolate(
    positions_key: Tensor,  # (K, T, 3)  current key node positions
    quats_key: Tensor,      # (K, T, 4)  current key node quaternions (wxyz)
    positions_ref: Tensor,  # (K, T, 3)  pretrained key node positions (canonical reference)
    nonkey_edges: Tensor,   # (N_nk, K+1)  indices into K for each non-key node's neighbors
    nonkey_weights: Tensor, # (N_nk, K+1)  normalized edge weights
) -> tuple[Tensor, Tensor]:
    """Compute DQB-interpolated positions and quats for all non-key nodes.

    The SE(3) transform T_{j,t} for key node j at time t is defined as the
    transform that maps the reference (pretrained) position to the current
    position.  Specifically:
        T_{j,t}: p_ref → p_cur  via  p_cur = R_j * p_ref + t_j
    Here we approximate the rigid transform per-key-node as:
        rotation = quats_key[j, t]  (already the world-frame orientation)
        translation = positions_key[j, t]  (world-frame position)
    and DQB blends these transforms to produce the non-key node's pose.

    Parameters
    ----------
    positions_key   : (K, T, 3)
    quats_key       : (K, T, 4)  wxyz unit quaternions
    positions_ref   : (K, T, 3)  pretrained positions (for DQ construction)
    nonkey_edges    : (N_nk, E)  indices into key nodes (local index 0..K-1)
    nonkey_weights  : (N_nk, E)  weights summing to 1 along dim 1

    Returns
    -------
    pos_dqb  : (N_nk, T, 3)
    quat_dqb : (N_nk, T, 4)  wxyz
    """
    N_nk, E = nonkey_edges.shape
    K, T, _ = positions_key.shape
    device = positions_key.device

    # Gather neighbor key-node poses: (N_nk, E, T, 3/4)
    # edges: (N_nk, E) → gather from (K, T, 3)
    edges_exp = nonkey_edges[:, :, None, None].expand(-1, -1, T, 3)  # (N_nk, E, T, 3)
    p_nbr = torch.gather(
        positions_key.unsqueeze(0).expand(N_nk, -1, -1, -1),  # (N_nk, K, T, 3)
        dim=1,
        index=edges_exp,
    )  # (N_nk, E, T, 3)

    edges_exp_q = nonkey_edges[:, :, None, None].expand(-1, -1, T, 4)
    q_nbr = torch.gather(
        quats_key.unsqueeze(0).expand(N_nk, -1, -1, -1),      # (N_nk, K, T, 4)
        dim=1,
        index=edges_exp_q,
    )  # (N_nk, E, T, 4)

    # Build dual quaternions for each (non-key, neighbor, frame).
    # q_r = q_nbr (rotation part)
    # q_d = 0.5 * [0, t] * q_r  where t = positions_key (translation)
    q_r = q_nbr                                   # (N_nk, E, T, 4)
    t   = p_nbr                                    # (N_nk, E, T, 3)
    q_d = _translation_to_dual(t, q_r)            # (N_nk, E, T, 4)

    # Blend: w shape (N_nk, E) → expand to (N_nk, E, T, 4)
    w = nonkey_weights[:, :, None, None].expand_as(q_r)  # (N_nk, E, T, 4)

    # Antipodal fix: flip sign so all q_r are in the same hemisphere as q_r[..., 0, :, :].
    ref_q = q_r[:, 0:1, :, :]  # (N_nk, 1, T, 4)  reference (first neighbor)
    dot = (q_r * ref_q).sum(dim=-1, keepdim=True)  # (N_nk, E, T, 1)
    sign = dot.sign().clamp(min=0) * 2 - 1  # +1 if same hemisphere, -1 otherwise
    # When dot == 0 exactly, keep the original sign (treat as +1).
    sign = torch.where(dot == 0, torch.ones_like(sign), sign)
    q_r = q_r * sign
    q_d = q_d * sign

    blended_r = (w * q_r).sum(dim=1)  # (N_nk, T, 4)
    blended_d = (w * q_d).sum(dim=1)  # (N_nk, T, 4)

    # Normalize.
    norm = blended_r.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (N_nk, T, 1)
    q_r_norm = blended_r / norm
    q_d_norm = blended_d / norm

    # Extract translation from dual quaternion: t = 2 * q_d * q_r*
    pos_dqb  = _dual_to_translation(q_r_norm, q_d_norm)  # (N_nk, T, 3)
    quat_dqb = q_r_norm                                   # (N_nk, T, 4)

    return pos_dqb, quat_dqb


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion / dual-quaternion math  (wxyz convention)
# ──────────────────────────────────────────────────────────────────────────────

def _quat_mult(q: Tensor, r: Tensor) -> Tensor:
    """Hamilton product of two quaternions (wxyz).

    q, r : (..., 4)
    returns (..., 4)
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_conj(q: Tensor) -> Tensor:
    """Quaternion conjugate (wxyz): (w, x, y, z) → (w, -x, -y, -z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _translation_to_dual(t: Tensor, q_r: Tensor) -> Tensor:
    """Build the dual part from translation and rotation.

    q_d = 0.5 * [0, t] * q_r

    t   : (..., 3)
    q_r : (..., 4)  wxyz rotation
    returns (..., 4)
    """
    # Pure quaternion for translation: [0, tx, ty, tz]
    t_quat = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)  # (..., 4)
    return 0.5 * _quat_mult(t_quat, q_r)


def _dual_to_translation(q_r: Tensor, q_d: Tensor) -> Tensor:
    """Extract translation from a unit dual quaternion.

    t = 2 * q_d * q_r*

    q_r, q_d : (..., 4)
    returns (..., 3)
    """
    t_quat = 2.0 * _quat_mult(q_d, _quat_conj(q_r))  # (..., 4)
    return t_quat[..., 1:]  # drop w component → (..., 3)
