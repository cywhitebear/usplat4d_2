"""
Depth-aware uncertainty matrix (eq. 6).

Lifts scalar uncertainty u_{i,t} into an anisotropic 3×3 matrix:

    U_{i,t} = R_wc · U_c · R_wc^T
    U_c = diag(rx · u_{i,t},  ry · u_{i,t},  rz · u_{i,t})

where R_wc is the camera-to-world rotation and [rx, ry, rz] = [1, 1, 0.01]
down-weights the depth axis because monocular depth is unreliable.

Spatial normalization (Appendix B.1): positions should be pre-normalized by
the scene's spatial scale before Mahalanobis distances are computed.
This function does not rescale u — the caller should ensure u was computed
on spatially-normalized coordinates, or pass the appropriate rx/ry/rz values.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_uncertainty_matrix(
    u: Tensor,       # (G, T)  scalar uncertainties
    w2cs: Tensor,    # (T, 4, 4)  world-to-camera matrices
    rx: float = 1.0,
    ry: float = 1.0,
    rz: float = 0.01,
) -> Tensor:
    """Compute the (G, T, 3, 3) depth-aware uncertainty matrices.

    Parameters
    ----------
    u    : (G, T)  scalar uncertainty per Gaussian per frame
    w2cs : (T, 4, 4)  world-to-camera transforms
    rx, ry, rz : axis-aligned scaling factors (depth axis is rz=0.01)

    Returns
    -------
    U : (G, T, 3, 3)
    """
    G, T = u.shape
    device = u.device

    # Camera-to-world rotation: R_wc = (R_cw)^T  where R_cw = w2c[:3, :3]
    R_cw = w2cs[:, :3, :3]        # (T, 3, 3)
    R_wc = R_cw.transpose(-1, -2) # (T, 3, 3)  camera-to-world rotation

    # Axis scaling vector [rx, ry, rz] broadcast to (G, T, 3)
    r = u.new_tensor([rx, ry, rz])          # (3,)
    r = r[None, None, :] * u[:, :, None]    # (G, T, 3)

    # Build diagonal U_c for all (i, t) simultaneously.
    # U_c = diag(r_x * u, r_y * u, r_z * u) → stored as (G, T, 3, 3)
    U_c = torch.diag_embed(r)  # (G, T, 3, 3)

    # Rotate into world frame: U_{i,t} = R_wc[t] @ U_c[i,t] @ R_wc[t]^T
    # Broadcast R_wc over the G dimension.
    R = R_wc[None, :, :, :]                 # (1, T, 3, 3)
    U = R @ U_c @ R.transpose(-1, -2)       # (G, T, 3, 3)

    return U


def aggregate_uncertainty_matrix(U: Tensor) -> Tensor:
    """Time-aggregate uncertainty for non-key nodes (eq. 11, no time index).

    Returns the mean over the T dimension.

    Parameters
    ----------
    U : (G, T, 3, 3)

    Returns
    -------
    U_agg : (G, 3, 3)
    """
    return U.mean(dim=1)
