"""
Key graph construction: Uncertainty-Aware kNN (UA-kNN, eq. 7).

For each key node i:
  1. Find its most reliable frame: t̂ = argmin_t u_{i,t}
  2. At frame t̂, compute Mahalanobis distance to all other key nodes j:
       d(i,j) = (p_{i,t̂} - p_{j,t̂})^T (U_{i,t̂} + U_{j,t̂})^{-1} (p_{i,t̂} - p_{j,t̂})
  3. Select k nearest neighbors → edges E_i (directed graph).
  4. Edge weights: w_{ij} = exp(-d_{ij}), normalized to sum to 1 over j ∈ E_i.

The graph is fixed after construction and never modified during optimization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class KeyGraph:
    """Stores the key-node directed graph as adjacency lists.

    Attributes
    ----------
    key_indices : (K,) int64  — global Gaussian indices for key nodes
    edges       : (K, k) int64 — for each key node, indices into key_indices
                  of its k nearest neighbors
    weights     : (K, k) float32 — normalized edge weights
    """

    def __init__(
        self,
        key_indices: Tensor,  # (K,)
        edges: Tensor,        # (K, k)  indices into key_indices
        weights: Tensor,      # (K, k)
    ):
        self.key_indices = key_indices
        self.edges = edges
        self.weights = weights

    @property
    def num_key_nodes(self) -> int:
        return len(self.key_indices)

    @property
    def k(self) -> int:
        return self.edges.shape[1]

    def neighbor_global_indices(self, key_local_idx: int) -> Tensor:
        """Return global Gaussian indices of neighbors of key node `key_local_idx`."""
        return self.key_indices[self.edges[key_local_idx]]


def build_key_graph(
    positions: Tensor,  # (G, T, 3)
    U: Tensor,          # (G, T, 3, 3)
    u: Tensor,          # (G, T)
    is_key: Tensor,     # (G,) bool
    k: int = 8,
) -> KeyGraph:
    """Build the UA-kNN key graph.

    Parameters
    ----------
    positions : (G, T, 3)
    U         : (G, T, 3, 3) depth-aware uncertainty matrices
    u         : (G, T) scalar uncertainties
    is_key    : (G,) bool mask for key nodes
    k         : number of nearest neighbors per key node

    Returns
    -------
    KeyGraph
    """
    device = positions.device
    key_indices = torch.where(is_key)[0]  # (K,)
    K = len(key_indices)

    pos_key = positions[key_indices]  # (K, T, 3)
    U_key   = U[key_indices]          # (K, T, 3, 3)
    u_key   = u[key_indices]          # (K, T)

    # For each key node, find its most reliable frame.
    t_hat = u_key.argmin(dim=1)  # (K,)

    # For each key node i, compute Mahalanobis distances to all other key nodes j
    # at frame t_hat[i].
    # We batch over i; for each i, gather the positions/U at t_hat[i].
    k_actual = min(k, K - 1)
    edges   = torch.zeros(K, k_actual, dtype=torch.long, device=device)
    weights = torch.zeros(K, k_actual, device=device)

    # Precompute all (K, T, 3) and (K, T, 3, 3) slices are already available.
    for i in range(K):
        t_i = t_hat[i].item()
        p_i = pos_key[i, t_i]   # (3,)
        U_i = U_key[i, t_i]     # (3, 3)

        # Positions and U of all other key nodes at frame t_i.
        p_j = pos_key[:, t_i, :]  # (K, 3)
        U_j = U_key[:, t_i, :, :] # (K, 3, 3)

        # Pairwise covariance: U_i + U_j  (broadcast U_i over K)
        M = U_i.unsqueeze(0) + U_j  # (K, 3, 3)

        # Mahalanobis distance: diff^T M^{-1} diff
        diff = p_i.unsqueeze(0) - p_j  # (K, 3)
        dist = _batch_mahalanobis(diff, M)  # (K,)

        # Exclude self.
        dist[i] = float("inf")

        # Top-k smallest distances.
        _, nn_idx = dist.topk(k_actual, largest=False)  # (k,) indices into K

        edges[i] = nn_idx

        # Weights: exp(-dist), normalized.
        w = torch.exp(-dist[nn_idx])
        w = w / (w.sum() + 1e-8)
        weights[i] = w

    return KeyGraph(key_indices, edges, weights)


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _batch_mahalanobis(diff: Tensor, M: Tensor) -> Tensor:
    """Compute Mahalanobis distances for a batch of (diff, M) pairs.

    Parameters
    ----------
    diff : (N, 3)
    M    : (N, 3, 3)  positive-definite covariance matrices

    Returns
    -------
    dist : (N,)  scalar distances  diff^T M^{-1} diff
    """
    # Use torch.linalg.solve for numerical stability: solve M @ x = diff → x = M^{-1} diff
    # diff: (N, 3) → unsqueeze to (N, 3, 1) for batched solve
    x = torch.linalg.solve(M, diff.unsqueeze(-1))  # (N, 3, 1)
    dist = (diff.unsqueeze(-2) @ x).squeeze(-1).squeeze(-1)  # (N,)
    return dist.clamp(min=0.0)  # numerical guard
