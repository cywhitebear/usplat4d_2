"""
Non-key graph construction (eq. 8) and the full graph container.

For each non-key node i:
  1. Find its closest key node j* across all frames (sum of Mahalanobis distances):
       j* = argmin_{l ∈ V_k}  Σ_t ‖p_{i,t} - p_{l,t}‖²_{U_{i,t}+U_{l,t}}
  2. E_i = E_{j*} ∪ {j*}  (non-key node inherits j*'s key-graph edges + j* itself)
  3. Edge weights: inverse Mahalanobis at the most reliable frame, normalized.

Efficiency: O(N_nk × K × T).  With N_nk~49k, K~1k, T~100 this is ~5B ops.
We chunk non-key nodes and use a coarse Euclidean pre-filter to reduce the
inner Mahalanobis computation to the top-C Euclidean candidates (C=50).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .key_graph import KeyGraph, _batch_mahalanobis


# ──────────────────────────────────────────────────────────────────────────────
# Full graph container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class USplat4DGraph:
    """Complete spatio-temporal graph for USplat4D.

    Attributes
    ----------
    is_key          : (G,) bool
    key_graph       : KeyGraph
    nonkey_edges    : (N_nk, K+1) int64  — local indices into key_indices
    nonkey_weights  : (N_nk, K+1) float32  — normalized
    nonkey_jstar    : (N_nk,) int64  — local key index of closest key node j*
    """
    is_key: Tensor
    key_graph: KeyGraph
    nonkey_edges: Tensor    # (N_nk, k+1)  local key indices
    nonkey_weights: Tensor  # (N_nk, k+1)
    nonkey_jstar: Tensor    # (N_nk,)  j* local key index

    @property
    def key_indices(self) -> Tensor:
        return self.key_graph.key_indices

    @property
    def nonkey_indices(self) -> Tensor:
        return torch.where(~self.is_key)[0]

    @property
    def num_key(self) -> int:
        return self.key_graph.num_key_nodes

    @property
    def num_nonkey(self) -> int:
        return self.nonkey_edges.shape[0]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_nonkey_graph(
    positions: Tensor,   # (G, T, 3)
    U: Tensor,           # (G, T, 3, 3)
    u: Tensor,           # (G, T)
    is_key: Tensor,      # (G,) bool
    key_graph: KeyGraph,
    chunk_size: int = 256,
    coarse_candidates: int = 50,
) -> USplat4DGraph:
    """Build non-key graph and return a complete USplat4DGraph.

    Parameters
    ----------
    positions          : (G, T, 3)
    U                  : (G, T, 3, 3)
    u                  : (G, T)
    is_key             : (G,) bool
    key_graph          : pre-built KeyGraph
    chunk_size         : non-key nodes processed per batch
    coarse_candidates  : number of Euclidean-closest key nodes to consider before
                         running the exact Mahalanobis computation

    Returns
    -------
    USplat4DGraph
    """
    device = positions.device
    G, T, _ = positions.shape
    K = key_graph.num_key_nodes
    k = key_graph.k

    key_indices  = key_graph.key_indices   # (K,)
    nonkey_mask  = ~is_key
    nonkey_idx   = torch.where(nonkey_mask)[0]  # (N_nk,)
    N_nk = len(nonkey_idx)

    pos_key = positions[key_indices]   # (K, T, 3)
    U_key   = U[key_indices]           # (K, T, 3, 3)
    u_key   = u[key_indices]           # (K, T)

    # Mean key-node positions across time (for coarse Euclidean filter).
    pos_key_mean = pos_key.mean(dim=1)   # (K, 3)

    nonkey_jstar   = torch.zeros(N_nk, dtype=torch.long, device=device)
    nonkey_edges   = torch.zeros(N_nk, k + 1, dtype=torch.long, device=device)
    nonkey_weights = torch.zeros(N_nk, k + 1, device=device)

    for start in range(0, N_nk, chunk_size):
        end   = min(start + chunk_size, N_nk)
        chunk = nonkey_idx[start:end]  # global indices, (C,)
        C_cur = len(chunk)

        pos_nk = positions[chunk]  # (C, T, 3)
        U_nk   = U[chunk]          # (C, T, 3, 3)
        u_nk   = u[chunk]          # (C, T)

        # ── Coarse Euclidean pre-filter ───────────────────────────────────────
        pos_nk_mean = pos_nk.mean(dim=1)  # (C, 3)
        euc_dist = torch.cdist(pos_nk_mean, pos_key_mean)  # (C, K)
        n_cand = min(coarse_candidates, K)
        _, cand_idx = euc_dist.topk(n_cand, dim=1, largest=False)  # (C, n_cand)

        # ── Exact Mahalanobis distance across all frames ──────────────────────
        # D[c, l] = Σ_t (p_{c,t} - p_{l,t})^T (U_{c,t} + U_{l,t})^{-1} (p_{c,t} - p_{l,t})
        D_cand = torch.full((C_cur, n_cand), float("inf"), device=device)

        for ci in range(C_cur):
            cands = cand_idx[ci]  # (n_cand,) local key indices
            for li, l in enumerate(cands):
                l = l.item()
                diff = pos_nk[ci] - pos_key[l]  # (T, 3)
                M    = U_nk[ci] + U_key[l]      # (T, 3, 3)
                # Batched Mahalanobis over T frames.
                d_t = _batch_mahalanobis(diff, M)  # (T,)
                D_cand[ci, li] = d_t.sum()

        # j* = argmin over candidates.
        best_cand_li = D_cand.argmin(dim=1)  # (C,) index into cand_idx
        j_star_local = cand_idx[torch.arange(C_cur, device=device), best_cand_li]  # (C,) local key index

        nonkey_jstar[start:end] = j_star_local

        # ── Edge set: E_i = E_{j*} ∪ {j*} ───────────────────────────────────
        # key_graph.edges[j_star_local[c]] gives k neighbor indices (local key indices).
        j_star_edges = key_graph.edges[j_star_local]  # (C, k)
        # Append j* itself → (C, k+1)
        edges_full = torch.cat([j_star_edges, j_star_local.unsqueeze(1)], dim=1)  # (C, k+1)
        nonkey_edges[start:end] = edges_full

        # ── Edge weights: exp(-Mahal at most reliable frame), normalized ──────
        t_hat_nk = u_nk.argmin(dim=1)  # (C,) most reliable frame for non-key

        w = torch.zeros(C_cur, k + 1, device=device)
        for ci in range(C_cur):
            t_i = t_hat_nk[ci].item()
            p_i = pos_nk[ci, t_i]    # (3,)
            U_i = U_nk[ci, t_i]      # (3, 3)

            nbr_ids = edges_full[ci]  # (k+1,) local key indices
            p_nbr = pos_key[nbr_ids, t_i, :]   # (k+1, 3)
            U_nbr = U_key[nbr_ids, t_i, :, :]  # (k+1, 3, 3)

            diff_w = p_i.unsqueeze(0) - p_nbr  # (k+1, 3)
            M_w    = U_i.unsqueeze(0) + U_nbr  # (k+1, 3, 3)
            d_w    = _batch_mahalanobis(diff_w, M_w)  # (k+1,)

            ww = torch.exp(-d_w)
            ww = ww / (ww.sum() + 1e-8)
            w[ci] = ww

        nonkey_weights[start:end] = w

    return USplat4DGraph(
        is_key=is_key,
        key_graph=key_graph,
        nonkey_edges=nonkey_edges,
        nonkey_weights=nonkey_weights,
        nonkey_jstar=nonkey_jstar,
    )
