"""
USplat4D Optimizer: wraps SoM's Trainer and adds uncertainty-aware graph losses.

Training loop (400 extra epochs):
  L^total = L^rgb + L^key + L^non-key

Density control schedule:
  - First 10%  of epochs: disabled
  - Middle 70%: enabled  (SoM's normal density control)
  - Last 20%  : disabled

After each density control step the Gaussian count may change (splits / culls),
invalidating stored indices.  We handle this with a lightweight re-identification:
after any control step that changes G, we re-match key nodes to the nearest
Gaussian (in canonical position space) and rebuild positions_ref.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger as guru
from torch import Tensor

SOM_ROOT = Path("/home/ee904/Yun/shape-of-motion")
if str(SOM_ROOT) not in sys.path:
    sys.path.insert(0, str(SOM_ROOT))

from flow3d.trainer import Trainer  # noqa: E402

from .config import USplat4DConfig
from .gaussian_state import GaussianState
from .graph.dqb import dqb_interpolate
from .graph.nonkey_graph import USplat4DGraph
from .losses.graph_loss import key_node_loss, nonkey_node_loss
from .losses.perception_loss import compute_perception_loss


class USplat4DOptimizer:
    """Wraps SoM's Trainer and injects graph-based losses.

    Parameters
    ----------
    state     : GaussianState (loaded SoM model + reference positions)
    graph     : USplat4DGraph (key + non-key graph)
    U         : (G, T, 3, 3) frozen per-Gaussian per-frame uncertainty matrices
    U_agg     : (G, 3, 3) frozen time-averaged uncertainty for non-key nodes
    cfg       : USplat4DConfig
    trainer   : SoM Trainer (provides compute_losses + optimizer + scheduler)
    """

    def __init__(
        self,
        state: GaussianState,
        graph: USplat4DGraph,
        U: Tensor,      # (G, T, 3, 3)
        U_agg: Tensor,  # (G, 3, 3)
        cfg: USplat4DConfig,
        trainer: Trainer,
    ):
        self.state   = state
        self.graph   = graph
        self.cfg     = cfg
        self.trainer = trainer

        # Cache frozen uncertainty (not trainable).
        self.U     = U.detach()
        self.U_agg = U_agg.detach()

        # Pretrained reference positions p° — used in graph loss anchors.
        # Kept synchronized with model size after density control.
        self.positions_ref = state.positions_ref.detach().clone()  # (G, T, 3)
        self.quats_ref     = state.quats_ref.detach().clone()       # (G, T, 4)

        # Remember canonical means at construction time for re-identification.
        with torch.no_grad():
            self._canonical_means_at_init = (
                state.model.fg.params["means"].detach().clone()
            )  # (G, 3)

        # Save original stop_control_steps so we can restore it.
        self._orig_stop_control = trainer.optim_cfg.stop_control_steps
        self._orig_stop_densify = trainer.optim_cfg.stop_densify_steps

    # ──────────────────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────────────────

    def train_step(
        self,
        batch: dict[str, Any],
        epoch: int,
        total_epochs: int,
    ) -> dict[str, float]:
        """Single training step.

        Computes L^total = L^rgb + L^key + L^non-key, calls backward,
        steps optimizers, and (conditionally) runs density control.

        Returns a stats dict for logging.
        """
        model  = self.state.model
        device = self.state.device
        T      = self.state.num_frames
        cfg    = self.cfg

        # ── Gate density control ──────────────────────────────────────────────
        in_control_window = self._in_density_control_window(epoch, total_epochs)
        if not in_control_window:
            # Disable by setting stop thresholds to 0 (condition: step < 0 → never).
            self.trainer.optim_cfg.stop_control_steps = 0
            self.trainer.optim_cfg.stop_densify_steps = 0
        else:
            self.trainer.optim_cfg.stop_control_steps = self._orig_stop_control
            self.trainer.optim_cfg.stop_densify_steps = self._orig_stop_densify

        # ── Perception loss (L^rgb) ───────────────────────────────────────────
        model.train()
        L_rgb, stats = compute_perception_loss(self.trainer, batch)

        # ── Current per-frame positions and quats (with grad) ─────────────────
        ts = torch.arange(T, device=device)
        positions_cur, quats_cur = model.compute_poses_fg(ts)   # (G_cur, T, 3/4)
        transforms_cur = model.compute_transforms(ts)            # (G_cur, T, 3, 4)
        G_cur = positions_cur.shape[0]

        # ── Key node loss ─────────────────────────────────────────────────────
        key_local = self.graph.key_graph  # KeyGraph
        key_gidx  = self._valid_key_global_indices(G_cur)  # (K_valid,)
        K_valid   = len(key_gidx)

        L_key = positions_cur.new_tensor(0.0)
        if K_valid > 0:
            # Map global indices to local (within valid key set).
            pos_k   = positions_cur[key_gidx]          # (K_valid, T, 3)
            q_k     = quats_cur[key_gidx]               # (K_valid, T, 4)
            tf_k    = transforms_cur[key_gidx]          # (K_valid, T, 3, 4)
            pref_k  = self.positions_ref[key_gidx]      # (K_valid, T, 3)
            U_k     = self.U[key_gidx]                  # (K_valid, T, 3, 3)

            # Edges referencing positions within key_gidx.
            k_edges_l, k_weights_l = self._remap_key_edges(key_gidx, G_cur)

            if k_edges_l is not None:
                L_key = key_node_loss(
                    positions_cur   = pos_k,
                    positions_ref   = pref_k,
                    quats_cur       = q_k,
                    transforms_cur  = tf_k,
                    U               = U_k,
                    key_edges       = k_edges_l,
                    key_weights     = k_weights_l,
                    lambda_iso      = cfg.lambda_iso,
                    lambda_rigid    = cfg.lambda_rigid,
                    lambda_rot      = cfg.lambda_rot,
                    lambda_vel      = cfg.lambda_vel,
                    lambda_acc      = cfg.lambda_acc,
                    delta           = cfg.motion_delta,
                )

        # ── Non-key node loss (with DQB) ──────────────────────────────────────
        nk_gidx = self._valid_nonkey_global_indices(G_cur)   # (N_nk_valid,)
        N_nk    = len(nk_gidx)

        L_nonkey = positions_cur.new_tensor(0.0)
        if N_nk > 0 and K_valid > 0:
            pos_nk  = positions_cur[nk_gidx]              # (N_nk, T, 3)
            q_nk    = quats_cur[nk_gidx]                   # (N_nk, T, 4)
            tf_nk   = transforms_cur[nk_gidx]              # (N_nk, T, 3, 4)
            pref_nk = self.positions_ref[nk_gidx]          # (N_nk, T, 3)
            Uagg_nk = self.U_agg[nk_gidx]                  # (N_nk, 3, 3)

            # Edges for non-key nodes (already in terms of local key indices 0..K-1).
            nk_local_mask = self._nonkey_local_mask(G_cur)
            nk_edges   = self.graph.nonkey_edges[nk_local_mask]    # (N_nk, k+1)
            nk_weights = self.graph.nonkey_weights[nk_local_mask]  # (N_nk, k+1)

            # DQB: use current key node poses.
            # positions_ref for key nodes needed only for DQ construction frame.
            pos_dqb, _ = dqb_interpolate(
                positions_key  = pos_k.detach(),
                quats_key      = q_k.detach(),
                positions_ref  = pref_k,
                nonkey_edges   = nk_edges,
                nonkey_weights = nk_weights,
            )  # (N_nk, T, 3)

            L_nonkey = nonkey_node_loss(
                positions_cur   = pos_nk,
                positions_ref   = pref_nk,
                positions_dqb   = pos_dqb,
                quats_cur       = q_nk,
                transforms_cur  = tf_nk,
                U_agg           = Uagg_nk,
                nk_edges        = nk_edges,
                nk_weights      = nk_weights,
                lambda_iso      = cfg.lambda_iso,
                lambda_rigid    = cfg.lambda_rigid,
                lambda_rot      = cfg.lambda_rot,
                lambda_vel      = cfg.lambda_vel,
                lambda_acc      = cfg.lambda_acc,
                delta           = cfg.motion_delta,
            )

        # ── Total loss + backward ─────────────────────────────────────────────
        L_total = L_rgb + L_key + L_nonkey

        if L_total.isnan():
            guru.warning(f"NaN loss at epoch {epoch}! rgb={L_rgb.item():.4f} "
                         f"key={L_key.item():.4f} nonkey={L_nonkey.item():.4f}")
            L_total = L_rgb  # fall back to RGB-only to avoid NaN cascade

        L_total.backward()

        for opt in self.trainer.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in self.trainer.scheduler.values():
            sched.step()

        self.trainer.global_step += 1

        # ── Density control + positions_ref sync ─────────────────────────────
        G_before = G_cur
        self.trainer.run_control_steps()
        G_after = model.num_fg_gaussians
        if G_after != G_before:
            self._sync_after_density_control(G_before, G_after)

        stats.update({
            "usplat4d/L_key":    L_key.item(),
            "usplat4d/L_nonkey": L_nonkey.item(),
            "usplat4d/L_rgb":    L_rgb.item(),
            "usplat4d/L_total":  L_total.item(),
        })
        return stats

    # ──────────────────────────────────────────────────────────────────────────
    # Index helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _valid_key_global_indices(self, G_cur: int) -> Tensor:
        """Return key-node global indices that are still within [0, G_cur)."""
        ki = self.graph.key_graph.key_indices
        return ki[ki < G_cur]

    def _valid_nonkey_global_indices(self, G_cur: int) -> Tensor:
        """Return non-key global indices that are still within [0, G_cur)."""
        nki = self.graph.nonkey_indices
        return nki[nki < G_cur]

    def _nonkey_local_mask(self, G_cur: int) -> Tensor:
        """Bool mask over non-key nodes that still have valid global indices."""
        nki = self.graph.nonkey_indices
        return nki < G_cur

    def _remap_key_edges(
        self, valid_key_gidx: Tensor, G_cur: int
    ) -> tuple[Tensor | None, Tensor | None]:
        """Remap stored key-graph edges to use indices into valid_key_gidx.

        Returns (edges, weights) where edge values are indices 0..K_valid-1,
        or (None, None) if there are too few key nodes for kNN.
        """
        K_valid = len(valid_key_gidx)
        k = self.graph.key_graph.k
        if K_valid <= k:
            return None, None

        # Original edges are local indices 0..K-1 (into key_graph.key_indices).
        # We need to find which of those correspond to valid_key_gidx.
        ki_all = self.graph.key_graph.key_indices  # (K,)
        # Map: local old index → local new index (-1 if invalid).
        valid_set = set(valid_key_gidx.tolist())
        old_to_new = {}
        new_i = 0
        for old_i, gi in enumerate(ki_all.tolist()):
            if gi in valid_set:
                old_to_new[old_i] = new_i
                new_i += 1

        # Remap edges.
        orig_edges   = self.graph.key_graph.edges    # (K, k)
        orig_weights = self.graph.key_graph.weights  # (K, k)

        # Only keep rows for valid key nodes.
        valid_local = [
            old_i for old_i, gi in enumerate(ki_all.tolist()) if gi in valid_set
        ]
        if len(valid_local) == 0:
            return None, None

        valid_local_t = torch.tensor(valid_local, device=valid_key_gidx.device)
        edges_sub   = orig_edges[valid_local_t]    # (K_valid, k)
        weights_sub = orig_weights[valid_local_t]  # (K_valid, k)

        # Remap neighbor indices.
        edges_new = torch.zeros_like(edges_sub)
        for r in range(K_valid):
            for c in range(k):
                old_j = edges_sub[r, c].item()
                new_j = old_to_new.get(old_j, 0)  # fallback to 0 if missing
                edges_new[r, c] = new_j

        return edges_new, weights_sub

    # ──────────────────────────────────────────────────────────────────────────
    # Density control synchronization
    # ──────────────────────────────────────────────────────────────────────────

    def _sync_after_density_control(self, G_before: int, G_after: int):
        """Update positions_ref and graph indices after density control changes G.

        Strategy:
        - Recompute current positions (no grad) → these become the new reference
          for any Gaussians that didn't previously have a reference entry.
        - If G_after < G_before (culling): some Gaussians were removed and the
          array was compacted.  Re-identify key nodes using nearest-neighbor
          matching in canonical position space.
        - If G_after > G_before (densification): new Gaussians are appended;
          extend positions_ref with their current poses.
        """
        model  = self.state.model
        device = self.state.device
        T      = self.state.num_frames

        with torch.no_grad():
            ts  = torch.arange(T, device=device)
            pos_new, q_new = model.compute_poses_fg(ts)  # (G_after, T, 3/4)

        if G_after > G_before:
            # Gaussians were added — extend positions_ref.
            extra = pos_new[G_before:]  # (G_after - G_before, T, 3)
            self.positions_ref = torch.cat([self.positions_ref, extra], dim=0)
            extra_q = q_new[G_before:]
            self.quats_ref = torch.cat([self.quats_ref, extra_q], dim=0)
            # Uncertainty: extend U and U_agg with zeros (new Gaussians are
            # uncertain — they won't participate in graph losses beyond G_before).
            extra_U    = torch.zeros(G_after - G_before, T, 3, 3, device=device)
            extra_Uagg = torch.zeros(G_after - G_before, 3, 3, device=device)
            self.U     = torch.cat([self.U,     extra_U   ], dim=0)
            self.U_agg = torch.cat([self.U_agg, extra_Uagg], dim=0)
            guru.info(f"Density control: added {G_after - G_before} Gaussians "
                      f"({G_before} → {G_after})")

        elif G_after < G_before:
            # Gaussians were culled — need to re-identify surviving entries.
            # Use nearest-neighbour matching in canonical position space.
            canon_after  = model.fg.params["means"].detach()       # (G_after, 3)
            canon_before = self._canonical_means_at_init[:G_before] # (G_before, 3)

            # For each remaining Gaussian, find its best match in the old set.
            # This is O(G_after * G_before) — bounded at ~2.5B ops for 50k×50k,
            # but in practice G is much smaller.  Chunk for memory.
            mapping = _nearest_neighbor_remap(canon_after, canon_before)  # (G_after,) int

            self.positions_ref = self.positions_ref[mapping]
            self.quats_ref     = self.quats_ref[mapping]
            self.U             = self.U[mapping]
            self.U_agg         = self.U_agg[mapping]

            # Remap graph key_indices to new space.
            new_key_indices = mapping.argsort()[self.graph.key_graph.key_indices]
            # Filter out any that map to invalid (beyond G_after).
            valid = new_key_indices < G_after
            self.graph.key_graph.key_indices = new_key_indices[valid]

            guru.info(f"Density control: culled to {G_after} Gaussians "
                      f"({G_before} → {G_after})")

        # Update the cached canonical means for future density control steps.
        with torch.no_grad():
            self._canonical_means_at_init = model.fg.params["means"].detach().clone()

    # ──────────────────────────────────────────────────────────────────────────
    # Density control window
    # ──────────────────────────────────────────────────────────────────────────

    def _in_density_control_window(self, epoch: int, total_epochs: int) -> bool:
        frac = epoch / max(total_epochs, 1)
        warmup_end   = self.cfg.density_control_warmup_frac
        cooldown_start = 1.0 - self.cfg.density_control_cooldown_frac
        return warmup_end <= frac < cooldown_start

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint save / load
    # ──────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str):
        self.trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.state.device)
        if "optimizers" in ckpt:
            self.trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            self.trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        self.trainer.global_step = ckpt.get("global_step", 0)


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_neighbor_remap(
    query: Tensor,    # (N, 3)  new canonical positions
    source: Tensor,   # (M, 3)  old canonical positions
    chunk: int = 4096,
) -> Tensor:
    """For each query point, return the index of the nearest source point.

    Returns a (N,) int64 tensor of source indices, one per query point.
    """
    N = query.shape[0]
    mapping = torch.zeros(N, dtype=torch.long, device=query.device)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        q_chunk = query[start:end]              # (C, 3)
        dists   = torch.cdist(q_chunk, source)  # (C, M)
        mapping[start:end] = dists.argmin(dim=1)
    return mapping
