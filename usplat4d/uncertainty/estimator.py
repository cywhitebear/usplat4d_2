"""
Dynamic uncertainty estimation (Section 4.1, eqs. 3–5).

For each Gaussian i and frame t, compute the scalar uncertainty u_{i,t}.

  u_{i,t} = 1_{i,t} * σ²_{i,t}  +  (1 - 1_{i,t}) * φ         (eq. 5)

where
  σ²_{i,t} = (Σ_{h ∈ Ω_{i,t}}  (T^h_{i,t} · α_i)²)⁻¹         (eq. 3)
  1_{i,t}  = Π_{h ∈ Ω_{i,t}}  𝟙[‖C̃^h_t - C^h_t‖₁ < η_c]     (eq. 4)

gsplat's rasterization() does not expose per-Gaussian per-pixel blending weights
directly.  We recover them via *chunked rendering*:

  For each chunk of C Gaussians in the scene:
    1. Set those C Gaussians' colors to the CxC identity matrix (unique "tag" per Gaussian).
    2. Set all other Gaussians' colors to zero.
    3. Render → output (1, H, W, C).  Channel c at pixel h = v^h_{i_c}.
    4. Accumulate squared contributions: Σ_h rendered_{h,c}^2 → σ²_{i_c}^{-1}.
    5. Check convergence: a Gaussian's indicator = product of converged flags at
       pixels where its rendered contribution > 0.

This requires ceil(N / C) renders per frame.  With C=64 and N~50k → ~782 renders/frame,
all under torch.no_grad(), typically < 2 minutes for a full sequence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

SOM_ROOT = Path("/home/ee904/Yun/shape-of-motion")
if str(SOM_ROOT) not in sys.path:
    sys.path.insert(0, str(SOM_ROOT))

from flow3d.scene_model import SceneModel  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_scalar_uncertainty(
    model: SceneModel,
    gt_images: list[Tensor],   # list of T tensors, each (H, W, 3) in [0, 1]
    img_wh: tuple[int, int],   # (W, H)
    eta_c: float = 0.5,
    phi: float = 1e6,
    chunk_size: int = 64,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Tensor:
    """Compute per-Gaussian per-frame scalar uncertainty.

    Parameters
    ----------
    model      : pretrained SoM SceneModel (eval mode, no grad needed)
    gt_images  : T ground-truth frames, each (H, W, 3) float32 in [0, 1]
    img_wh     : (W, H) pixel dimensions
    eta_c      : color-error convergence threshold in [0, 1]
    phi        : large constant for unconverged Gaussians
    chunk_size : number of Gaussians per render chunk (C in the doc above)
    device     : compute device; inferred from model if None
    verbose    : print progress

    Returns
    -------
    u : Tensor  (G, T)  scalar uncertainty per Gaussian per frame
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    G = model.num_fg_gaussians
    T = model.num_frames
    W, H = img_wh

    # Pre-extract fixed Gaussian attributes (no grad needed for uncertainty).
    with torch.no_grad():
        opacities = model.fg.get_opacities()   # (G,)  in (0,1)
        scales    = model.fg.get_scales()      # (G, 3)
        # Canonical colors (used only for convergence check render).
        colors    = model.fg.get_colors()      # (G, 3)

    # Output tensors.
    sigma2_inv = torch.zeros(G, T, device=device)   # Σ_h (v^h_i)²
    indicator  = torch.ones(G, T, device=device)    # product of convergence flags

    n_chunks = math.ceil(G / chunk_size)

    for t in range(T):
        if verbose and t % 10 == 0:
            print(f"  [uncertainty] frame {t}/{T}", flush=True)

        w2c = model.w2cs[t : t + 1].to(device)   # (1, 4, 4)
        K   = model.Ks[t : t + 1].to(device)     # (1, 3, 3)

        # ── Step 1: render RGB and compute per-pixel convergence ─────────────
        with torch.no_grad():
            ts_t = torch.tensor([t], device=device)
            means_t, quats_t = model.compute_poses_fg(ts_t)  # (G, 1, 3/4)
            means_t  = means_t[:, 0].contiguous()   # (G, 3)
            quats_t  = quats_t[:, 0].contiguous()   # (G, 4)

            render_out = _render_rgb(
                means_t, quats_t, scales, opacities, colors, w2c, K, W, H, device
            )  # (1, H, W, 3)

        gt_t = gt_images[t].to(device)  # (H, W, 3)
        l1_err = (render_out[0] - gt_t).abs().mean(dim=-1)  # (H, W) in [0,1]
        converged_map = (l1_err < eta_c)  # (H, W) bool

        # ── Step 2: chunked rendering for blending-weight accumulation ───────
        for c_start in range(0, G, chunk_size):
            c_end   = min(c_start + chunk_size, G)
            C_cur   = c_end - c_start          # actual chunk size
            chunk_i = torch.arange(c_start, c_end, device=device)

            # Build per-Gaussian "tag" colors:
            # Only the C_cur Gaussians in this chunk have non-zero colors;
            # each gets an identity-row of dimension C_cur.
            # All other Gaussians get zero colors.
            tag_colors = torch.zeros(G, C_cur, device=device)
            tag_colors[chunk_i] = torch.eye(C_cur, device=device)

            with torch.no_grad():
                rendered_chunk, _, _ = rasterization(
                    means=means_t,
                    quats=quats_t,
                    scales=scales,
                    opacities=opacities,
                    colors=tag_colors,
                    backgrounds=torch.zeros(1, C_cur, device=device),
                    viewmats=w2c,
                    Ks=K,
                    width=W,
                    height=H,
                    packed=False,
                    render_mode="RGB",
                )
                # rendered_chunk: (1, H, W, C_cur)
                rendered_chunk = rendered_chunk[0]  # (H, W, C_cur)

            # Σ_h (v^h_i)²  for each i in chunk
            sq_sum = (rendered_chunk ** 2).sum(dim=(0, 1))  # (C_cur,)
            sigma2_inv[c_start:c_end, t] = sq_sum

            # Convergence indicator: product over pixels where v^h_i > 0.
            # A pixel contributes to Gaussian i if rendered_chunk[h,w,c] > 0.
            # 1_{i,t} = 0  if ANY covered pixel is not converged.
            # We compute:  1 - max over covered pixels of (1 - converged_map)
            # i.e. indicator = 0 if any covered pixel is unconverged.
            eps = 1e-6
            covered   = rendered_chunk > eps  # (H, W, C_cur) bool
            uncovered_or_converged = ~covered | converged_map.unsqueeze(-1)
            # indicator[i, t] = 0 if any covered pixel is unconverged
            all_converged = uncovered_or_converged.all(dim=(0, 1))  # (C_cur,) bool
            # Multiply existing indicator (in case of previous chunks, though
            # each Gaussian only appears once; this is safety).
            indicator[c_start:c_end, t] *= all_converged.float()

    # ── Combine into final uncertainty (eq. 5) ────────────────────────────────
    # σ²_{i,t} = 1 / Σ_h (v^h_i)²  ;  clamp denominator to avoid division by zero.
    sigma2 = 1.0 / (sigma2_inv + 1e-10)  # (G, T)
    # Invisible Gaussians (no pixel contribution) have sigma2_inv=0 → sigma2=1e10.
    # Their convergence indicator is vacuously 1 (no covered pixels → all converged).
    # Force indicator=0 so they receive phi instead of 1/eps.
    indicator[sigma2_inv == 0] = 0
    u = indicator * sigma2 + (1.0 - indicator) * phi

    return u


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _render_rgb(
    means: Tensor,     # (G, 3)
    quats: Tensor,     # (G, 4)
    scales: Tensor,    # (G, 3)
    opacities: Tensor, # (G,)
    colors: Tensor,    # (G, 3)
    w2c: Tensor,       # (1, 4, 4)
    K: Tensor,         # (1, 3, 3)
    W: int,
    H: int,
    device: torch.device,
) -> Tensor:
    """Single RGB forward render, returns (1, H, W, 3)."""
    render_out, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        backgrounds=torch.ones(1, 3, device=device),
        viewmats=w2c,
        Ks=K,
        width=W,
        height=H,
        packed=False,
        render_mode="RGB",
    )
    return render_out  # (1, H, W, 3)
