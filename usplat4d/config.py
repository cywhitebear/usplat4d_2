from dataclasses import dataclass, field


@dataclass
class USplat4DConfig:
    # ── Uncertainty estimation ──────────────────────────────────────────────
    eta_c: float = 0.5        # color-error convergence threshold, normalized [0,1]
    phi: float = 1e6          # large constant assigned to unconverged Gaussians
    rx: float = 1.0           # x-axis uncertainty scale (image-plane)
    ry: float = 1.0           # y-axis uncertainty scale (image-plane)
    rz: float = 0.01          # z-axis (depth) uncertainty scale — strongly down-weighted
    unc_chunk_size: int = 64  # Gaussians per render chunk for blending-weight computation

    # ── Key node selection ─────────────────────────────────────────────────
    key_ratio: float = 0.02   # top 2% lowest-uncertainty Gaussians become key nodes
    spt: int = 5              # Significant Period Threshold (min consecutive low-u frames)
    voxel_resolution: int = 50  # voxels per spatial axis for per-frame candidate sampling

    # ── Graph construction ─────────────────────────────────────────────────
    k_neighbors: int = 8      # kNN edges per key node (UA-kNN)

    # ── Motion loss weights (eq. S13) ──────────────────────────────────────
    lambda_iso: float = 1.0
    lambda_rigid: float = 1.0
    lambda_rot: float = 0.01
    lambda_vel: float = 0.01
    lambda_acc: float = 0.01
    motion_delta: int = 1     # time interval Δ used in rigidity / rotation losses

    # ── Training schedule ──────────────────────────────────────────────────
    extra_epochs: int = 400
    batch_size: int = 8
    # Density control is disabled in the first and last fraction of training
    density_control_warmup_frac: float = 0.10   # disable for first 10%
    density_control_cooldown_frac: float = 0.20  # disable for last 20%
