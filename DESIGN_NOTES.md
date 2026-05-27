# USplat4D Implementation Design Notes

## 1. SoM Gaussian Parameterization

SoM stores motion via **canonical positions + motion bases**, not per-frame dense tensors.
Per-frame positions are computed on-the-fly: `model.compute_poses_fg(ts)` → `(G, T, 3)`.

`gaussian_state.py` precomputes `(G, T, 3)` once at load time (no grad, for graph
construction only).  During training, positions are re-computed with gradient flow so
that backprop reaches `motion_bases` and `motion_coefs`.

## 2. Uncertainty Estimation: Chunked Rendering

`gsplat.rasterization()` does not expose per-Gaussian per-pixel blending weights
`v^h_i = T^h_i · α_i` needed for eq. 3.

**Solution**: chunked rendering.  For each chunk of C Gaussians:
- Set those Gaussians' colors to a `C × C` identity matrix.
- All other Gaussians' colors → zero.
- Render → output `(1, H, W, C)`: channel `c` at pixel `h` = `v^h_{i_c}`.
- `σ²_{i_c} = (Σ_h rendered[h,c]²)⁻¹`.

Cost: `⌈N/C⌉` no-grad renders per frame.  With C=64, N≈50k, T≈100: ~78k renders,
< 2 minutes total on a single GPU.

## 3. Key Graph: Inner Loop Over K Key Nodes

`key_graph.py::build_key_graph` iterates over K key nodes and for each does a batched
Mahalanobis computation over all K candidates.  This is `O(K²)` with K ≈ 1000 → acceptable.
No further optimization needed unless K grows significantly.

## 4. Non-key Graph: Euclidean Pre-filter

The exact non-key graph build is `O(N_nk × K × T)` ≈ 5B ops.  To reduce cost:

1. Compute mean-position Euclidean distances `(N_nk, K)` → keep top-50 candidates.
2. Run exact Mahalanobis only over those 50 candidates.

Reduces inner loop 20× with negligible accuracy loss (key nodes ≥ 50 apart in
Euclidean space are unlikely to be the Mahalanobis nearest neighbor either).

## 5. DQB Uses Current Key Node Transforms

During training, `dqb_interpolate` is called each step with **current** (trainable)
key node positions and quaternions — not frozen pretrained values.

This means DQB targets update as key nodes optimize, propagating motion corrections
to non-key nodes continuously.  The graph **structure** (edges, weights) is fixed; only
the **transforms** being blended update.

## 6. Non-key Uncertainty: Time-Averaged U_{w,i}

Eq. 11 uses `U_{w,i}` with no time index.  We use `U[i].mean(dim=0)` (mean over T).
This is the safest interpretation: a single covariance matrix summarizing the node's
overall reliability across the sequence.

## 7. Density Control + Graph Index Synchronization

Density control (Gaussian splits/culls) invalidates stored global Gaussian indices.
Strategy after each control step:

- **Splits (G increases)**: new Gaussians appended at end; extend `positions_ref`,
  `U`, `U_agg` with the new Gaussians' current poses / zero uncertainties.  New
  Gaussians beyond `G_at_init` receive no graph loss (L^rgb only).
- **Culls (G decreases)**: indices shift.  Re-identify surviving Gaussians via
  nearest-neighbour matching in canonical position space, then remap `positions_ref`,
  `U`, `U_agg`, and `key_graph.key_indices`.

Key nodes are specifically chosen for stability (low uncertainty, high visibility), so
they are rarely culled in practice.

## 8. Perception Loss Wrapper

`losses/perception_loss.py` calls SoM's `Trainer.compute_losses(batch)` directly and
returns `(loss, stats)`.  This includes all of SoM's internal regularizers (smoothness,
scale variance, z-acceleration) — we do **not** strip them.  USplat4D's motion losses
add on top; they are not a replacement.

## 9. Density Control Gating

Paper: disable first 10% and last 20% of training.  Implementation: set
`optim_cfg.stop_control_steps = 0` outside the window (the condition
`global_step < stop_control_steps` is False when threshold is 0), restore inside.

## 10. Open Tuning Knobs

| Parameter | Default | Notes |
|---|---|---|
| `unc_chunk_size` | 64 | Larger → fewer renders but more VRAM |
| `voxel_resolution` | 50 | Increase for denser key coverage in large scenes |
| `k_neighbors` | 8 | Not specified in paper; ablate if graph is too sparse/dense |
| `coarse_candidates` | 50 | Non-key pre-filter; reduce if non-key graph build is slow |
| `phi` | 1e6 | "Large constant" for unconverged Gaussians; ablate if needed |
