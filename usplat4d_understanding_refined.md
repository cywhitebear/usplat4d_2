# Refined Understanding of USplat4D

## Pipeline Overview

USplat4D is a **post-processing optimizer** that takes the output of a pretrained dynamic Gaussian Splatting model (e.g., SoM or MoSca) as initialization, and refines it with uncertainty-aware graph-based optimization. It does **not** modify the base model's training loop; it runs afterward with additional iterations.

The five main steps are:

1. Dynamic Uncertainty Estimation
2. Key Node Selection
3. Graph Construction (Key + Non-key)
4. Non-key Graph Interpolation (DQB)
5. Uncertainty-aware Optimization

---

## Step 1: Dynamic Uncertainty Estimation

**Computed once at initialization**, before any USplat4D optimization begins, using the pretrained Gaussian states.

### Scalar Uncertainty (eq. 3)

The scalar uncertainty of Gaussian `i` at frame `t` is derived from the photometric loss (eq. 2) under a maximum likelihood / local minimum assumption:

```
σ²_{i,t} = ( Σ_{h ∈ Ω_{i,t}} (T^h_{i,t} · α_i)² )^{-1}
```

where `Ω_{i,t}` is the set of pixels that Gaussian `i` contributes to at frame `t`, `T^h_{i,t}` is the transmittance, and `α_i` is the opacity. Intuitively: a Gaussian seen clearly by many pixels will have a large sum of squared blending weights → small variance → low uncertainty.

### Occlusion / Convergence Check (eq. 4–5)

To handle unconverged pixels (where the local minimum assumption doesn't hold), we define a per-pixel indicator:

```
1_t(h) = 1  if ||C̄^h_t - C^h_t||₁ < η_c,   else 0
```

with `η_c = 0.5` (Appendix D.2). The **L1 color error is normalized to [0, 1]**, so
(255,255,255) − (0,0,0) maps to 1.

The per-Gaussian aggregate indicator is:

```
1_{i,t} = Π_{h ∈ Ω_{i,t}} 1_t(h)     # 1 only if ALL covered pixels are converged
```

Final scalar uncertainty:

```
u_{i,t} = 1_{i,t} · σ²_{i,t} + (1 − 1_{i,t}) · φ
```

If any covered pixel is unconverged, the Gaussian is assigned the large constant `φ` (high uncertainty).

### Depth-aware Uncertainty Matrix (eq. 6)

Scalar uncertainty is isotropic, but monocular depth is less reliable than image-plane directions.
We lift it into an anisotropic 3D matrix:

```
U_{i,t} = R_wc · U_c · R_wc^T
U_c = diag(r_x · u_{i,t},  r_y · u_{i,t},  r_z · u_{i,t})
```

where `R_wc` is the camera-to-world rotation. Default scaling: **`[r_x, r_y, r_z] = [1, 1, 0.01]`**
— depth axis is strongly down-weighted because monocular depth is unreliable. (Only rotation is needed to propagate uncertainty; translation does not affect covariance.)

---

## Step 2: Key Node Selection

**Computed once** after uncertainty estimation, before optimization. Never updated during training.

Key nodes are the top 2% lowest-uncertainty Gaussians (~1000 nodes), selected via a two-stage process:

### Stage 1: Candidate Sampling via 3D Voxel Grid

At **each frame independently**, partition the scene into a 3D voxel grid. Voxels containing only high-uncertainty Gaussians are discarded. From each remaining voxel, randomly select **one** low-uncertainty Gaussian as a candidate.

> **Why per-frame?** Gaussians scatter to different positions over time, so a different set of voxels may be occupied per frame. This ensures spatial coverage across the whole sequence.
> The voxel size may implicitly vary per frame due to the spatial normalization step (Appendix B.1).

The union of all per-frame candidates forms the candidate pool.

### Stage 2: Filter by Significant Period

For each candidate, compute its **significant period**: the number of **consecutive frames** where `u_{i,t}` stays below the uncertainty threshold.

Keep only candidates with significant period ≥ **5 frames**.

> The paper says "at least 5 frames"; ablation (Appendix D.3) confirms SPT ≥ 3 is a stable plateau. The paper's wording ("insufficient temporal coverage") implies continuity is the intent.

### Stage 3: Rank and Select Top 2%

From the filtered candidates, rank by uncertainty (ascending). Keep the **top 2%** (~1000
Gaussians). These become the key node set `V_k`; all others are `V_n` (non-key nodes).

---

## Step 3: Key Graph Construction

**Computed once**, never updated during optimization.

The graph `G = (V, E)` is a **directed graph**. Edges for key and non-key nodes are constructed separately.

### Key Node Edges — Uncertainty-Aware kNN (UA-kNN, eq. 7)

For each key node `i`:

1. Find its most reliable frame: `t̂ = argmin_t { u_{i,t} }`
2. At frame `t̂`, find its **k nearest key-node neighbors** (pick k=8, not specified in paper) using the **Mahalanobis distance** weighted by the *sum* of both nodes' uncertainty matrices:

```
E_i = kNN_{j ∈ V_k \ {i}}  ||p_{i,t̂} − p_{j,t̂}||_{(U_{w,t̂,i} + U_{w,t̂,j})}
```

The Mahalanobis metric **up-weights directions of high uncertainty**, so it favors connections between nodes that are spatially close *along reliable axes*. Edges are **directed**: node `i` connects to `j`, but not necessarily vice versa.

The graph is **fixed after construction** and never modified during optimization. Uncertainty only appears again as loss *weights* during training — it does not change which edges exist.

---

## Step 4: Non-key Graph Interpolation

**Computed once**, never updated during optimization.

For each non-key node `i`, find its **closest key node across the entire sequence** (eq. 8):

```
j* = argmin_{l ∈ V_k}  Σ_{t=0}^{T-1}  ||p_{i,t} − p_{l,t}||_{(U_{w,t,i} + U_{w,t,l})}
```

Sum (equivalently, mean) of Mahalanobis distances over **all frames**. All Gaussians exist across every frame.

Then: `E_i = E_{j*} ∪ {j*}`

Non-key node `i` connects to: its closest key node `j*` **plus** all of `j*`'s key-graph neighbors
→ total of **k+1** connections.

### DQB Interpolation (eq. 10)

The position and rotation of non-key node `i` at time `t` are interpolated from its key neighbors using **Dual Quaternion Blending (DQB)**:

```
(p^DQB_{i,t}, q^DQB_{i,t}) = DQB{ (w_{ij}, T_{j,t}) }_{j ∈ E_i}
```

where:

- `T_{j,t} ∈ SE(3)` is the full rigid transform (position + rotation) of key node `j` at time `t`
- `w_{ij}` are **normalized edge weights** (sum to 1 over all j ∈ E_i)
- DQB blends the SE(3) transforms, then extracts position and rotation from the result

---

## Step 5: Uncertainty-aware Optimization

### Key Node Loss (eq. 9)

```
L^key = Σ_t Σ_{i ∈ V_k}  ||p_{i,t} − p°_{i,t}||_{U^{-1}_{w,t,i}}  +  L^motion_key
```

- Mahalanobis distance to **pretrained (pre-optimization) positions** `p°`, weighted by the **inverse** of the uncertainty matrix — corrections are made mainly along reliable axes
- `L^motion_key` = motion regularizer (see below)

### Non-key Node Loss (eq. 11)

```
L^non-key = Σ_t Σ_{i ∈ V_n}  ||p_{i,t} − p°_{i,t}||_{U^{-1}_{w,i}}
           + Σ_t Σ_{i ∈ V_n}  ||p_{i,t} − p^DQB_{i,t}||_{U^{-1}_{w,i}}
           + L^motion_non-key
```

Note: the non-key uncertainty `U_{w,i}` has **no time index** — it uses a per-node aggregated value (unlike the per-frame `U_{w,t,i}` for key nodes).

### Motion Loss (Appendix A.2)

```
L^motion = λ_iso · L^iso  +  λ_rigid · L^rigid  +  λ_rot · L^rot  +  λ_vel · L^vel  +  λ_acc · L^acc
```

with: `λ_iso = λ_rigid = 1`,  `λ_rot = λ_vel = λ_acc = 0.01`

- **Isometry** (`L^iso`): edge lengths between neighbors should be preserved across time vs canonical
- **Rigidity** (`L^rigid`): local rigid transformation consistency between neighbor pairs over Δ frames
- **Relative rotation** (`L^rot`): relative quaternion between neighbors should be consistent over Δ
- **Velocity** (`L^vel`): L1 penalty on position and rotation changes between consecutive frames
- **Acceleration** (`L^acc`): L1 penalty on second-order finite differences (position and rotation)

Applied **separately** to key nodes and non-key nodes.

### Perception Loss (Appendix A.2.2)

```
L^rgb = L1 + SSIM + mask loss + depth loss + depth gradient loss + tracking loss
```

The 2D prior losses (mask, depth, tracking) are **inherited from the base model** (SoM or MoSca).
They are not novel contributions of USplat4D.

### Total Loss (eq. 12)

```
L^total = L^rgb + L^key + L^non-key
```

### Training Schedule (Appendix B.2)

- Base: SoM (+400 epochs) or MoSca (+1600 steps), batch size 8
- **First 10% and last 20%** of training: disable density control and opacity reset
- **Middle 70%**: enable density control and opacity reset

---

## Key Hyperparameters

| Parameter                          | Value            | Notes                                       |
| ---------------------------------- | ---------------- | ------------------------------------------- |
| η_c (color threshold)              | 0.5              | L1 color error in [0,1]; plateau [0.4, 0.8] |
| φ (high-uncertainty constant)      | large constant   | assigned when pixel unconverged             |
| [r_x, r_y, r_z]                    | [1, 1, 0.01]     | depth axis down-weighted                    |
| Key ratio                          | 2% (~1000 nodes) | stable from 0.5%–4%                         |
| Significant Period Threshold (SPT) | 5 frames         | stable for SPT ≥ 3                          |
| k (kNN edges)                      | 8                | not specified in paper; reasonable default  |
| λ_iso, λ_rigid                     | 1.0              | geometry preservation                       |
| λ_rot, λ_vel, λ_acc                | 0.01             | smoothness                                  |

---

## What USplat4D Is NOT

- It does **not** re-train SoM/MoSca from scratch
- It does **not** update the graph during optimization
- It does **not** recompute uncertainty during optimization
- The 2D prior losses are from the base model, not from USplat4D
