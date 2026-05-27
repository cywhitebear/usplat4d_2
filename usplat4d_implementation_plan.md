# USplat4D Re-implementation Plan

## Architecture Decision

USplat4D is a **standalone module** that:

1. Accepts pretrained 4D Gaussians from SoM (or MoSca) as input
2. Runs its own graph construction + optimization pass
3. Outputs refined 4D Gaussians

Claude Code should read the SoM codebase to understand the Gaussian data format, then
implement USplat4D as an independent module that interfaces with it.

---

## Dataset Conventions

```
Datasets:  /media/ee904/DATA1/Yun/Datasets/shape-of-motion/{dataset}/{scene}/
SoM output: /media/ee904/DATA1/Yun/Outputs/shape-of-motion/{dataset}/{scene}/
USplat4D output:  /media/ee904/DATA1/Yun/Outputs/usplat4d/{dataset}/{scene}/
```

Supported dataset types (inherited from SoM): `iphone`, `nvidia`, `davis`, `custom`

---

## Repository Structure

```
usplat4d/
├── uncertainty/
│   ├── __init__.py
│   ├── estimator.py          # Step 1: uncertainty computation
│   └── depth_aware.py        # scalar → anisotropic matrix
├── graph/
│   ├── __init__.py
│   ├── key_node_selection.py # Step 2: voxel sampling + SPT filter
│   ├── key_graph.py          # Step 3: UA-kNN for key nodes
│   ├── nonkey_graph.py       # Step 4: closest key node + DQB setup
│   └── dqb.py                # Dual Quaternion Blending
├── losses/
│   ├── __init__.py
│   ├── motion_loss.py        # isometry, rigidity, rotation, velocity, acceleration
│   ├── graph_loss.py         # key node loss + non-key node loss
│   └── perception_loss.py    # wrapper for base model's RGB + 2D prior losses
├── debug/
│   ├── render_uncertainty.py # color render the uncertainty of key nodes
│   ├── render_graph.py       # show how the graph is connected in 3D space
│   └── diagnostics.py        # show the loss logs as plots
├── optimizer.py              # main training loop for USplat4D
├── train_usplat4d.py           # top-level: load SoM output → run USplat4D → save
├── render_usplat4d.py        # render output: refernce fron SoM
└── config.py                 # all hyperparameters
```

---

## Implementation Steps

### Phase 0: Setup & Interface with SoM

**Goal**: understand SoM's Gaussian data format so we know what we're consuming.

Tasks for Claude Code:

- Read SoM codebase to identify:
  - How 4D Gaussians are stored (positions over time, quaternions, scale, opacity, color)
  - How to load a pretrained SoM checkpoint
  - What the rendering pipeline looks like (needed for uncertainty estimation)
  - What the existing motion/perception losses look like (to inherit for `L^rgb`)
- Write a `GaussianState` data class that represents the full T-frame trajectory of all N Gaussians
- Write a loader that reads SoM output and populates `GaussianState`

---

### Phase 1: Dynamic Uncertainty Estimation (`uncertainty/`)

**Input**: pretrained Gaussians + rendered frames + ground truth frames + camera poses  
**Output**: `u_{i,t}` scalar (N × T), `U_{i,t}` matrix (N × T × 3 × 3)

#### `estimator.py`

- For each frame t:
  - Do a forward rendering pass using the pretrained Gaussians (no grad needed)
  - For each Gaussian i, collect the set of pixels Ω_{i,t} it contributes to
  - Compute blending weights `v^h_{i,t} = T^h_{i,t} · α_i` per pixel
  - Compute scalar uncertainty: `σ²_{i,t} = (Σ_h (v^h_{i,t})²)^{-1}`
  - Compute per-pixel L1 error: `||C̄^h_t − C^h_t||₁`
  - Compute per-pixel convergence indicator (threshold η_c = 0.5, normalized [0,1])
  - Compute per-Gaussian aggregate indicator: product over all covered pixels
  - Apply eq. 5: replace σ² with φ if indicator = 0
- Output: `u` tensor of shape (N, T)

#### `depth_aware.py`

- For each (i, t): build U_c = diag(r_x, r_y, r_z) * u_{i,t} with [r_x,r_y,r_z]=[1,1,0.01]
- Rotate by R_wc (camera-to-world rotation for frame t): `U_{i,t} = R_wc @ U_c @ R_wc.T`
- Output: `U` tensor of shape (N, T, 3, 3)

**Implementation notes**:

- The rendering pass can reuse SoM's rasterizer
- Pixel-to-Gaussian assignment needs to be extracted from the rasterizer (alpha-compositing order)
- φ should be set large enough to be practically infinite (e.g., 1e6); ablate if needed

---

### Phase 2: Key Node Selection (`graph/key_node_selection.py`)

**Input**: `u` (N × T), Gaussian positions `p` (N × T × 3)  
**Output**: boolean mask `is_key` (N,)

Steps:

1. **Per-frame voxel sampling**:
   
   - For each frame t:
     - Compute voxel grid over current Gaussian positions
     - Discard voxels where all Gaussians have u_{i,t} > uncertainty_threshold
     - From each remaining voxel, randomly sample 1 low-uncertainty Gaussian
   - Union the per-frame candidate sets → candidate pool

2. **Significant period filter**:
   
   - For each candidate i: compute number of consecutive frames where u_{i,t} < threshold
   - Discard candidates with max consecutive run < SPT (default 5)

3. **Top-k selection**:
   
   - Sort remaining candidates by mean uncertainty (ascending)
   - Keep top 2% of total N Gaussians (~1000)

**Implementation notes**:

- Voxel size: not explicitly stated; use spatial normalization from Appendix B.1 (unify spatial
  volume). Reasonable default: divide bounding box into ~10³ voxels, or set voxel edge = 2% of
  scene diagonal. May need tuning.
- The uncertainty_threshold for voxel filtering is separate from SPT; it is the same top-2%
  threshold (i.e., keep voxels that have at least one Gaussian in the bottom 2% of overall uncertainty)

---

### Phase 3: Key Graph Construction (`graph/key_graph.py`)

**Input**: key node positions `p` (|V_k| × T × 3), uncertainty matrices `U` (|V_k| × T × 3 × 3)  
**Output**: edge list (directed), edge weights

For each key node i:

1. Find `t̂ = argmin_t u_{i,t}`

2. Compute Mahalanobis distance to all other key nodes j at frame t̂:
   
   ```
   d(i,j) = (p_{i,t̂} − p_{j,t̂})^T (U_{w,t̂,i} + U_{w,t̂,j})^{-1} (p_{i,t̂} − p_{j,t̂})
   ```

3. Select k=8 nearest neighbors → edges `E_i`

4. Compute edge weights `w_{ij}` (e.g., inverse distance, then normalize)

The graph is **fixed after this point**. Uncertainty does not modify edges during optimization;
it only appears as loss weights. Do not add any graph pruning or update logic.

**Implementation notes**:

- Matrix inversion of (U_i + U_j) can be done with torch.linalg.inv or Cholesky
- For efficiency, batch this across all key nodes (|V_k| is only ~1000)
- Store edges as a sparse adjacency list

---

### Phase 4: Non-key Graph Construction (`graph/nonkey_graph.py`)

**Input**: all Gaussian positions (N × T × 3), uncertainty matrices (N × T × 3 × 3), key node edges  
**Output**: for each non-key node: its closest key node j* and its full edge list E_i

For each non-key node i:

1. Compute total Mahalanobis distance to each key node l across all T frames:
   
   ```
   D(i,l) = Σ_t (p_{i,t} − p_{l,t})^T (U_{w,t,i} + U_{w,t,l})^{-1} (p_{i,t} − p_{l,t})
   ```

2. j* = argmin_l D(i,l)

3. E_i = E_{j*} ∪ {j*}  (k+1 connections total)

4. Compute normalized edge weights w_{ij} for blending

**Implementation notes**:

- This is O(N_nonkey × N_key × T) — with N~50000, N_key~1000, T~100, this is 5B ops
- Batch over non-key nodes; use vectorized Mahalanobis; may need chunking
- Consider approximating with a coarser distance first to prune candidates

---

### Phase 5: DQB Interpolation (`graph/dqb.py`)

**Input**: key node SE(3) transforms `T_{j,t}` over time, edge weights `w_{ij}`  
**Output**: interpolated positions and rotations for non-key nodes

Implement Dual Quaternion Blending (Kavan et al., 2007):

1. Convert each SE(3) transform T_{j,t} to dual quaternion representation
2. For non-key node i at time t: weighted sum of dual quaternions from E_i neighbors
3. Normalize the blended dual quaternion
4. Extract position p^DQB_{i,t} and rotation q^DQB_{i,t}

**Implementation notes**:

- Use a standard DQB library or implement from scratch:
  - Dual quaternion: (q_r, q_d) where q_r is rotation quat, q_d = 0.5 * t * q_r (t = translation quat)
  - Normalize: divide by ||q_r||
  - Handle sign flipping (antipodal issue): flip q_j if dot(q_i, q_j) < 0
- DQB produces correct results for near-rigid blending; it's not iterative

---

### Phase 6: Losses (`losses/`)

#### `motion_loss.py`

Implement all five terms with graph-neighbor structure:

- **L_iso** (eq. S8): preserve edge lengths between current and canonical positions
- **L_rigid** (eq. S9): rigid transform consistency between neighbor pairs over Δ frames
- **L_rot** (eq. S10): relative quaternion consistency over Δ frames
- **L_vel** (eq. S11): L1 of position/rotation change between consecutive frames
- **L_acc** (eq. S12): L1 of second-order finite differences

Combine as: `L_motion = 1·L_iso + 1·L_rigid + 0.01·L_rot + 0.01·L_vel + 0.01·L_acc`

Apply separately to key node set and non-key node set using their respective edge sets.

#### `graph_loss.py`

- **Key node loss** (eq. 9): Mahalanobis distance to pretrained positions `p°`, weighted by U^{-1}_{w,t,i} + motion loss
- **Non-key node loss** (eq. 11):
  - Mahalanobis distance to pretrained positions `p°`, weighted by U^{-1}_{w,i}
  - Mahalanobis distance to DQB-interpolated positions, weighted by U^{-1}_{w,i}
  - Motion loss

**Note on non-key U_{w,i} (no time index)**: aggregate over time, e.g., mean of U_{i,t} over t,
or use the minimum-uncertainty frame's value. Paper notation suggests a time-independent value;
the safest interpretation is to use the mean.

#### `perception_loss.py`

Thin wrapper that calls SoM's existing perception loss (RGB L1 + SSIM + mask + depth + tracking).
USplat4D does not modify this.

---

### Phase 7: Main Optimizer (`optimizer.py`)

```python
class USplat4DOptimizer:
    def __init__(self, gaussian_state, graph, config):
        ...

    def train_step(self, batch, epoch, total_epochs):
        # Compute DQB interpolated positions for non-key nodes
        # Compute key + non-key + perception losses
        # Handle density control / opacity reset schedule:
        #   - First 10% epochs: disabled
        #   - Middle 70%: enabled
        #   - Last 20%: disabled
        ...
```

Training schedule:

- SoM base: 400 extra epochs
- Batch size: 8
- Optimizer: inherit from base model (Adam, same LR schedule as SoM)

---

### Phase 8: Training Pipeline (`train_usplat4d`)

Top-level script:

```
1. Load pretrained SoM checkpoint
2. Extract 4D Gaussian state (positions × T, quats, scale, opacity, color)
3. Run uncertainty estimation (once)
4. Run key node selection (once)
5. Build key graph (once)
6. Build non-key graph (once)
7. Run USplat4D optimization loop
8. Save refined Gaussian state
```

---

## Implementation Order for Claude Code

1. **Read SoM codebase** → understand Gaussian format, rendering, existing losses
2. `config.py` → all hyperparameters in one place
3. `uncertainty/estimator.py` → scalar uncertainty
4. `uncertainty/depth_aware.py` → anisotropic matrix
5. `graph/key_node_selection.py`
6. `graph/key_graph.py` (UA-kNN)
7. `graph/dqb.py` (Dual Quaternion Blending)
8. `graph/nonkey_graph.py`
9. `losses/motion_loss.py`
10. `losses/graph_loss.py`
11. `losses/perception_loss.py`
12. `optimizer.py`
13. `train_usplat4d.py`
14. Integration test on a small iphone scene

---

## Key Open Questions for Claude Code to Resolve from SoM Codebase

- What format are Gaussian positions stored in? (per-frame dense, or basis coefficients?)
- How does SoM's rasterizer expose per-Gaussian pixel contributions? (needed for uncertainty)
- What is SoM's motion loss structure? (to avoid duplication)
- What learning rate / scheduler does SoM use in its fine-tuning stage?
- How does SoM handle density control (Gaussian splitting/pruning)? (to implement the schedule)
