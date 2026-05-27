"""
GaussianState: thin data container wrapping a loaded SoM SceneModel with
precomputed per-frame trajectories needed for graph construction.

The SoM model stores motion via canonical positions + motion bases (not dense
per-frame tensors), so we precompute (G, T, 3) position and (G, T, 4) quat
arrays once at load time for use in graph construction and loss computation.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

# Make SoM importable regardless of where the script is invoked from.
SOM_ROOT = Path("/home/ee904/Yun/shape-of-motion")
if str(SOM_ROOT) not in sys.path:
    sys.path.insert(0, str(SOM_ROOT))

from flow3d.scene_model import SceneModel  # noqa: E402


@dataclass
class GaussianState:
    """Container for a loaded SoM model plus precomputed trajectory tensors.

    Attributes
    ----------
    model : SceneModel
        The live SoM model whose parameters are optimized during USplat4D.
    positions : Tensor  (G, T, 3)
        Per-frame 3D means, computed once at load time (no grad).
        Used for graph construction.  Re-computed with grad during training.
    quats : Tensor  (G, T, 4)
        Per-frame unit quaternions (wxyz), same timing as positions.
    positions_ref : Tensor  (G, T, 3)
        Frozen copy of the pretrained positions — the anchor p° in the graph
        losses (eq. 9 / 11).
    quats_ref : Tensor  (G, T, 4)
        Frozen copy of the pretrained quaternions.
    w2cs : Tensor  (T, 4, 4)
        World-to-camera matrices for each frame (from checkpoint).
    Ks : Tensor  (T, 3, 3)
        Camera intrinsics for each frame (from checkpoint).
    """

    model: SceneModel
    positions: Tensor       # (G, T, 3)
    quats: Tensor           # (G, T, 4)
    positions_ref: Tensor   # (G, T, 3)  frozen
    quats_ref: Tensor       # (G, T, 4)  frozen
    w2cs: Tensor            # (T, 4, 4)
    Ks: Tensor              # (T, 3, 3)

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def num_frames(self) -> int:
        return self.positions.shape[1]

    @property
    def device(self):
        return self.positions.device

    def compute_current_poses(self) -> tuple[Tensor, Tensor]:
        """Re-compute per-frame positions and quats from the *current* (trainable)
        model parameters.  Returns tensors with gradient flow for loss computation.

        Returns
        -------
        positions : (G, T, 3)
        quats     : (G, T, 4)  wxyz, unit-normalized
        """
        T = self.num_frames
        device = self.device
        ts = torch.arange(T, device=device)
        positions, quats = self.model.compute_poses_fg(ts)  # (G, T, 3/4)
        return positions, quats


def load_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
) -> tuple["GaussianState", dict]:
    """Load a pretrained SoM checkpoint and construct a GaussianState.

    Parameters
    ----------
    ckpt_path : str
        Path to the SoM ``last.ckpt`` file.
    device : torch.device

    Returns
    -------
    state : GaussianState
    ckpt  : dict  (raw checkpoint dict, needed to restore optimizer state)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct the SceneModel from the saved state dict.
    model_state = ckpt["model"]
    model = SceneModel.init_from_state_dict(model_state)
    model = model.to(device)
    model.eval()

    T = model.num_frames

    # Precompute per-frame positions and quats (no gradient — for graph init).
    with torch.no_grad():
        ts = torch.arange(T, device=device)
        positions, quats = model.compute_poses_fg(ts)  # (G, T, 3/4)
        # roma returns wxyz; verify shape
        assert positions.shape[-1] == 3 and quats.shape[-1] == 4

    positions_ref = positions.clone()
    quats_ref = quats.clone()

    # Extract camera matrices (registered buffers, not trainable).
    w2cs = model.w2cs.to(device)   # (T, 4, 4)
    Ks = model.Ks.to(device)       # (T, 3, 3)

    state = GaussianState(
        model=model,
        positions=positions,
        quats=quats,
        positions_ref=positions_ref,
        quats_ref=quats_ref,
        w2cs=w2cs,
        Ks=Ks,
    )

    return state, ckpt
