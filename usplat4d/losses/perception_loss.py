"""
Perception loss wrapper (Appendix A.2.2).

L^rgb includes RGB L1 + SSIM + mask + depth + depth gradient + tracking.
These are inherited from the base model (SoM) without modification.

We call SoM's Trainer.compute_losses(batch) directly and return only the
scalar loss (discarding the stats dict).  SoM's smoothness losses on the
motion bases are also included here — USplat4D adds its own motion losses
on top, but does not remove SoM's internal regularizers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

SOM_ROOT = Path("/home/ee904/Yun/shape-of-motion")
if str(SOM_ROOT) not in sys.path:
    sys.path.insert(0, str(SOM_ROOT))

from flow3d.trainer import Trainer  # noqa: E402


def compute_perception_loss(
    trainer: Trainer,
    batch: dict[str, Any],
) -> tuple[Tensor, dict]:
    """Call SoM's full perception loss.

    Parameters
    ----------
    trainer : the SoM Trainer instance (holds the model and optimizer)
    batch   : standard SoM batch dict with keys:
                ts, w2cs, Ks, imgs, masks, depths, valid_masks,
                query_tracks_2d, target_ts, target_w2cs, target_Ks,
                target_tracks_2d, target_visibles, target_invisibles,
                target_confidences, target_track_depths

    Returns
    -------
    loss  : scalar Tensor
    stats : dict of named sub-losses for logging
    """
    loss, stats, _, _ = trainer.compute_losses(batch)
    return loss, stats
