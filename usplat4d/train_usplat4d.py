"""
USplat4D training script.

Usage:
  python -m usplat4d.train_usplat4d \
    --ckpt /path/to/som/checkpoints/last.ckpt \
    --data-type iphone \
    --data-dir /media/ee904/DATA1/Yun/Datasets/shape-of-motion/iphone/backpack \
    --output-dir /media/ee904/DATA1/Yun/Outputs/usplat4d/iphone/backpack

Pipeline:
  1. Load pretrained SoM checkpoint → GaussianState
  2. Load dataset
  3. Collect GT frames for uncertainty estimation
  4. Compute scalar uncertainty u (G, T)
  5. Compute uncertainty matrices U (G, T, 3, 3)
  6. Select key nodes (is_key mask)
  7. Build key graph
  8. Build non-key graph
  9. Run USplat4DOptimizer for extra_epochs
  10. Save refined checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm

SOM_ROOT = Path("/home/ee904/Yun/shape-of-motion")
if str(SOM_ROOT) not in sys.path:
    sys.path.insert(0, str(SOM_ROOT))

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig, FGLRConfig, BGLRConfig, MotionLRConfig, CameraPoseLRConfig, CameraScalesLRConfig  # noqa: E402
from flow3d.data import (  # noqa: E402
    BaseDataset,
    DavisDataConfig,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
)
from flow3d.data.utils import to_device  # noqa: E402
from flow3d.trainer import Trainer  # noqa: E402

from .config import USplat4DConfig
from .gaussian_state import load_from_checkpoint
from .graph.key_node_selection import select_key_nodes
from .graph.key_graph import build_key_graph
from .graph.nonkey_graph import build_nonkey_graph
from .optimizer import USplat4DOptimizer
from .uncertainty.depth_aware import compute_uncertainty_matrix, aggregate_uncertainty_matrix
from .uncertainty.estimator import compute_scalar_uncertainty


# ──────────────────────────────────────────────────────────────────────────────

def build_data_config(data_type: str, data_dir: str):
    """Build the appropriate SoM data config for the given dataset type."""
    if data_type == "iphone":
        return iPhoneDataConfig(data_dir=data_dir)
    elif data_type == "nvidia":
        return NvidiaDataConfig(data_dir=data_dir)
    elif data_type == "davis":
        return DavisDataConfig(data_dir=data_dir)
    elif data_type == "custom":
        return CustomDataConfig(data_dir=data_dir)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def collect_gt_images(
    dataset: BaseDataset,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return a list of T tensors, each (H, W, 3) float32 in [0, 1]."""
    T = dataset.num_frames
    imgs = []
    for t in range(T):
        img = dataset.get_image(t)  # (H, W, 3) or (3, H, W)
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        img = img.float()
        if img.max() > 1.5:
            img = img / 255.0
        imgs.append(img.to(device))
    return imgs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/checkpoints", exist_ok=True)

    cfg = USplat4DConfig()

    # ── Load SoM checkpoint ───────────────────────────────────────────────────
    guru.info(f"Loading SoM checkpoint from {args.ckpt}")
    state, raw_ckpt = load_from_checkpoint(args.ckpt, device)
    guru.info(f"G={state.num_gaussians}, T={state.num_frames}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    data_cfg = build_data_config(args.data_type, args.data_dir)
    train_dataset, _, _, _ = get_train_val_datasets(data_cfg, load_val=False)

    img_wh = train_dataset.get_img_wh()
    W, H   = img_wh

    assert train_dataset.num_frames == state.num_frames, (
        f"Dataset frames ({train_dataset.num_frames}) != checkpoint frames "
        f"({state.num_frames})"
    )

    # ── Collect GT images for uncertainty estimation ──────────────────────────
    guru.info("Collecting GT images …")
    gt_images = collect_gt_images(train_dataset, device)

    # ── Step 1: Scalar uncertainty ────────────────────────────────────────────
    guru.info("Computing scalar uncertainty …")
    u = compute_scalar_uncertainty(
        model      = state.model,
        gt_images  = gt_images,
        img_wh     = img_wh,
        eta_c      = cfg.eta_c,
        phi        = cfg.phi,
        chunk_size = cfg.unc_chunk_size,
        device     = device,
        verbose    = True,
    )
    guru.info(f"u: min={u.min():.4f}, max={u.max():.4f}, "
              f"mean={u.mean():.4f} — shape {tuple(u.shape)}")

    # ── Step 2: Depth-aware uncertainty matrices ──────────────────────────────
    guru.info("Computing depth-aware uncertainty matrices …")
    U = compute_uncertainty_matrix(
        u    = u,
        w2cs = state.w2cs,
        rx   = cfg.rx,
        ry   = cfg.ry,
        rz   = cfg.rz,
    )  # (G, T, 3, 3)
    U_agg = aggregate_uncertainty_matrix(U)  # (G, 3, 3) for non-key

    # ── Step 3: Key node selection ────────────────────────────────────────────
    scene_scale = float(state.model.fg.scene_scale.cpu())
    guru.info(f"Selecting key nodes (key_ratio={cfg.key_ratio}, scene_scale={scene_scale:.4f}) …")
    is_key = select_key_nodes(
        u               = u,
        positions       = state.positions,
        scene_scale     = scene_scale,
        key_ratio       = cfg.key_ratio,
        spt             = cfg.spt,
        voxel_resolution= cfg.voxel_resolution,
    )
    guru.info(f"Key nodes: {is_key.sum().item()} / {state.num_gaussians} "
              f"({100*is_key.float().mean().item():.2f}%)")

    # ── Step 4: Key graph ─────────────────────────────────────────────────────
    guru.info("Building key graph (UA-kNN) …")
    key_graph = build_key_graph(
        positions = state.positions,
        U         = U,
        u         = u,
        is_key    = is_key,
        k         = cfg.k_neighbors,
    )
    guru.info(f"Key graph: {key_graph.num_key_nodes} nodes, k={key_graph.k}")

    # ── Step 5: Non-key graph ─────────────────────────────────────────────────
    guru.info("Building non-key graph …")
    graph = build_nonkey_graph(
        positions  = state.positions,
        U          = U,
        u          = u,
        is_key     = is_key,
        key_graph  = key_graph,
    )
    guru.info(f"Non-key graph: {graph.num_nonkey} non-key nodes")

    # ── Build SoM Trainer (re-uses optimizer + scheduler from checkpoint) ─────
    lr_cfg = SceneLRConfig(
        fg            = FGLRConfig(),
        bg            = BGLRConfig(),
        motion_bases  = MotionLRConfig(),
        camera_poses  = CameraPoseLRConfig(),
        camera_scales = CameraScalesLRConfig(),
    )
    loss_cfg  = LossesConfig()
    optim_cfg = OptimizerConfig()

    trainer = Trainer(
        model      = state.model,
        device     = device,
        lr_cfg     = lr_cfg,
        losses_cfg = loss_cfg,
        optim_cfg  = optim_cfg,
        work_dir   = args.output_dir,
    )
    # Restore optimizer + scheduler states from the SoM checkpoint.
    if "optimizers" in raw_ckpt:
        try:
            trainer.load_checkpoint_optimizers(raw_ckpt["optimizers"])
        except Exception as e:
            guru.warning(f"Could not restore optimizer states: {e}")
    if "schedulers" in raw_ckpt:
        try:
            trainer.load_checkpoint_schedulers(raw_ckpt["schedulers"])
        except Exception as e:
            guru.warning(f"Could not restore scheduler states: {e}")

    # ── Build USplat4D optimizer ──────────────────────────────────────────────
    optimizer = USplat4DOptimizer(
        state   = state,
        graph   = graph,
        U       = U,
        U_agg   = U_agg,
        cfg     = cfg,
        trainer = trainer,
    )

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size     = cfg.batch_size,
        num_workers    = 4,
        persistent_workers = True,
        collate_fn     = BaseDataset.train_collate_fn,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    total_epochs = cfg.extra_epochs
    guru.info(f"Starting USplat4D training for {total_epochs} epochs …")

    for epoch in (pbar := tqdm(range(total_epochs))):
        trainer.set_epoch(epoch)
        epoch_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            batch = to_device(batch, device)
            stats = optimizer.train_step(batch, epoch, total_epochs)
            epoch_loss += stats.get("usplat4d/L_total", 0.0)
            n_steps += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        pbar.set_description(f"Loss: {avg_loss:.5f}")

        # Periodic checkpoint.
        if (epoch + 1) % 50 == 0 or epoch == total_epochs - 1:
            ckpt_path = f"{args.output_dir}/checkpoints/epoch_{epoch+1:04d}.ckpt"
            optimizer.save_checkpoint(ckpt_path)
            guru.info(f"Saved checkpoint → {ckpt_path}")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_path = f"{args.output_dir}/checkpoints/last.ckpt"
    optimizer.save_checkpoint(final_path)
    guru.info(f"USplat4D training complete. Final checkpoint: {final_path}")


# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train USplat4D on top of a SoM checkpoint")
    p.add_argument("--ckpt",       required=True,  help="Path to SoM last.ckpt")
    p.add_argument("--data-type",  required=True,
                   choices=["iphone", "nvidia", "davis", "custom"],
                   help="Dataset type")
    p.add_argument("--data-dir",   required=True,  help="Path to scene dataset directory")
    p.add_argument("--output-dir", required=True,  help="Where to save USplat4D outputs")
    # Optional overrides
    p.add_argument("--extra-epochs", type=int, default=None,
                   help="Override number of extra training epochs (default 400)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
