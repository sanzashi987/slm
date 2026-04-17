# ----------------------------------------------------------------------------------------------
# Minimal FreiHAND trainer for FastMETRO, optimized for a single RTX 4090.
# Derived from: kaist-ami/FastMETRO (MIT License)
# ----------------------------------------------------------------------------------------------
#
# 4090-specific optimizations (all enabled by default, no flags):
#   1. bfloat16 autocast around forward + losses.
#      (bf16 range == fp32, so no GradScaler needed.)
#   2. TF32 matmul / cudnn.allow_tf32 -> faster non-autocast fp32 paths.
#   3. cudnn.benchmark=True (input size 224x224 is fixed).
#   4. channels_last memory format for CNN backbone.
#   5. Vectorized mesh sampling (single sparse matmul instead of per-sample python loop).
#   6. torch.no_grad() around MANO GT mesh regeneration.
#   7. batch_size 96, num_workers 8, prefetch_factor 4.
#
# Usage:
#   python -m src.train_freihand \
#       --freihand_dir /path/to/FreiHAND_pub_v2 \
#       --mano_dir     /path/to/mano_v1_2/models \
#       --output_dir   ./outputs/run1
# ----------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import os.path as op
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import FreiHANDDataset
from src.modeling.mano_utils import MANO, Mesh, WRIST_IDX
from src.modeling.model import FastMETRO_Hand_Network
from src.utils.mesh_io import save_mesh_obj


# -------------------- utilities --------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("fastmetro_min")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    fh = logging.FileHandler(op.join(output_dir, "train.log")); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def orthographic_projection(X, camera):
    """X: B x N x 3;  camera: B x 3 (scale, tx, ty).  returns B x N x 2."""
    scale = camera[:, 0:1].unsqueeze(-1)
    trans = camera[:, 1:].unsqueeze(1)
    return scale * X[:, :, :2] + trans


# -------------------- losses --------------------

def keypoint_2d_loss(pred_2d, gt_2d_with_conf):
    conf = gt_2d_with_conf[..., -1:].clone()
    gt = gt_2d_with_conf[..., :-1].clone()
    return (conf * F.l1_loss(pred_2d, gt, reduction="none")).mean()


def keypoint_3d_loss(pred_3d, gt_3d):
    pred = pred_3d - pred_3d[:, WRIST_IDX:WRIST_IDX + 1, :]
    gt = gt_3d - gt_3d[:, WRIST_IDX:WRIST_IDX + 1, :]
    return F.l1_loss(pred, gt)


def vertices_loss(pred_v, gt_v):
    return F.l1_loss(pred_v, gt_v)


class EdgeLengthGTLoss(nn.Module):
    def __init__(self, face):
        super().__init__()
        self.register_buffer("face", torch.as_tensor(face, dtype=torch.long))

    def forward(self, pred_v, gt_v):
        f = self.face
        def _edges(v):
            d1 = (v[:, f[:, 0]] - v[:, f[:, 1]]).pow(2).sum(-1).clamp(min=1e-8).sqrt()
            d2 = (v[:, f[:, 0]] - v[:, f[:, 2]]).pow(2).sum(-1).clamp(min=1e-8).sqrt()
            d3 = (v[:, f[:, 1]] - v[:, f[:, 2]]).pow(2).sum(-1).clamp(min=1e-8).sqrt()
            return d1, d2, d3
        po = _edges(pred_v); go = _edges(gt_v)
        return sum((po[i] - go[i]).abs().mean() for i in range(3)) / 3.0


class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super().__init__()
        self.register_buffer("face", torch.as_tensor(face, dtype=torch.long))

    def forward(self, pred_v, gt_v):
        f = self.face
        v1p = F.normalize(pred_v[:, f[:, 1]] - pred_v[:, f[:, 0]], dim=2)
        v2p = F.normalize(pred_v[:, f[:, 2]] - pred_v[:, f[:, 0]], dim=2)
        v3p = F.normalize(pred_v[:, f[:, 2]] - pred_v[:, f[:, 1]], dim=2)
        v1g = F.normalize(gt_v[:, f[:, 1]] - gt_v[:, f[:, 0]], dim=2)
        v2g = F.normalize(gt_v[:, f[:, 2]] - gt_v[:, f[:, 0]], dim=2)
        normal_gt = F.normalize(torch.cross(v1g, v2g, dim=2), dim=2)
        c1 = (v1p * normal_gt).sum(-1).abs()
        c2 = (v2p * normal_gt).sum(-1).abs()
        c3 = (v3p * normal_gt).sum(-1).abs()
        return (c1.mean() + c2.mean() + c3.mean()) / 3.0


class AvgMeter:
    def __init__(self):
        self.n = 0; self.s = 0.0
    def update(self, v, k=1):
        self.n += k; self.s += float(v) * k
    @property
    def avg(self):
        return self.s / max(self.n, 1)


# -------------------- backbone --------------------

def build_resnet50_backbone():
    """torchvision ResNet-50, strip avgpool + fc. Output: B x 2048 x 7 x 7 for 224 input."""
    m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    return nn.Sequential(*list(m.children())[:-2])


# -------------------- checkpoint / export --------------------

def save_checkpoint(model, optimizer, lr_scheduler, epoch, out_dir, tag="latest"):
    os.makedirs(out_dir, exist_ok=True)
    path = op.join(out_dir, f"checkpoint_{tag}.pth")
    payload = {
        "epoch": epoch,
        "model": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }
    torch.save(payload, path)
    return path


def export_sample_objs(model, mano_model, dataset, device, out_dir, num_samples=3, tag="latest"):
    obj_dir = op.join(out_dir, f"meshes_{tag}")
    os.makedirs(obj_dir, exist_ok=True)
    model.eval()
    faces = mano_model.face
    with torch.no_grad():
        for i in range(num_samples):
            sample = dataset[i]
            img = sample["image"].unsqueeze(0).to(device).to(memory_format=torch.channels_last)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(img)
            verts = out["pred_3d_vertices_fine"][0].float().cpu().numpy()
            save_mesh_obj(verts, faces, op.join(obj_dir, f"sample_{i:04d}.obj"))
    model.train()
    return obj_dir


# -------------------- training --------------------

def train_one_epoch(args, model:FastMETRO_Hand_Network, mano_model:MANO, mesh_sampler:Mesh, loader, optimizer, logger, epoch):
    model.train()
    meters = {k: AvgMeter() for k in ["loss", "l_j3d", "l_v3d", "l_edge", "l_norm", "l_j2d"]}

    edge_loss_fn = EdgeLengthGTLoss(mano_model.face).to(args.device)
    normal_loss_fn = NormalVectorLoss(mano_model.face).to(args.device)

    t0 = time.time()
    for it, batch in enumerate(loader):
        images = batch["image"].to(args.device, non_blocking=True) \
                              .to(memory_format=torch.channels_last)
        pose = batch["pose"].to(args.device, non_blocking=True)
        betas = batch["betas"].to(args.device, non_blocking=True)
        gt_joints_2d = batch["joints_2d"].to(args.device, non_blocking=True)
        bs = images.size(0)

        # GT mesh regeneration -- no gradient, no autocast (manopth LBS is fp32-sensitive).
        with torch.no_grad():
            gt_vertices_fine, gt_joints = mano_model.layer(pose, betas)
            gt_vertices_fine = gt_vertices_fine / 1000.0
            gt_joints = gt_joints / 1000.0
            gt_vertices_coarse = mesh_sampler.downsample(gt_vertices_fine)

            root = gt_joints[:, WRIST_IDX:WRIST_IDX + 1, :]
            gt_vertices_fine = gt_vertices_fine - root
            gt_vertices_coarse = gt_vertices_coarse - root
            gt_joints = gt_joints - root

        # Forward + losses in bf16 autocast.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(images)
            pred_cam = out["pred_cam"]
            pred_joints_tok = out["pred_3d_joints"]
            pred_v_coarse = out["pred_3d_vertices_coarse"]
            pred_v_fine = out["pred_3d_vertices_fine"]

            pred_joints_mano = mano_model.get_3d_joints(pred_v_fine)
            root_m = pred_joints_mano[:, WRIST_IDX:WRIST_IDX + 1, :]
            pred_v_fine = pred_v_fine - root_m
            pred_v_coarse = pred_v_coarse - root_m
            pred_joints_mano = pred_joints_mano - root_m
            root_t = pred_joints_tok[:, WRIST_IDX:WRIST_IDX + 1, :]
            pred_joints_tok = pred_joints_tok - root_t

            pred_2d_mano = orthographic_projection(pred_joints_mano, pred_cam)
            pred_2d_tok = orthographic_projection(pred_joints_tok, pred_cam)

            l_j3d = keypoint_3d_loss(pred_joints_tok, gt_joints) + keypoint_3d_loss(pred_joints_mano, gt_joints)
            l_v3d = (args.w_v_coarse * vertices_loss(pred_v_coarse, gt_vertices_coarse)
                        + args.w_v_fine * vertices_loss(pred_v_fine, gt_vertices_fine))
            l_edge = edge_loss_fn(pred_v_fine, gt_vertices_fine)
            l_norm = normal_loss_fn(pred_v_fine, gt_vertices_fine)
            l_j2d = keypoint_2d_loss(pred_2d_tok, gt_joints_2d) + keypoint_2d_loss(pred_2d_mano, gt_joints_2d)

            loss = (args.w_j3d * l_j3d
                    + args.w_v3d * l_v3d
                    + args.w_edge_normal * (args.w_edge * l_edge + args.w_normal * l_norm)
                    + args.w_j2d * l_j2d)

        # Backward + step outside autocast. No GradScaler with bf16.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        meters["loss"].update(loss.item(), bs)
        meters["l_j3d"].update(l_j3d.item(), bs)
        meters["l_v3d"].update(l_v3d.item(), bs)
        meters["l_edge"].update(l_edge.item(), bs)
        meters["l_norm"].update(l_norm.item(), bs)
        meters["l_j2d"].update(l_j2d.item(), bs)

        if (it + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            ips = (it + 1) * bs / elapsed
            logger.info(
                f"epoch {epoch} iter {it+1}/{len(loader)}  "
                f"loss={meters['loss'].avg:.4f}  "
                f"j3d={meters['l_j3d'].avg:.4f}  v3d={meters['l_v3d'].avg:.4f}  "
                f"edge={meters['l_edge'].avg:.4f}  norm={meters['l_norm'].avg:.4f}  "
                f"j2d={meters['l_j2d'].avg:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"{ips:.1f} img/s"
            )
    return meters["loss"].avg


# -------------------- args --------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--freihand_dir", type=str, required=True)
    p.add_argument("--mano_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs/run1")
    p.add_argument("--model_data_dir", type=str, default="src/modeling/data")

    # Training (tuned for RTX 4090 24GB with bf16)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_drop_epochs", type=int, default=80)
    p.add_argument("--clip_max_norm", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=88)
    p.add_argument("--log_every", type=int, default=50)

    # Loss weights
    p.add_argument("--w_j3d", type=float, default=1000.0)
    p.add_argument("--w_v3d", type=float, default=100.0)
    p.add_argument("--w_edge_normal", type=float, default=100.0)
    p.add_argument("--w_j2d", type=float, default=100.0)
    p.add_argument("--w_v_coarse", type=float, default=0.5)
    p.add_argument("--w_v_fine", type=float, default=0.5)
    p.add_argument("--w_edge", type=float, default=1.0)
    p.add_argument("--w_normal", type=float, default=0.1)

    # Model (FastMETRO-S + ResNet-50)
    p.add_argument("--model_name", type=str, default="FastMETRO-S",
                   choices=["FastMETRO-S", "FastMETRO-M", "FastMETRO-L"])
    p.add_argument("--model_dim_1", type=int, default=512)
    p.add_argument("--model_dim_2", type=int, default=128)
    p.add_argument("--feedforward_dim_1", type=int, default=2048)
    p.add_argument("--feedforward_dim_2", type=int, default=512)
    p.add_argument("--conv_1x1_dim", type=int, default=2048)
    p.add_argument("--transformer_dropout", type=float, default=0.1)
    p.add_argument("--transformer_nhead", type=int, default=8)
    p.add_argument("--pos_type", type=str, default="sine")

    # Export / resume
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--obj_samples", type=int, default=3)
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


# -------------------- main --------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    logger = setup_logger(args.output_dir)

    # --- 4090 CUDA toggles ---
    # TF32 for fp32 matmul/convs (faster non-autocast paths, free on Ada).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 224x224 fixed input -> benchmark selects fastest conv algos once.
    torch.backends.cudnn.benchmark = True

    assert torch.cuda.is_available(), "This trainer is tuned for CUDA; no CPU fallback."
    args.device = torch.device("cuda")
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Args: {json.dumps(vars(args), default=str, indent=2)}")

    # --- Dataset & loader ---
    train_ds = FreiHANDDataset(args.freihand_dir, split="train")
    logger.info(f"Train dataset: {len(train_ds)} images")
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, **loader_kwargs)

    # --- MANO + mesh sampler ---
    mano_model = MANO(args.mano_dir).to(args.device)
    mesh_sampler = Mesh(
        sampling_npz_path=op.join(args.model_data_dir, "mano_downsampling.npz"),
        device=args.device,
    )

    # --- Backbone + FastMETRO (channels_last for conv efficiency) ---
    backbone = build_resnet50_backbone()
    model = FastMETRO_Hand_Network(args, backbone, mesh_sampler).to(args.device)
    model = model.to(memory_format=torch.channels_last)

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_back = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(f"Total trainable params : {n_total:,}")
    logger.info(f"  of which backbone    : {n_back:,}")
    logger.info(f"  of which transformer : {n_total - n_back:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epochs, gamma=0.1)

    start_epoch = 0
    if args.resume and op.isfile(args.resume):
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"] + 1

    logger.info("Starting training (bf16 autocast, TF32, cudnn.benchmark, channels_last)...")
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(args, model, mano_model, mesh_sampler, train_loader,
                                   optimizer, logger, epoch)
        lr_scheduler.step()
        logger.info(f"[epoch {epoch}] avg loss = {avg_loss:.4f}")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = save_checkpoint(model, optimizer, lr_scheduler, epoch,
                                        args.output_dir, tag=f"epoch{epoch:03d}")
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args.output_dir, tag="latest")
            logger.info(f"Saved checkpoint -> {ckpt_path}")

            if args.obj_samples > 0:
                obj_dir = export_sample_objs(
                    model, mano_model, train_ds, args.device, args.output_dir,
                    num_samples=args.obj_samples, tag=f"epoch{epoch:03d}",
                )
                logger.info(f"Saved {args.obj_samples} sample meshes -> {obj_dir}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
