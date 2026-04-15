"""
FastMETRO - FreiHAND Hand Mesh Training Script (Minimal Viable Version)
基于 https://github.com/kaist-ami/FastMETRO 官方实现改写
支持:
  - FreiHAND 数据集训练
  - 训练检查点保存 (每 N epoch 存一次)
  - 验证集推理 + 保存 pred.json
  - 输出结果导出为 OBJ / MTL 格式

运行示例 (单卡):
  python train_freihand.py \
    --data_root ./data/FreiHAND \
    --mano_dir ./models/mano \
    --output_dir ./output/freihand \
    --arch resnet50 \
    --model_name FastMETRO-S \
    --num_train_epochs 200 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --lr 1e-4
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as op
import json
import time
import datetime
import logging

import torch
import torch.nn as nn
import numpy as np

from src.modeling.fastmetro import FastMETRO_Hand_Network, build_fastmetro
from src.datasets.freihand_dataset import FreiHandDataset
from src.utils.mesh_sampler import Mesh
from src.utils.mano_wrapper import MANOWrapper
from src.utils.obj_exporter import export_obj_mtl
from src.utils.geometric_layers import orthographic_projection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Loss helpers
# ─────────────────────────────────────────────


def keypoint_3d_loss(criterion, pred_kp3d, gt_kp3d, has_kp3d):
    """Root-relative 3D keypoint loss."""
    conf = gt_kp3d[:, :, -1:].clone()
    gt = gt_kp3d[:, :, :-1].clone()
    gt = gt[has_kp3d == 1]
    conf = conf[has_kp3d == 1]
    pred = pred_kp3d[has_kp3d == 1]
    if len(gt) == 0:
        return torch.zeros(1, device=pred_kp3d.device, requires_grad=True).squeeze()
    # root-relative
    gt = gt - gt[:, 0:1, :]
    pred = pred - pred[:, 0:1, :]
    return (conf * criterion(pred, gt)).mean()


def vertices_loss(criterion, pred_verts, gt_verts, has_mesh):
    pv = pred_verts[has_mesh == 1]
    gv = gt_verts[has_mesh == 1]
    if len(gv) == 0:
        return torch.zeros(1, device=pred_verts.device, requires_grad=True).squeeze()
    return criterion(pv, gv)


def keypoint_2d_loss(criterion, pred_kp2d, gt_kp2d, has_kp2d):
    conf = gt_kp2d[:, :, -1:].clone()
    return (conf * criterion(pred_kp2d, gt_kp2d[:, :, :-1])).mean()


# ─────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────


def save_checkpoint(model, args, epoch, step):
    if args.local_rank != 0:
        return
    ckpt_dir = op.join(args.output_dir, f"checkpoint-epoch{epoch:04d}-step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), op.join(ckpt_dir, "state_dict.bin"))
    torch.save(args, op.join(ckpt_dir, "training_args.bin"))
    logger.info(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir


# ─────────────────────────────────────────────
#  Adjust LR
# ─────────────────────────────────────────────


def adjust_lr(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs / 2.0)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────


def run_train(args, train_loader, model, mano, mesh_sampler):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0
    )
    crit_kp = nn.MSELoss(reduction="none").to(args.device)
    crit_vert = nn.L1Loss().to(args.device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    max_iter = len(train_loader) * args.num_train_epochs
    iters_per_ep = len(train_loader)
    global_step = 0
    t0 = time.time()

    logger.info(
        f"Start training: {args.num_train_epochs} epochs, {iters_per_ep} iters/epoch"
    )

    for epoch in range(args.num_train_epochs):
        model.train()
        adjust_lr(optimizer, epoch, args)

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            images = batch["image"].to(args.device)
            gt_v = batch["vertices"].to(args.device)  # (B, 778, 3) in meters
            gt_kp3d = batch["joints_3d"].to(args.device)  # (B, 21, 4)  last=conf
            gt_kp2d = batch["joints_2d"].to(args.device)  # (B, 21, 3)  last=conf
            has_mesh = batch["has_mesh"].to(args.device)  # (B,)
            has_kp3d = has_mesh

            # ── down-sample GT vertices ─────────────────────────
            gt_v_sub = mesh_sampler.downsample(gt_v)

            # ── root-normalize GT ───────────────────────────────
            wrist_idx = 0  # index 0 = Wrist in MANO joint order
            root = gt_kp3d[:, wrist_idx, :3].clone()
            gt_v = gt_v - root[:, None, :]
            gt_v_sub = gt_v_sub - root[:, None, :]

            # ── forward ─────────────────────────────────────────
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                pred_cam, pred_kp3d, pred_v_sub, pred_v = model(
                    images, mano, mesh_sampler, is_train=True
                )

            pred_kp3d_from_mesh = mano.get_3d_joints(pred_v)
            pred_kp2d_from_mesh = orthographic_projection(pred_kp3d_from_mesh, pred_cam)
            pred_kp2d = orthographic_projection(pred_kp3d, pred_cam)

            # ── losses ──────────────────────────────────────────
            l_kp3d = keypoint_3d_loss(crit_kp, pred_kp3d, gt_kp3d, has_kp3d)
            l_kp3d_mesh = keypoint_3d_loss(
                crit_kp, pred_kp3d_from_mesh, gt_kp3d, has_kp3d
            )
            l_vert = args.vloss_w_sub * vertices_loss(
                crit_vert, pred_v_sub, gt_v_sub, has_mesh
            ) + args.vloss_w_full * vertices_loss(crit_vert, pred_v, gt_v, has_mesh)
            l_kp2d = keypoint_2d_loss(
                crit_kp, pred_kp2d, gt_kp2d, has_mesh
            ) + keypoint_2d_loss(crit_kp, pred_kp2d_from_mesh, gt_kp2d, has_mesh)

            loss = (
                args.joints_loss_weight * (l_kp3d + l_kp3d_mesh)
                + args.vertices_loss_weight * l_vert
                + args.vertices_loss_weight * l_kp2d
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % args.logging_steps == 0 and args.local_rank == 0:
                elapsed = time.time() - t0
                eta = elapsed / global_step * (max_iter - global_step)
                logger.info(
                    f"Epoch {epoch+1}/{args.num_train_epochs} "
                    f"Step {batch_idx+1}/{iters_per_ep} | "
                    f"loss={loss.item():.4f} "
                    f"kp3d={l_kp3d.item():.4f} "
                    f"vert={l_vert.item():.4f} "
                    f"2d={l_kp2d.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"ETA={str(datetime.timedelta(seconds=int(eta)))}"
                )

        # ── per-epoch checkpoint ─────────────────────────────
        if (
            epoch + 1
        ) % args.save_every_n_epochs == 0 or epoch == args.num_train_epochs - 1:
            save_checkpoint(model, args, epoch + 1, global_step)

    logger.info("Training finished.")
    save_checkpoint(model, args, args.num_train_epochs, global_step)


# ─────────────────────────────────────────────
#  Evaluation + save results
# ─────────────────────────────────────────────


def run_eval(args, val_loader, model, mano, mesh_sampler):
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    model.eval()

    mesh_output_save = []
    joint_output_save = []
    fname_save = []

    # Directory for per-sample OBJ exports
    obj_dir = op.join(args.output_dir, "obj_exports")
    if args.local_rank == 0:
        os.makedirs(obj_dir, exist_ok=True)

    faces = mano.face  # (F, 3) numpy int array

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(args.device)
            fnames = batch.get(
                "fname",
                [
                    str(i * args.per_gpu_eval_batch_size + j)
                    for j in range(images.shape[0])
                ],
            )

            pred_cam, pred_kp3d, pred_v_sub, pred_v = model(
                images, mano, mesh_sampler, is_train=False
            )

            pred_kp3d_from_mesh = mano.get_3d_joints(pred_v)

            # Root-normalize (wrist = origin)
            wrist = pred_kp3d_from_mesh[:, 0:1, :]
            pred_v = pred_v - wrist
            pred_kp3d_from_mesh = pred_kp3d_from_mesh - wrist

            for j in range(images.shape[0]):
                verts = pred_v[j].cpu().numpy()  # (778, 3)
                joints = pred_kp3d_from_mesh[j].cpu().numpy()  # (21, 3)
                fname = (
                    fnames[j] if isinstance(fnames[j], str) else str(fnames[j].item())
                )

                mesh_output_save.append(verts.tolist())
                joint_output_save.append(joints.tolist())
                fname_save.append(fname)

                # Export OBJ + MTL for every sample (or use --export_obj_every N)
                if (
                    args.export_obj
                    and (i * args.per_gpu_eval_batch_size + j) % args.export_obj_every
                    == 0
                ):
                    stem = (
                        op.splitext(op.basename(fname))[0]
                        if "/" in fname or "." in fname
                        else fname
                    )
                    obj_path = op.join(obj_dir, f"{stem}.obj")
                    mtl_path = op.join(obj_dir, f"{stem}.mtl")
                    export_obj_mtl(verts, faces, obj_path, mtl_path)

            if i % 50 == 0:
                logger.info(f"Eval [{i}/{len(val_loader)}]")

    # Save pred.json (FreiHAND submission format)
    if args.local_rank == 0:
        pred_path = op.join(args.output_dir, "pred.json")
        with open(pred_path, "w") as f:
            json.dump([joint_output_save, mesh_output_save], f)
        logger.info(f"Saved pred.json → {pred_path}  ({len(mesh_output_save)} samples)")

        # Zip for leaderboard submission
        zip_path = op.join(args.output_dir, "freihand_pred.zip")
        os.system(f"zip {zip_path} {pred_path}")
        logger.info(f"Zipped → {zip_path}")


# ─────────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser("FastMETRO FreiHAND Training")

    # Data
    p.add_argument("--data_root", default="./data/FreiHAND", type=str)
    p.add_argument(
        "--mano_dir",
        default="./models/mano",
        type=str,
        help="Directory containing MANO_RIGHT.pkl",
    )
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--img_size", default=224, type=int)

    # Checkpoints
    p.add_argument("--output_dir", default="./output/freihand", type=str)
    p.add_argument("--resume_checkpoint", default=None, type=str)
    p.add_argument("--save_every_n_epochs", default=10, type=int)

    # Model
    p.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        choices=["resnet50", "hrnet-w64"],
        help="CNN backbone. hrnet-w64 is stronger but needs pretrained weights.",
    )
    p.add_argument(
        "--model_name",
        default="FastMETRO-S",
        type=str,
        choices=["FastMETRO-S", "FastMETRO-L"],
    )
    p.add_argument("--input_feat_dim", default="2051,512,128", type=str)
    p.add_argument("--hidden_feat_dim", default="1024,256,64", type=str)
    p.add_argument("--drop_out", default=0.1, type=float)

    # Training
    p.add_argument("--per_gpu_train_batch_size", default=32, type=int)
    p.add_argument("--per_gpu_eval_batch_size", default=32, type=int)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--num_train_epochs", default=200, type=int)
    p.add_argument("--vertices_loss_weight", default=1.0, type=float)
    p.add_argument("--joints_loss_weight", default=1.0, type=float)
    p.add_argument("--vloss_w_full", default=0.5, type=float)
    p.add_argument("--vloss_w_sub", default=0.5, type=float)
    p.add_argument("--logging_steps", default=100, type=int)

    # Eval / export
    p.add_argument("--run_eval_only", action="store_true")
    p.add_argument(
        "--export_obj",
        action="store_true",
        help="Export predicted meshes as OBJ/MTL files during eval",
    )
    p.add_argument(
        "--export_obj_every",
        default=1,
        type=int,
        help="Export OBJ every N samples (1 = all)",
    )

    # Distributed
    p.add_argument("--local_rank", default=0, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--device", default="cuda", type=str)

    return p.parse_args()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────


def main():
    args = parse_args()

    # ── distributed setup ───────────────────────────────────
    args.num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(op.join(args.output_dir, "train.log")),
        ],
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Args: {args}")

    # ── MANO + Mesh sampler ─────────────────────────────────
    mano = MANOWrapper(mano_dir=args.mano_dir).to(args.device)
    mesh_sampler = Mesh()

    # ── Build model ─────────────────────────────────────────
    input_feat_dim = [int(x) for x in args.input_feat_dim.split(",")]
    hidden_feat_dim = [int(x) for x in args.hidden_feat_dim.split(",")]

    model = build_fastmetro(args, input_feat_dim, hidden_feat_dim)
    model.to(args.device)

    if args.resume_checkpoint:
        logger.info(f"Loading checkpoint: {args.resume_checkpoint}")
        state = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)

    # ── Datasets ─────────────────────────────────────────────
    if not args.run_eval_only:
        train_dataset = FreiHandDataset(
            root=args.data_root, split="training", img_size=args.img_size, augment=True
        )
        train_sampler = (
            torch.utils.data.distributed.DistributedSampler(train_dataset)
            if args.distributed
            else None
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.per_gpu_train_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_dataset = FreiHandDataset(
        root=args.data_root, split="evaluation", img_size=args.img_size, augment=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Run ───────────────────────────────────────────────────
    if args.run_eval_only:
        logger.info("=== Evaluation only mode ===")
        run_eval(args, val_loader, model, mano, mesh_sampler)
    else:
        logger.info("=== Training ===")
        run_train(args, train_loader, model, mano, mesh_sampler)
        # logger.info("=== Post-training evaluation ===")
        # model_eval = model.module if hasattr(model, "module") else model
        # run_eval(args, val_loader, model_eval, mano, mesh_sampler)


if __name__ == "__main__":
    main()
