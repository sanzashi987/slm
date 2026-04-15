"""
FreiHAND Dataset Loader
数据集结构 (下载自 https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html):

  FreiHAND/
    training/
      rgb/         <- 00000000.jpg ... (130240 images, 4 augmentation versions)
      mask/        <- 00000000.jpg ...
    evaluation/
      rgb/         <- 00000000.jpg ... (3960 images)
    training_K.json      <- (N, 3, 3) camera intrinsics
    training_mano.json   <- (N,) MANO params {pose, shape}
    training_xyz.json    <- (N, 21, 3) 3D joint positions (in camera space, meters)
    evaluation_K.json
    evaluation_scals.json

training 的 130240 张是 32560 张 × 4 种增强 (颜色抖动等), 本 dataloader 视作 130240 独立样本.
"""

import os
import os.path as op
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class FreiHandDataset(Dataset):
    """
    Parameters
    ----------
    root    : path to FreiHAND root (contains training/ evaluation/ *.json)
    split   : 'training' | 'evaluation'
    img_size: resize target (square), default 224
    augment : apply random augmentations during training
    """

    JOINT_NUM   = 21
    VERTEX_NUM  = 778  # MANO mesh vertices

    def __init__(self, root: str, split: str = "training",
                 img_size: int = 224, augment: bool = True):
        super().__init__()
        assert split in ("training", "evaluation"), \
            f"split must be 'training' or 'evaluation', got '{split}'"

        self.root     = root
        self.split    = split
        self.img_size = img_size
        self.augment  = augment and (split == "training")

        # ── Load annotations ────────────────────────────────
        k_path = op.join(root, f"{split}_K.json")
        with open(k_path) as f:
            self.K_list = np.array(json.load(f), dtype=np.float32)  # (N, 3, 3)

        if split == "training":
            xyz_path  = op.join(root, "training_xyz.json")
            mano_path = op.join(root, "training_mano.json")
            with open(xyz_path) as f:
                self.xyz_list = np.array(json.load(f), dtype=np.float32)  # (N, 21, 3) meters
            with open(mano_path) as f:
                mano_raw = json.load(f)
            self.mano_pose  = np.array([m[0] for m in mano_raw], dtype=np.float32)  # (N, 48)
            self.mano_shape = np.array([m[1] for m in mano_raw], dtype=np.float32)  # (N, 10)
        else:
            # Evaluation set: no GT mesh; scals only (scale factors per sample)
            scals_path = op.join(root, "evaluation_scals.json")
            with open(scals_path) as f:
                self.scals = np.array(json.load(f), dtype=np.float32)

        self.img_dir = op.join(root, split, "rgb")
        self.n       = len(self.K_list)

        # ── Transforms ───────────────────────────────────────
        norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                           std= [0.229, 0.224, 0.225])
        if self.augment:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomHorizontalFlip(p=0.0),  # keep off: MANO is right-hand only
                T.ToTensor(),
                norm,
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                norm,
            ])

    def __len__(self) -> int:
        return self.n

    def _load_image(self, idx: int) -> Image.Image:
        fname = f"{idx:08d}.jpg"
        path  = op.join(self.img_dir, fname)
        img   = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> dict:
        img   = self._load_image(idx)
        K     = self.K_list[idx]           # (3, 3) intrinsics
        image = self.transform(img)        # (3, H, W)

        item = {"image": image, "fname": f"{idx:08d}"}

        if self.split == "training":
            xyz  = self.xyz_list[idx].copy()  # (21, 3)  joint positions in camera space (m)
            pose = self.mano_pose[idx]
            beta = self.mano_shape[idx]

            # ── Project 3D joints → 2D using K ───────────────
            # xyz in meters → pixel coords, then normalize to [-1, 1]
            w, h = img.size
            xy_proj  = xyz @ K.T                               # (21, 3) homogeneous
            xy_proj  = xy_proj[:, :2] / (xy_proj[:, 2:3] + 1e-8)  # (21, 2) pixels
            xy_norm  = xy_proj / np.array([[w, h]], dtype=np.float32) * 2 - 1  # [-1,1]

            # Confidence = 1 for all GT joints
            conf_2d  = np.ones((self.JOINT_NUM, 1), dtype=np.float32)
            kp2d     = np.concatenate([xy_norm, conf_2d], axis=-1)  # (21, 3)

            conf_3d  = np.ones((self.JOINT_NUM, 1), dtype=np.float32)
            kp3d     = np.concatenate([xyz, conf_3d], axis=-1)       # (21, 4)

            item.update({
                "joints_2d": torch.from_numpy(kp2d),            # (21, 3)
                "joints_3d": torch.from_numpy(kp3d),            # (21, 4)
                "vertices":  torch.zeros(self.VERTEX_NUM, 3),   # placeholder; filled by MANO in trainer
                "has_mesh":  torch.tensor(1, dtype=torch.float32),
                "pose":      torch.from_numpy(pose),            # (48,)
                "betas":     torch.from_numpy(beta),            # (10,)
                "K":         torch.from_numpy(K),               # (3,3)
            })
        else:
            # Evaluation: no GT joints / mesh
            item.update({
                "joints_2d": torch.zeros(self.JOINT_NUM, 3),
                "joints_3d": torch.zeros(self.JOINT_NUM, 4),
                "vertices":  torch.zeros(self.VERTEX_NUM, 3),
                "has_mesh":  torch.tensor(0, dtype=torch.float32),
                "K":         torch.from_numpy(K),
            })

        return item
