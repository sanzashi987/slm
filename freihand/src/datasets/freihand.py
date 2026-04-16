# ----------------------------------------------------------------------------------------------
# Minimal FreiHAND dataset
# Expected directory layout:
#   <root>/
#     training/
#       rgb/         00000000.jpg ... 00032559.jpg  (x4 augmentation copies -> 130240 images)
#     evaluation/
#       rgb/         00000000.jpg ...
#     training_K.json
#     training_mano.json
#     training_xyz.json
#     evaluation_K.json
#     evaluation_scale.json (optional)
# ----------------------------------------------------------------------------------------------

import json
import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet normalization for ResNet50 / HRNet pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


class FreiHANDDataset(Dataset):
    """
    FreiHAND dataset.

    The FreiHAND v2 training set has 32,560 unique samples that are replicated 4 times with
    different backgrounds -> 130,240 images in ./training/rgb.  Annotations (K, MANO, xyz) are
    length 32,560; image idx `i` corresponds to annotation idx `i % 32560`.
    """

    _NUM_UNIQUE_TRAIN = 32560

    def __init__(self, root, split="train", image_size=224):
        assert split in ("train", "eval")
        self.root = root
        self.split = split
        self.image_size = image_size

        if split == "train":
            self.img_dir = os.path.join(root, "training", "rgb")
            self.K_list = _load_json(os.path.join(root, "training_K.json"))
            self.mano_list = _load_json(os.path.join(root, "training_mano.json"))
            self.xyz_list = _load_json(os.path.join(root, "training_xyz.json"))
            # All 130,240 images - each unique sample has 4 versions
            self.num_samples = len(os.listdir(self.img_dir))
        else:
            self.img_dir = os.path.join(root, "evaluation", "rgb")
            self.K_list = _load_json(os.path.join(root, "evaluation_K.json"))
            self.mano_list = None
            self.xyz_list = None
            self.num_samples = len(os.listdir(self.img_dir))

        # ImageNet normalization + resize to 224
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return self.num_samples

    def _unique_idx(self, idx):
        """Map image index (0..130239) to annotation index (0..32559)."""
        return idx % self._NUM_UNIQUE_TRAIN

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx:08d}.jpg")
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # 224 x 224 for FreiHAND
        img_tensor = self.transform(img)

        if self.split == "eval":
            # Inference only - annotations not needed
            K = torch.tensor(self.K_list[idx], dtype=torch.float32)
            return {
                "img_key": f"{idx:08d}",
                "image": img_tensor,
                "K": K,
            }

        u = self._unique_idx(idx)
        mano_params = np.asarray(self.mano_list[u], dtype=np.float32).reshape(-1)  # length 61
        # FreiHAND mano: [48 pose, 10 shape, 3 trans] (first 3 of pose are global rotation)
        pose = torch.from_numpy(mano_params[:48]).float()           # 48
        betas = torch.from_numpy(mano_params[48:58]).float()        # 10
        K = torch.tensor(self.K_list[u], dtype=torch.float32)       # 3x3
        xyz = torch.tensor(self.xyz_list[u], dtype=torch.float32)   # 21x3  (meters, camera space)

        # 2D joints via camera projection
        # xyz is in camera space (meters). Project: pixel = K @ (xyz / z)
        xyz_np = xyz.numpy()
        uv = (xyz_np @ K.numpy().T)
        uv = uv[:, :2] / uv[:, 2:3]
        # Normalize to [-1, 1] range (this is the convention used by FastMETRO for L1 2D loss
        # vs orthographic-projected predictions).
        uv_norm = np.zeros_like(uv, dtype=np.float32)
        uv_norm[:, 0] = 2.0 * uv[:, 0] / orig_w - 1.0
        uv_norm[:, 1] = 2.0 * uv[:, 1] / orig_h - 1.0
        joints_2d = torch.zeros(21, 3, dtype=torch.float32)
        joints_2d[:, :2] = torch.from_numpy(uv_norm)
        joints_2d[:, 2] = 1.0  # visibility flag

        return {
            "img_key": f"{idx:08d}",
            "image": img_tensor,
            "pose": pose,          # 48
            "betas": betas,        # 10
            "joints_2d": joints_2d,  # 21 x 3
            "K": K,
            "xyz": xyz,            # for eval/debug only
        }
