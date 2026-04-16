# ----------------------------------------------------------------------------------------------
# Run a trained FastMETRO checkpoint on a single image (or a folder of images)
# and export each predicted hand mesh as .obj + .mtl.
#
# Usage:
#   python -m src.export_obj \
#       --checkpoint ./outputs/run1/checkpoint_latest.pth \
#       --mano_dir   /path/to/mano_v1_2/models \
#       --input      ./some_hand_image.jpg \
#       --output_dir ./exports
#
#   python -m src.export_obj \
#       --checkpoint ./outputs/run1/checkpoint_latest.pth \
#       --mano_dir   /path/to/mano_v1_2/models \
#       --input      /path/to/FreiHAND_pub_v2/evaluation/rgb \
#       --output_dir ./exports \
#       --limit      20
# ----------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import argparse
import os
import os.path as op
from glob import glob

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
import torchvision.models as tv_models

from src.modeling.mano_utils import MANO, Mesh
from src.modeling.model import FastMETRO_Hand_Network
from src.utils.mesh_io import save_mesh_obj

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet50_backbone():
    m = tv_models.resnet50(weights=None)  # weights come from the checkpoint
    return nn.Sequential(*list(m.children())[:-2])


def make_args_namespace(model_name="FastMETRO-S", model_data_dir="src/modeling/data"):
    """Reconstruct the minimal namespace needed by FastMETRO_Hand_Network."""
    ns = argparse.Namespace(
        model_name=model_name,
        model_dim_1=512, model_dim_2=128,
        feedforward_dim_1=2048, feedforward_dim_2=512,
        conv_1x1_dim=2048,
        transformer_dropout=0.1, transformer_nhead=8, pos_type="sine",
        model_data_dir=model_data_dir,
    )
    return ns


def load_image(path, size=224):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return t(img)


def collect_inputs(inp):
    if op.isdir(inp):
        paths = sorted(glob(op.join(inp, "*.jpg")) + glob(op.join(inp, "*.png")))
    else:
        paths = [inp]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--mano_dir", type=str, required=True)
    ap.add_argument("--input", type=str, required=True,
                    help="Path to an image OR a folder of images (*.jpg/*.png).")
    ap.add_argument("--output_dir", type=str, default="./exports")
    ap.add_argument("--model_name", type=str, default="FastMETRO-S",
                    choices=["FastMETRO-S", "FastMETRO-M", "FastMETRO-L"])
    ap.add_argument("--model_data_dir", type=str, default="src/modeling/data")
    ap.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model with the same config used for training
    ns = make_args_namespace(model_name=args.model_name, model_data_dir=args.model_data_dir)
    mano_model = MANO(args.mano_dir).to(device)
    mesh_sampler = Mesh(sampling_npz_path=op.join(args.model_data_dir, "mano_downsampling.npz"),
                        device=device)
    backbone = build_resnet50_backbone()
    model = FastMETRO_Hand_Network(ns, backbone, mesh_sampler).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model.eval()
    faces = mano_model.face

    paths = collect_inputs(args.input)
    if args.limit > 0:
        paths = paths[:args.limit]
    print(f"Processing {len(paths)} image(s)...")

    with torch.no_grad():
        for p in paths:
            x = load_image(p).unsqueeze(0).to(device)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(x)
            else:
                out = model(x)
            verts = out["pred_3d_vertices_fine"][0].float().cpu().numpy()
            stem = op.splitext(op.basename(p))[0]
            obj_path = op.join(args.output_dir, f"{stem}.obj")
            save_mesh_obj(verts, faces, obj_path)
            print(f"  {p} -> {obj_path}")

    print("Done.")


if __name__ == "__main__":
    main()
