"""
单张图片推理脚本
用法:
    python infer.py \
        --image path/to/hand.jpg \
        --checkpoint output/freihand/checkpoint-epoch0200-step.../state_dict.bin \
        --mano_dir models/mano \
        --output_dir output/infer \
        --arch resnet50 \
        --model_name FastMETRO-S \
        --export_obj

输出:
    output/infer/hand.obj
    output/infer/hand.mtl
"""

import argparse
import os
import os.path as op
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from src.modeling.fastmetro import build_fastmetro
from src.utils.mano_wrapper import MANOWrapper
from src.utils.mesh_sampler import Mesh
from src.utils.obj_exporter import export_obj_mtl


def parse_args():
    p = argparse.ArgumentParser("FastMETRO Single Image Inference")
    p.add_argument("--image",       required=True, type=str)
    p.add_argument("--checkpoint",  required=True, type=str)
    p.add_argument("--mano_dir",    default="./models/mano", type=str)
    p.add_argument("--output_dir",  default="./output/infer", type=str)
    p.add_argument("--arch",        default="resnet50", type=str)
    p.add_argument("--model_name",  default="FastMETRO-S", type=str)
    p.add_argument("--img_size",    default=224, type=int)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--input_feat_dim",  default="2051,512,128", type=str)
    p.add_argument("--hidden_feat_dim", default="1024,256,64",  type=str)
    p.add_argument("--drop_out",        default=0.1, type=float)
    p.add_argument("--export_obj",  action="store_true", default=True)
    return p.parse_args()


def preprocess(image_path: str, img_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Setup ────────────────────────────────────────────
    mano         = MANOWrapper(mano_dir=args.mano_dir).to(device)
    mesh_sampler = Mesh()

    input_feat_dim  = [int(x) for x in args.input_feat_dim.split(",")]
    hidden_feat_dim = [int(x) for x in args.hidden_feat_dim.split(",")]

    model = build_fastmetro(args, input_feat_dim, hidden_feat_dim)
    model.to(device)

    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Inference ────────────────────────────────────────
    image = preprocess(args.image, args.img_size).to(device)

    with torch.no_grad():
        pred_cam, pred_kp3d, pred_v_sub, pred_v = model(
            image, mano, mesh_sampler, is_train=False)

        # Root-normalize (wrist = origin)
        pred_kp3d_from_mesh = mano.get_3d_joints(pred_v)
        wrist = pred_kp3d_from_mesh[:, 0:1, :]
        pred_v = pred_v - wrist
        pred_kp3d_from_mesh = pred_kp3d_from_mesh - wrist

    vertices = pred_v[0].cpu().numpy()    # (778, 3)
    joints   = pred_kp3d_from_mesh[0].cpu().numpy()  # (21, 3)
    faces    = mano.face

    print(f"Predicted vertices: {vertices.shape}, range: [{vertices.min():.3f}, {vertices.max():.3f}]")
    print(f"Predicted joints:   {joints.shape}")

    # ── Export OBJ/MTL ───────────────────────────────────
    stem     = op.splitext(op.basename(args.image))[0]
    obj_path = op.join(args.output_dir, f"{stem}.obj")
    mtl_path = op.join(args.output_dir, f"{stem}.mtl")

    if args.export_obj:
        export_obj_mtl(vertices, faces, obj_path, mtl_path)
        print(f"Saved OBJ → {obj_path}")
        print(f"Saved MTL → {mtl_path}")

    # ── Save joints as numpy ─────────────────────────────
    np.save(op.join(args.output_dir, f"{stem}_joints.npy"), joints)
    np.save(op.join(args.output_dir, f"{stem}_vertices.npy"), vertices)
    print("Done.")


if __name__ == "__main__":
    main()
