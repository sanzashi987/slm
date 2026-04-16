# FastMETRO Minimal — FreiHAND on a single RTX 4090

Minimal, single-GPU training code for **FastMETRO** (ECCV'22) on **FreiHAND**, distilled from the official [kaist-ami/FastMETRO](https://github.com/kaist-ami/FastMETRO) repo and tuned specifically for one **RTX 4090** (24GB, Ada Lovelace).

**Stripped vs upstream:**
- No distributed training (single GPU).
- No TSV preprocessing — reads FreiHAND's raw directory directly.
- No HRNet-W64 backbone — uses torchvision ResNet-50 (ImageNet weights auto-downloaded).
- No OpenDR / PyRender — hardest-to-install deps removed.
- No SMPL body network / param regressor — hand-only.

**Added:**
- OBJ + MTL exporter.
- End-to-end inference script: image → `.obj` / `.mtl`.

---

## Model & config (fixed for cost efficiency on 4090)

| Component | Value | Notes |
|---|---|---|
| Backbone | ResNet-50 (~25M) | torchvision `IMAGENET1K_V2` weights |
| Transformer layers | 1 enc + 1 dec (×2 stages = 4) | `FastMETRO-S` |
| `model_dim_1` / `model_dim_2` | 512 / 128 | |
| Total trainable params | ~32.7M | |
| Batch size | **96** | fits 4090 with bf16 |
| Num workers | 8, `prefetch_factor=4` | |
| Optimizer | AdamW, lr=1e-4, wd=1e-4 | |
| LR schedule | StepLR, ×0.1 at epoch 80 | |
| Epochs | 100 | |

### 4090-specific optimizations (all default-on, no flags)

1. **bfloat16 autocast** around forward + loss. No `GradScaler` (bf16 range == fp32).
2. **TF32** for non-autocast fp32 paths (`matmul.allow_tf32 = True`, `cudnn.allow_tf32 = True`).
3. **`cudnn.benchmark = True`** — input is fixed 224×224.
4. **`channels_last`** memory format for the CNN backbone.
5. **Vectorized mesh sampler** — single sparse matmul, no per-sample python loop.
6. **`torch.no_grad()`** around MANO GT mesh regeneration.

Combined speedup over a naive fp32 implementation: roughly **2-2.5×** throughput, with VRAM at batch 96 around **14-16 GB**.

Expected: one epoch ≈ **3-5 min** on 4090 for FreiHAND (130,240 images). Full 100 epochs ≈ **6-9 h**.

---

## Project layout

```
fastmetro_min/
├── requirements.txt
├── scripts/train.sh
└── src/
    ├── train_freihand.py          # main training script
    ├── export_obj.py              # inference: checkpoint + image → .obj/.mtl
    ├── datasets/freihand.py       # FreiHAND raw directory reader
    ├── modeling/
    │   ├── mano_utils.py          # MANO wrapper + vectorized mesh sampler
    │   ├── data/                  # mano_195_adjmat_*.pt, mano_downsampling.npz
    │   └── model/
    │       ├── modeling_fastmetro_hand.py
    │       ├── transformer.py     # (from upstream)
    │       └── position_encoding.py         # (from upstream)
    └── utils/mesh_io.py           # OBJ + MTL writer
```

---

## 1. Install

```bash
conda create -n fastmetro python=3.10 -y
conda activate fastmetro

# PyTorch for CUDA 12.1 (4090 works well with CUDA 12.x):
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# MANO layer - requires numpy<2 because manopth pulls chumpy.
pip install "git+https://github.com/hassony2/manopth.git"
```

---

## 2. Data preparation

### FreiHAND

Download from <https://lmb.informatik.uni-freiburg.de/projects/freihand/>, unpack so you have:

```
FreiHAND_pub_v2/
├── training/rgb/00000000.jpg ... 00130239.jpg      # 32,560 unique × 4 background aug
├── evaluation/rgb/
├── training_K.json                 # 32,560 × 3×3
├── training_mano.json              # 32,560 × 61   (48 pose + 10 shape + 3 trans)
├── training_xyz.json               # 32,560 × 21×3 (meters, camera space)
└── evaluation_K.json
```

### MANO

1. Register at <https://mano.is.tue.mpg.de/> and download MANO v1.2.
2. Unpack. The folder containing `MANO_RIGHT.pkl` (typically `mano_v1_2/models/`) is your `--mano_dir`.

---

## 3. Train

```bash
python -m src.train_freihand \
    --freihand_dir /path/to/FreiHAND_pub_v2 \
    --mano_dir     /path/to/mano_v1_2/models \
    --output_dir   ./outputs/run1
```

Or edit paths in `scripts/train.sh` and run it.

### What gets saved (in `--output_dir`)

- `train.log` — full training log
- `checkpoint_latest.pth` — most recent checkpoint (default: every 5 epochs)
- `checkpoint_epoch005.pth`, ... — periodic snapshots
- `meshes_epoch005/sample_*.obj` + `.mtl` — sanity-check meshes exported after each save

Each checkpoint `.pth` contains:
```python
{"epoch": int, "model": state_dict, "optimizer": state_dict, "lr_scheduler": state_dict}
```

### Resume

```bash
python -m src.train_freihand ... --resume ./outputs/run1/checkpoint_latest.pth
```

---

## 4. Export OBJ / MTL from a trained model

Inference uses the same bf16 autocast + channels_last path as training for consistency.

```bash
# Single image
python -m src.export_obj \
    --checkpoint ./outputs/run1/checkpoint_latest.pth \
    --mano_dir   /path/to/mano_v1_2/models \
    --input      ./my_hand.jpg \
    --output_dir ./exports

# Folder of images
python -m src.export_obj \
    --checkpoint ./outputs/run1/checkpoint_latest.pth \
    --mano_dir   /path/to/mano_v1_2/models \
    --input      /path/to/FreiHAND_pub_v2/evaluation/rgb \
    --output_dir ./exports \
    --limit      20
```

Each image → one `.obj` + one matching `.mtl` (simple flesh-colored Lambert material). Both files open directly in MeshLab / Blender / macOS Preview.

---

## Troubleshooting

**OOM at batch 96**: drop `--batch_size 64` (back to the pre-bf16 default) as a safety margin.

**Loss explodes / becomes NaN**: very unlikely with bf16 on Ada, but if it happens, check that `mano_dir` contains the correct `MANO_RIGHT.pkl` — corrupt GT mesh is the #1 cause.

**Slow data loading (throughput bottlenecked below GPU capacity)**: try `--num_workers 12` and/or `--prefetch_factor 6`. The FreiHAND images are tiny (224×224 JPEGs) so SSD read rarely matters; CPU decode is the usual bottleneck.

**`torch.load` warning about `weights_only`**: safe to ignore on your own checkpoints.

---

## License & credits

Derived from [kaist-ami/FastMETRO](https://github.com/kaist-ami/FastMETRO) (MIT). Files `transformer.py`, `position_encoding.py`, and the MANO adjacency matrices / downsampling `.npz` are reused verbatim. Model wrapper, dataset, training loop, and OBJ exporter are rewritten / simplified.

Cite the original paper if you use this code:
```
Cho et al., "Cross-Attention of Disentangled Modalities for 3D Human Mesh Recovery with Transformers", ECCV 2022.
```
