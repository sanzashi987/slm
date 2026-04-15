# FastMETRO FreiHAND — 最小可用训练代码

基于 [kaist-ami/FastMETRO](https://github.com/kaist-ami/FastMETRO) (ECCV 2022) 改写。  
支持 **FreiHAND 数据集训练**、**检查点保存**、**OBJ/MTL 格式导出**。

---

## 文件结构

```
fastmetro_freihand/
├── train_freihand.py          ← 主训练入口
├── infer.py                   ← 单图推理 + OBJ 导出
├── requirements.txt
├── src/
│   ├── modeling/
│   │   └── fastmetro.py       ← 模型定义 (Encoder-Decoder Transformer)
│   ├── datasets/
│   │   └── freihand_dataset.py ← FreiHAND 数据加载
│   └── utils/
│       ├── mano_wrapper.py    ← MANO 模型封装
│       ├── mesh_sampler.py    ← 网格下/上采样
│       ├── geometric_layers.py← 弱透视投影
│       └── obj_exporter.py    ← OBJ/MTL 导出
└── README.md
```

---

## 环境配置

```bash
# 1. 克隆 FastMETRO (获取 manopth 子模块)
git clone --recursive https://github.com/kaist-ami/FastMETRO.git
cd FastMETRO

# 2. 将本目录的文件覆盖/添加到 FastMETRO 目录
#    (或直接在本目录操作, 需保证 manopth 在 Python 路径中)

# 3. 安装依赖
pip install -r requirements.txt
pip install -e manopth/
```

---

## 数据准备

### FreiHAND 数据集

1. 注册并下载: https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
2. 解压后目录结构:

```
data/FreiHAND/
├── training/
│   └── rgb/          ← 130240 张 .jpg
├── evaluation/
│   └── rgb/          ← 3960 张 .jpg
├── training_K.json
├── training_mano.json
├── training_xyz.json
├── evaluation_K.json
└── evaluation_scals.json
```

### MANO 模型

1. 注册并下载: https://mano.is.tue.mpg.de/
2. 将 `MANO_RIGHT.pkl` 放入 `models/mano/`

### 网格采样矩阵 (可选, 提升精度)

从 FastMETRO 或 METRO 仓库的 `src/modeling/data/` 目录获取 `mesh_downsampling.npz`，  
放入 `src/modeling/data/`。  
若未提供, 代码会使用近似 FPS 采样 (功能正常, 精度略低)。

---

## 训练

### 单卡训练 (ResNet-50 backbone, 快速验证)

```bash
python train_freihand.py \
    --data_root ./data/FreiHAND \
    --mano_dir  ./models/mano \
    --output_dir ./output/freihand_r50 \
    --arch resnet50 \
    --model_name FastMETRO-S \
    --per_gpu_train_batch_size 32 \
    --num_train_epochs 200 \
    --lr 1e-4 \
    --save_every_n_epochs 10
```

### 多卡训练 (HRNet-W64, 官方配置)

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
    train_freihand.py \
    --data_root ./data/FreiHAND \
    --mano_dir  ./models/mano \
    --output_dir ./output/freihand_h64 \
    --arch hrnet-w64 \
    --model_name FastMETRO-L \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs 200 \
    --lr 1e-4
```

训练完成后会自动运行验证集推理，输出:
- `output/freihand_*/pred.json`         ← FreiHAND 官方提交格式
- `output/freihand_*/freihand_pred.zip` ← 直接上传到 Codalab 的 zip

---

## 仅推理 + 导出 OBJ

```bash
# 从验证集批量导出 OBJ
python train_freihand.py \
    --data_root ./data/FreiHAND \
    --mano_dir  ./models/mano \
    --output_dir ./output/eval \
    --resume_checkpoint ./output/freihand_r50/checkpoint-epoch0200-step.../state_dict.bin \
    --run_eval_only \
    --export_obj \
    --export_obj_every 10   # 每10个样本导出一个 OBJ

# 从单张图片推理
python infer.py \
    --image ./test_images/hand.jpg \
    --checkpoint ./output/freihand_r50/checkpoint-epoch0200-step.../state_dict.bin \
    --mano_dir ./models/mano \
    --output_dir ./output/infer \
    --export_obj
```

---

## OBJ/MTL 格式说明

每个导出样本生成两个文件:

| 文件 | 内容 |
|------|------|
| `sample.obj` | 778 顶点坐标 (米制), 1538 三角面, 引用 MTL |
| `sample.mtl` | Phong 着色, 皮肤色 RGB, 无需纹理贴图 |

可直接在 Blender、MeshLab、Three.js 等工具中打开。

自定义颜色:
```python
from src.utils.obj_exporter import export_obj_mtl
export_obj_mtl(vertices, faces, "hand.obj", color_rgb=(1.0, 0.5, 0.3))
```

---

## 关键超参数

| 参数 | FastMETRO-S | FastMETRO-L |
|------|------------|------------|
| `--arch` | `resnet50` | `hrnet-w64` |
| `--model_name` | `FastMETRO-S` | `FastMETRO-L` |
| `--hidden_feat_dim` | `512,128,32` | `1024,256,64` |
| `--lr` | `1e-4` | `1e-4` |
| `--num_train_epochs` | `200` | `200` |
| Batch per GPU | 32 | 16 |

---

## 评估结果提交

```bash
# 上传到 FreiHAND 官方 Codalab 评估服务器
# https://competitions.codalab.org/competitions/21238
zip freihand_pred.zip pred.json
# 上传 freihand_pred.zip
```
