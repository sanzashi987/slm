---
title: FastMETRO 实践篇：Cross-Attention 与完整 Encoder-Decoder 架构在手部网格重建中的应用
description: 基于 FastMETRO (Cross-attention of Disentangled Modalities for 3D Human Mesh Recovery) 论文在 FreiHAND 数据集上的最小化实现。承接前两篇文档建立的 Transformer 基础，完整展示 Encoder-Decoder 架构、Cross-Attention 设计、2D 正弦位置编码、渐进式降维、MANO 参数化手部模型，以及面向手部网格重建的多维损失函数。
---

> **说明**: 本文是前两篇文档（[自动微分与 MLP](autograd-mlp-implementation.md)、[GPT 实现](gpt-implementation.md)）的**实践篇**。前两篇已详细推导的概念（Self-Attention、残差连接、Layer Norm、Multi-Head、位置编码原理）在此仅做简短引用。本文聚焦于在前置知识基础上新引入的设计：Encoder 的必要性、Cross-Attention 机制、2D 位置编码、可学习的 query token，以及手部网格重建特有的工程细节。

---

## 1: 任务定义与整体架构

### 1.1 问题：从单张 RGB 图像恢复 3D 手部网格

给定一张 224×224 的手部 RGB 图像，目标是预测：

- **21 个关节点**的 3D 坐标（腕部 + 5 指各 4 个关节）
- **778 个顶点**的 3D 坐标，构成完整的手部网格

这是一个从 2D 观测重建 3D 结构的**高度欠定问题**（ill-posed problem）——同一张 2D 图像可以对应无数种 3D 姿态。解决它需要模型同时理解：
- **局部纹理**（皮肤褶皱、关节轮廓）
- **全局几何约束**（手指长度比例、关节角度范围）
- **跨模态关联**（关节点与网格顶点之间的解剖约束）

### 1.2 为什么选 ResNet-50 而不是 ViT

整体架构是 **CNN 骨干网络 + Transformer**，而不是纯 ViT：

```python
def build_resnet50_backbone():
    """strip avgpool + fc, 输出 B x 2048 x 7 x 7（224 输入）"""
    m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    return nn.Sequential(*list(m.children())[:-2])
```

ResNet-50 的选型优势在三个维度：

**归纳偏置（Inductive Bias）**：卷积天然具有局部性（local connectivity）和平移等变性（translation equivariance）——这正是图像低层特征（边缘、纹理）的先验结构。ViT 需要大量数据才能从零学习这些偏置，ResNet 则开箱即用。

**迁移学习**：ImageNet 预训练的 ResNet-50 已经学会了通用视觉特征。去掉最后的全局平均池化（avgpool）和分类头（fc），保留的卷积特征图 $\mathbb{R}^{B \times 2048 \times 7 \times 7}$ 保存了空间位置信息，这是 Transformer 后续做空间注意力的基础。若用全局 avgpool 输出，空间信息就丢失了。

**计算效率**：ResNet-50 将 $224 \times 224$ 的高分辨率输入压缩到 $7 \times 7$ 的特征图（步长 32），Transformer 只需处理 $7 \times 7 = 49$ 个 patch，而不是 $224^2$ 个像素。

### 1.3 整体数据流

```
RGB Image (B, 3, 224, 224)
    ↓ ResNet-50 (去掉 avgpool/fc)
Feature Map (B, 2048, 7, 7)
    ↓ 1×1 Conv: 2048 → d1=512
img_features (49, B, 512)    ← 序列形式，供 Transformer 处理

cam_token    (1,  B, 512)    ← 可学习：相机参数
jv_tokens    (216, B, 512)   ← 可学习：21 关节 + 195 顶点

              ┌──────────────────────┐
              │  Transformer 1 (d1)  │
              │  Encoder + Decoder   │
              └──────────────────────┘
                         ↓ 线性降维 d1→d2
              ┌──────────────────────┐
              │  Transformer 2 (d2)  │
              │  Encoder + Decoder   │
              └──────────────────────┘
                         ↓
    cam_feat → cam_predictor → pred_cam (B, 3)
    jv_feat  → xyz_regressor → pred_3d  (B, 216, 3)
                         ↓ 网格上采样 195→778
    pred_3d_vertices_fine (B, 778, 3)
```

---

## 2: MANO 参数化手部模型

### 2.1 为什么需要参数化模型

直接回归 778 个顶点的 3D 坐标（$778 \times 3 = 2334$ 个自由度）在训练初期非常困难，因为没有任何约束：预测的网格可能自相交、关节角度不合理。

**MANO**（Hand Model with Articulations and Non-rigid Deformations）通过低维参数控制手部形状，将自由度压缩到：
- **姿态参数** $\theta \in \mathbb{R}^{48}$：手腕全局旋转（3 维）+ 15 个关节的局部旋转（各 3 维）
- **形状参数** $\beta \in \mathbb{R}^{10}$：PCA 空间中控制手的大小、胖瘦等形状变化

MANO 的生成过程（Linear Blend Skinning）：

$$\mathbf{V}(\theta, \beta) = W\!\left(T(\theta, \beta),\ J(\beta),\ \theta,\ \mathcal{W}\right)$$

其中 $T(\theta, \beta)$ 是休息姿态网格，$J(\beta)$ 是关节中心位置，$\mathcal{W}$ 是蒙皮权重矩阵。这个公式将低维参数映射到完整的 778 顶点网格。

在训练中，MANO 被用于**生成 GT 网格**（不是预测目标），其参数由数据集提供：

```python
with torch.no_grad():
    gt_vertices_fine, gt_joints = mano_model.layer(pose, betas)  # (B,778,3), (B,21,3)
    gt_vertices_fine /= 1000.0   # mm → m
    gt_joints /= 1000.0
```

`torch.no_grad()` 是因为 MANO 的 LBS 前向传播对精度敏感，而且 GT 生成不需要梯度。

### 2.2 关节点回归器（Joint Regressor）

MANO 直接输出 778 个顶点，关节点坐标由线性回归矩阵从顶点推导：

$$\mathbf{J} = \mathcal{R} \cdot \mathbf{V}, \quad \mathcal{R} \in \mathbb{R}^{21 \times 778}$$

```python
def get_3d_joints(self, vertices):
    """vertices: B x 778 x 3  →  B x 21 x 3"""
    return torch.einsum("bik,ji->bjk", [vertices, self.joint_regressor_torch])
```

`einsum("bik,ji->bjk")` 的含义：对每个 batch 样本，计算 $J_{bj3} = \sum_i \mathcal{R}_{ji} \cdot V_{bi3}$，即对顶点加权求和得到关节点坐标。

指尖（拇指、食指、中指、无名指、小指各一个）在原始 MANO 关节回归器中没有对应点，通过直接查找特定顶点（如拇指指尖 = 顶点 745）来补充：

```python
fingertip_idx = {745: "thumb", 317: "index", 445: "middle", 556: "ring", 673: "pinky"}
for v_idx in fingertip_idx.keys():
    oh = np.zeros((1, 778))
    oh[0, v_idx] = 1.0          # one-hot：指定顶点权重为 1
    onehots.append(oh)
```

### 2.3 网格下采样：778 → 195

FastMETRO 的 Decoder 不直接预测 778 个顶点——这会引入巨大的计算开销和优化难度。取而代之的是先预测粗糙的 195 个顶点，再上采样到 778。

下采样矩阵 $D \in \mathbb{R}^{195 \times 778}$ 和上采样矩阵 $U \in \mathbb{R}^{778 \times 195}$ 由图谱分析预先计算并存储为稀疏矩阵：

```python
class Mesh:
    def downsample(self, x):  # B x 778 x 3 → B x 195 x 3
        return self._batched_spmm(self._D[0], x)

    def upsample(self, x):    # B x 195 x 3 → B x 778 x 3
        return self._batched_spmm(self._U[0], x)

    @staticmethod
    def _batched_spmm(sparse, x):
        """单次稀疏矩阵乘法替代逐样本 Python 循环"""
        B, N, C = x.shape
        x_flat = x.permute(1, 0, 2).reshape(N, B * C)   # (N, B*C)
        y_flat = _spmm(sparse, x_flat)                   # (M, B*C)
        M = y_flat.shape[0]
        return y_flat.reshape(M, B, C).permute(1, 0, 2).contiguous()
```

将批次展开为 `(N, B*C)` 后做一次稀疏矩阵乘法，比在 Python 层对每个样本循环快数倍。

---

## 3: 2D 正弦位置编码

### 3.1 为什么 1D 位置编码不够

[GPT 实现篇](gpt-implementation.md)的位置编码是 1D 的：每个 token 在序列中有一个位置索引 $0, 1, 2, \ldots, T-1$，对应一个可学习的向量。

图像特征不同。ResNet 输出的特征图是 $7 \times 7$ 的二维网格，每个位置同时具有**行坐标**和**列坐标**。如果把 $7 \times 7 = 49$ 个特征点展平为 1D 序列，只用 1D 编码，位置 $(i, j)$ 和 $(i', j')$ 的关系（如"同一行"、"相邻列"）就被抹平了。

2D 位置编码的目标：让 $(i_1, j_1)$ 和 $(i_2, j_2)$ 的编码之间的**距离**反映二维空间的邻近程度。

### 3.2 2D 正弦编码的推导

沿用 "Attention is All You Need" 的正弦公式，但分别对 $x$ 坐标和 $y$ 坐标编码，各占 $d/2$ 维，最后拼接：

$$\text{PE}_{(i,j), 2k}^{(y)} = \sin\!\left(\frac{i}{T^{2k/d_{\text{half}}}}\right), \quad \text{PE}_{(i,j), 2k+1}^{(y)} = \cos\!\left(\frac{i}{T^{2k/d_{\text{half}}}}\right)$$

$$\text{PE}_{(i,j), 2k}^{(x)} = \sin\!\left(\frac{j}{T^{2k/d_{\text{half}}}}\right), \quad \text{PE}_{(i,j), 2k+1}^{(x)} = \cos\!\left(\frac{j}{T^{2k/d_{\text{half}}}}\right)$$

$$\text{PE}_{(i,j)} = \left[\text{PE}^{(y)}_{(i,j)},\ \text{PE}^{(x)}_{(i,j)}\right] \in \mathbb{R}^{d}$$

其中 $T = 10000$，$d_{\text{half}} = d/2$。每个维度用不同频率的正弦波编码，低维度频率高（捕捉细粒度差异），高维度频率低（捕捉粗粒度差异）。

**实现细节**：

```python
class PositionEmbeddingSine(nn.Module):
    def forward(self, bs, h, w, device):
        ones = torch.ones((bs, h, w), dtype=torch.bool, device=device)
        # cumsum 沿行/列累积，得到 1..h 和 1..w 的坐标网格
        y_embed = ones.cumsum(1, dtype=torch.float32)  # (B, h, w): row indices
        x_embed = ones.cumsum(2, dtype=torch.float32)  # (B, h, w): col indices

        # 归一化到 [0, 2π]，使坐标范围与频率尺度对齐
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # scale = 2π
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 频率分母：T^(2k/num_pos_feats)，k = 0..num_pos_feats-1
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, h, w, num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t

        # 奇偶交错：sin(偶数维), cos(奇数维)，再展平
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        # y 和 x 编码各占一半维度，拼接后调整为 (B, d, h, w)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos  # (B, d1, h, w)
```

`dim_t // 2` 而非 `dim_t`，使得相邻两个维度共享相同的频率（一个用 sin，一个用 cos），这是标准实现中奇偶对称的来源。

### 3.3 位置编码如何注入

位置编码**只加在 Q 和 K 上，不加在 V 上**：

```python
# 编码器层
def with_pos_embed(self, tensor, pos):
    # tensor[0] 是 cam_token，不加位置编码
    return tensor if pos is None else torch.cat([tensor[:1], (tensor[1:] + pos)], dim=0)

def forward(self, src, ..., pos=None):
    src2 = self.norm1(src)
    q = k = self.with_pos_embed(src2, pos)          # Q, K 含位置信息
    src2 = self.self_attn(q, k, value=src2, ...)[0] # V 不含位置信息
```

> **为什么 V 不加位置编码？**
> Q 和 K 的点积决定"两个位置有多相关"——这个相关性计算需要知道双方的位置。而 V 是"相关后要聚合的内容"，聚合的是像素特征本身（颜色、纹理），不需要位置标签。将位置信息混入 V 反而会干扰特征的语义含义。这是 DETR 论文（本实现参考的上游代码）实验验证的设计。

注意 `cam_token`（相机参数 token）**不加位置编码**：相机参数是全局属性，没有空间位置的概念，强行加入 2D 位置编码反而会引入错误的归纳偏置。

---

## 4: Encoder——为什么 GPT 不需要但这里需要

### 4.1 GPT（Decoder-only）的局限性

[GPT 实现篇](gpt-implementation.md)实现的是纯 Decoder 架构：输入 token 序列经过多个 Self-Attention Block，输出下一个 token 的概率分布。

Decoder 有一个关键设计：**因果掩码**（causal mask）。每个位置只能看到自己和之前的位置，不能看到未来。这在语言生成中是必须的——生成第 $t$ 个词时，第 $t+1$ 个词还没有被生成。

但在图像理解任务中，没有"时间顺序"的约束。图像的 $7 \times 7$ 特征图中，每个 patch 都应该能关注到所有其他 patch——位置 $(3, 4)$ 的特征理应参考 $(0, 0)$ 和 $(6, 6)$ 的信息。这就是 Encoder 的作用。

### 4.2 Encoder 层：双向全局 Self-Attention

```python
class TransformerEncoderLayer(nn.Module):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        # 无因果掩码：每个位置可以关注所有其他位置
        src2 = self.self_attn(q, k, value=src2,
                              attn_mask=src_mask,                    # 通常为 None
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)   # 残差连接
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)   # 残差连接
        return src
```

与 GPT Decoder 层（见 [GPT 实现篇](gpt-implementation.md) 第 4 章）相比，Encoder 层去掉了因果掩码，允许双向注意力。其他组件——Multi-Head Self-Attention、FeedForward、残差连接、Pre-norm Layer Norm——完全相同。

| | GPT Decoder | FastMETRO Encoder |
|---|---|---|
| 注意力方向 | 单向（只看历史） | 双向（全局） |
| 掩码 | 因果掩码 | 无（或 padding 掩码） |
| 输入 | token 序列（离散） | 图像 patch 特征（连续） |
| 位置编码 | 可学习 1D | 正弦 2D（加在 Q/K 上） |
| 归一化 | Pre-norm | Pre-norm |

### 4.3 cam_token 与图像特征的联合编码

Encoder 的输入不只有图像特征，还包含一个特殊的**相机 token**：

```python
cam_with_img = torch.cat([cam_token, img_features], dim=0)
# (1 + 49) x B x d1：cam_token 在序列开头

e_outputs = self.encoder(cam_with_img, src_key_padding_mask=mem_mask, pos=pos_enc_1)
cam_features, enc_img_features = e_outputs.split([1, hw], dim=0)
```

将 `cam_token` 与图像特征序列拼接后一起送入 Encoder，让 cam_token 通过 Self-Attention 聚合全局图像信息（整个手掌的姿态、透视关系），编码出相机参数的先验。

`src_key_padding_mask` 的 `mem_mask` 让 cam_token 对应的位置掩码为 False（不屏蔽），图像特征位置也为 False，因此全部参与注意力计算（在没有 padding 的情况下整个掩码是全零）。

---

## 5: Cross-Attention 解码器——核心设计

### 5.1 核心问题：如何从图像特征提取结构化的几何信息

经过 Encoder，我们得到了富含上下文的图像特征 `enc_img_features`（形状 $49 \times B \times d$）。现在需要从中"提取"216 个特定位置（21 关节 + 195 顶点）的 3D 坐标。

朴素方法：把 $49 \times d$ 维图像特征展平成一个向量，然后用 MLP 回归出 $216 \times 3$ 个坐标。这样做的问题是：

1. **丢失空间结构**：展平操作把图像的二维拓扑结构完全丢弃
2. **无法表达关节约束**：关节点之间的解剖学约束（如食指三节相连）无法自然地编码
3. **参数量爆炸**：$49 \times 512 \to 216 \times 3$ 需要大量 MLP 参数

Cross-Attention 的解法：为每个关节点和顶点设计一个**可学习的 query token**，让它主动"向图像特征提问"，按需聚合所需的视觉信息。

### 5.2 可学习的 Query Token

```python
# 关节点和顶点的 query tokens
self.joint_token_embed  = nn.Embedding(num_joints=21,    d1)
self.vertex_token_embed = nn.Embedding(num_vertices=195, d1)

# 前向传播
jv_tokens = torch.cat([self.joint_token_embed.weight,
                        self.vertex_token_embed.weight], dim=0)  # (216, d1)
jv_tokens = jv_tokens.unsqueeze(1).repeat(1, bs, 1)             # (216, B, d1)
```

每个关节点和顶点都有一个独立的可学习向量。训练过程中，这些向量会逐渐编码出"某个解剖位置需要从图像中查询什么视觉线索"。

例如，拇指指尖的 token 可能学到：查询图像中圆形高亮区域（指甲反光）附近的特征；腕部 token 可能学到：查询手腕轮廓和皮肤皱褶。

### 5.3 Decoder 层：三步结构

```python
class TransformerDecoderLayer(nn.Module):
    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # ① Self-Attention on queries
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)  # query_pos = 全零
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ② Cross-Attention：queries 向图像特征"提问"
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),   # Q：关节/顶点 token
            key=self.with_pos_embed(memory, pos),          # K：图像特征 + 2D 位置编码
            value=memory,                                  # V：图像特征本身
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        # ③ FeedForward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
```

**步骤 ①：Self-Attention（关节/顶点之间的相互感知）**

216 个 query token 彼此做 Self-Attention。这一步让关节点和顶点之间能够交换信息——"我是中指第二关节，旁边的关节在哪"。

同时使用了**MANO 网格邻接掩码**（见第 6 章）：在顶点间的 Self-Attention 中，只允许解剖学上相邻的顶点互相注意，避免遥远顶点的噪声干扰。

**步骤 ②：Cross-Attention（关节/顶点向图像提问）**

这是整个设计的核心，数学形式：

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q_{\text{jv}} \cdot K_{\text{img}}^T}{\sqrt{d_k}}\right) V_{\text{img}}$$

其中：
- $Q_{\text{jv}} \in \mathbb{R}^{216 \times d}$：关节/顶点 query token（经由 Decoder Self-Attention 更新后）
- $K_{\text{img}} \in \mathbb{R}^{49 \times d}$：图像特征（含 2D 位置编码）
- $V_{\text{img}} \in \mathbb{R}^{49 \times d}$：图像特征（不含位置编码）

注意力矩阵 $A \in \mathbb{R}^{216 \times 49}$ 的每一行代表"某个关节/顶点对图像 49 个 patch 的关注分布"。对于拇指指尖，它的行向量应该在图像右上角（或左上角，取决于视角）的 patch 上有高权重。

Cross-Attention 与 Self-Attention 的根本区别是 **Q 和 K/V 来自不同的模态**：Q 来自几何 query token，K/V 来自视觉特征。这就是论文名称中"Disentangled Modalities（解耦模态）"的含义——几何结构和视觉特征在各自的空间里独立表达，通过 Cross-Attention 桥接。

### 5.4 Decoder 与 GPT Decoder 的对比

GPT Decoder（[GPT 实现篇](gpt-implementation.md)第 4 章）没有 Cross-Attention：所有信息来自同一个 token 序列内的 Self-Attention。这是因为 GPT 的信息来源只有一个——前缀 token 序列。

FastMETRO Decoder 有两个信息来源：

| 来源 | 通过哪种 Attention 接收 |
|------|------------------------|
| 其他关节/顶点的状态 | Self-Attention（步骤 ①） |
| CNN 图像特征 | Cross-Attention（步骤 ②） |

这正是 Encoder-Decoder 架构与 Decoder-only 架构的核心区别。

---

## 6: MANO 网格邻接掩码

### 6.1 问题：顶点间 Self-Attention 的噪声

195 个粗糙顶点分布在整个手部网格上。如果允许所有顶点之间做全连接 Self-Attention，位于手背的顶点会注意到手掌的顶点——在 3D 空间中它们可能很近，但在 MANO 网格拓扑上没有直接边连接。这种"跨过手指"的注意力引入了噪声。

### 6.2 用网格邻接矩阵构造掩码

将 MANO 网格的稀疏邻接矩阵加载为注意力掩码，**只允许解剖学上相邻的顶点互相注意**：

```python
adjacency = torch.sparse_coo_tensor(adj_indices, adj_values, size=adj_size).to_dense()
# adjacency[i, j] = 1 表示顶点 i 和 j 在网格中相邻

# 构造 (J+V) x (J+V) 的完整掩码
# 规则：
#   关节-关节：全部允许（zeros_2）
#   关节-顶点：关节可以注意所有顶点（zeros_1）
#   顶点-顶点：只允许相邻顶点（adjacency == 0 表示非邻居，应屏蔽）
zeros_1 = torch.zeros((num_vertices, num_joints), dtype=torch.bool)   # V x J
zeros_2 = torch.zeros((num_joints, num_joints + num_vertices), dtype=torch.bool)  # J x (J+V)
temp_mask_1 = adjacency == 0                          # V x V：非邻居处为 True（屏蔽）
temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)  # V x (J+V)
attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)  # (J+V) x (J+V)
```

掩码中 `True` 表示"屏蔽此位置"，即在 softmax 之前将对应的注意力分数设为 $-\infty$。

这个设计的物理意义：Decoder Self-Attention 尊重手部网格的拓扑结构，强迫每个顶点的表示主要由解剖学近邻决定，而不是全局平均。关节点因为数量少且语义明确，不施加限制（全部 False）。

---

## 7: 两阶段渐进式降维

### 7.1 为什么要降维

FastMETRO 使用两个串联的 Transformer，分别在维度 $d_1 = 512$ 和 $d_2 = 128$ 下工作：

```python
self.transformer_1 = build_transformer({"model_dim": 512, ...})
self.transformer_2 = build_transformer({"model_dim": 128, ...})

# 三条降维路径
self.dim_reduce_enc_cam = nn.Linear(512, 128)
self.dim_reduce_enc_img = nn.Linear(512, 128)
self.dim_reduce_dec     = nn.Linear(512, 128)
```

**降维的动机**：

1. **计算效率**：注意力的计算量是 $O(n^2 d)$，降低 $d$ 直接降低 FeedForward 和线性投影的参数量与计算量。$d$ 从 512 降到 128，FeedForward 从 $2048 \times 512 \times 2 = 2M$ 个参数降到 $512 \times 128 \times 2 = 131K$。

2. **信息瓶颈（Information Bottleneck）**：第一阶段（高维）用大容量捕捉图像特征的丰富细节；降维迫使网络只保留与几何预测最相关的信息，起到正则化作用，类似于 AutoEncoder 的瓶颈层。

3. **渐进式精化**：第二阶段（低维）在压缩后的语义空间里做精细调整，此时特征空间更小、更结构化，模型更容易学到几何预测的规律。

### 7.2 两阶段前向传播

```python
# 第一阶段：d1=512，图像特征 + 全量 token
cam_feat_1, enc_img_1, jv_feat_1 = self.transformer_1(
    img_features, cam_token, jv_tokens, pos_enc_1, attention_mask=self.attention_mask
)

# 线性降维：512 → 128
r_cam = self.dim_reduce_enc_cam(cam_feat_1)   # (1, B, 128)
r_img = self.dim_reduce_enc_img(enc_img_1)    # (49, B, 128)
r_jv  = self.dim_reduce_dec(jv_feat_1)        # (216, B, 128)

# 第二阶段：d2=128，使用降维后的特征
cam_feat_2, _, jv_feat_2 = self.transformer_2(
    r_img, r_cam, r_jv, pos_enc_2, attention_mask=self.attention_mask
)
```

第二个 Transformer 的输入和第一个完全对称：`r_img` 作为 Encoder 的图像特征，`r_cam` 作为相机 token，`r_jv` 作为关节/顶点 token 送入 Decoder。两个阶段共享相同的 Transformer 结构，只是维度不同。

---

## 8: 损失函数设计

### 8.1 多维损失的必要性

单独的顶点坐标 L1 损失不能充分约束手部网格的**局部几何质量**——两个顶点坐标都正确，但它们之间的边长或三角面的法向量可能不对（网格扭曲）。FastMETRO 使用 5 种损失共同约束：

$$\mathcal{L} = \lambda_{\text{j3d}} \mathcal{L}_{\text{j3d}} + \lambda_{\text{v3d}} \mathcal{L}_{\text{v3d}} + \lambda_{\text{en}} (\lambda_e \mathcal{L}_{\text{edge}} + \lambda_n \mathcal{L}_{\text{normal}}) + \lambda_{\text{j2d}} \mathcal{L}_{\text{j2d}}$$

### 8.2 3D 关节点损失（L1）

关节点先根对齐（wrist-relative），消除全局平移的影响：

```python
def keypoint_3d_loss(pred_3d, gt_3d):
    pred = pred_3d - pred_3d[:, WRIST_IDX:WRIST_IDX+1, :]   # 以腕部为原点
    gt   = gt_3d   - gt_3d  [:, WRIST_IDX:WRIST_IDX+1, :]
    return F.l1_loss(pred, gt)
```

L1 损失比 MSE 对离群点（如严重遮挡的手指）更鲁棒——L1 梯度恒为 ±1，不会因大误差产生爆炸性梯度。

关节点损失来自两个来源：Decoder 直接预测的关节 token（`pred_joints_tok`）和从预测的精细网格通过 Joint Regressor 回归的关节（`pred_joints_mano`），两者都与 GT 计算损失。这种**双重监督**（dual supervision）让关节预测和网格拓扑互相约束。

### 8.3 网格顶点损失（L1）

分粗糙（195）和精细（778）两个分辨率：

```python
l_v3d = (args.w_v_coarse * vertices_loss(pred_v_coarse, gt_v_coarse)
       + args.w_v_fine   * vertices_loss(pred_v_fine,   gt_v_fine))
```

粗糙顶点损失直接监督 Decoder 的 195 维输出；精细顶点损失监督上采样后的 778 维网格，让上采样矩阵 $U$ 的插值误差也被约束。

### 8.4 边长损失（Edge Length Loss）

两个相邻顶点之间的边长之差：

```python
class EdgeLengthGTLoss(nn.Module):
    def forward(self, pred_v, gt_v):
        def _edges(v):
            d1 = (v[:, f[:, 0]] - v[:, f[:, 1]]).pow(2).sum(-1).clamp(min=1e-8).sqrt()
            d2 = (v[:, f[:, 0]] - v[:, f[:, 2]]).pow(2).sum(-1).clamp(min=1e-8).sqrt()
            d3 = (v[:, f[:, 1]] - v[:, f[:, 2]]).pow(2).sum(-1).clamp(min=1e-8).sqrt()
            return d1, d2, d3
        po = _edges(pred_v); go = _edges(gt_v)
        return sum((po[i] - go[i]).abs().mean() for i in range(3)) / 3.0
```

对 MANO 的所有三角面，分别计算三条边的长度差的 L1 均值。边长损失强制网格的**局部度量结构**（metric structure）正确——即使顶点坐标有偏差，相邻顶点之间的相对形状应该和 GT 一致。

`.clamp(min=1e-8)` 防止距离为零时 `sqrt` 的梯度为无穷大。

### 8.5 法向量损失（Normal Vector Loss）

三角面的法向量与 GT 一致，强制面的**朝向**正确：

$$\mathcal{L}_{\text{normal}} = \frac{1}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \frac{1}{3}\left(|v_1^T n| + |v_2^T n| + |v_3^T n|\right)$$

其中 $n = \hat{v}_1 \times \hat{v}_2$ 是 GT 面的单位法向量，$v_1, v_2, v_3$ 是预测面的三条边的单位向量。理论上面内的向量与法向量正交，点积应为 0。

```python
def forward(self, pred_v, gt_v):
    f = self.face
    # 计算预测面的三条边方向向量
    v1p = F.normalize(pred_v[:, f[:, 1]] - pred_v[:, f[:, 0]], dim=2)
    v2p = F.normalize(pred_v[:, f[:, 2]] - pred_v[:, f[:, 0]], dim=2)
    # GT 法向量
    v1g = F.normalize(gt_v[:, f[:, 1]] - gt_v[:, f[:, 0]], dim=2)
    v2g = F.normalize(gt_v[:, f[:, 2]] - gt_v[:, f[:, 0]], dim=2)
    normal_gt = F.normalize(torch.cross(v1g, v2g, dim=2), dim=2)
    # 预测面的边向量与 GT 法向量的点积绝对值（越接近 0 越好）
    c1 = (v1p * normal_gt).sum(-1).abs()
    c2 = (v2p * normal_gt).sum(-1).abs()
    c3 = (v3p * normal_gt).sum(-1).abs()
    return (c1.mean() + c2.mean() + c3.mean()) / 3.0
```

### 8.6 2D 关节点损失（正交投影）

3D 预测完成后，用相机参数将关节点正交投影到 2D，与图像中标注的 2D 关节点计算 L1 损失：

```python
def orthographic_projection(X, camera):
    """X: B x N x 3;  camera: B x 3 (scale, tx, ty)"""
    scale = camera[:, 0:1].unsqueeze(-1)    # (B, 1, 1)
    trans = camera[:, 1:].unsqueeze(1)      # (B, 1, 2)
    return scale * X[:, :, :2] + trans      # (B, N, 2)
```

正交投影（与透视投影相比）忽略深度变化对 2D 坐标的影响，仅做缩放和平移：$\hat{u} = s \cdot x + t_x$，$\hat{v} = s \cdot y + t_y$。对于手部这种小视场角物体，正交近似足够准确，且简化了相机参数学习。

2D 损失有置信度加权：

```python
def keypoint_2d_loss(pred_2d, gt_2d_with_conf):
    conf = gt_2d_with_conf[..., -1:]   # 第三维是置信度（0 或 1）
    gt   = gt_2d_with_conf[..., :-1]
    return (conf * F.l1_loss(pred_2d, gt, reduction="none")).mean()
```

被遮挡的关节点置信度为 0，不参与梯度计算。

---

## 9: 训练工程细节

### 9.1 权重初始化：Xavier Uniform

```python
def _reset_parameters(self):
    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
```

Xavier 均匀初始化（$U[-\sqrt{6/(fan\_in + fan\_out)},\ \sqrt{6/(fan\_in + fan\_out)}]$）与 [GPT 实现篇](gpt-implementation.md)的正态分布初始化（`std=0.02`）不同。两者都是为了控制激活值的方差，但方向不同：GPT 针对残差连接累积做了额外的 $1/\sqrt{2N}$ 缩放；DETR/FastMETRO 使用 Xavier，更适合 ReLU 激活的 FeedForward 层。

### 9.2 channels_last 内存格式

```python
model = model.to(memory_format=torch.channels_last)
images = images.to(memory_format=torch.channels_last)
```

PyTorch 默认的 NCHW 格式（batch, channel, height, width）中，同一个空间位置的不同 channel 在内存中不连续。`channels_last`（NHWC）将同一位置的所有 channel 排列在一起，对 ResNet 的卷积运算更友好——GPU 的 tensor core 处理 NHWC 格式时通常有 10-30% 的速度提升。

### 9.3 多维损失的权重设计

```python
# Loss weights
w_j3d = 1000.0   # 3D 关节点
w_v3d = 100.0    # 3D 顶点
w_edge_normal = 100.0  # 边长 + 法向量的组合系数
w_j2d = 100.0    # 2D 关节点
```

3D 关节点损失权重最高（1000），因为关节点是最直接的几何约束，也是评测指标（MPJPE）的核心。几何正则损失（边长、法向量）权重次之，防止网格质量问题。这些权重需要根据各项损失的数值量级调整，确保梯度信号的方向平衡。

---

## 总结

本文从 FreiHAND 手部网格重建任务出发，完整展示了 FastMETRO 的 Encoder-Decoder 架构：

- **ResNet-50 骨干**：卷积归纳偏置 + ImageNet 预训练 + 保留空间结构，将 224×224 图像压缩为 7×7 特征图，适配 Transformer
- **2D 正弦位置编码**：正弦公式从 1D 扩展到 2D，x/y 坐标各占一半维度，只加在 Q/K 上，保持 V 的语义纯净性
- **Encoder**：无因果掩码的双向 Self-Attention，让所有图像 patch 全局感知，cam_token 联合编码获取全局相机先验
- **Cross-Attention Decoder**：可学习的关节/顶点 query token 主动向图像特征"提问"，Query 来自几何域，Key/Value 来自视觉域，实现跨模态信息融合；MANO 邻接掩码约束顶点间 Self-Attention 遵循解剖拓扑
- **两阶段降维**：512→128 的信息瓶颈在保留关键语义的同时大幅减少计算量，两阶段渐进式精化提升预测质量
- **多维损失**：3D 关节点/顶点（全局坐标）+ 边长/法向量（局部几何）+ 2D 投影（跨维度一致性），五路损失从不同角度共同约束手部网格的重建质量
