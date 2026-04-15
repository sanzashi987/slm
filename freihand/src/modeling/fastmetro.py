"""
FastMETRO 手部网格重建模型
Encoder-Decoder Transformer 架构
参考: https://github.com/kaist-ami/FastMETRO (ECCV 2022)

关键思路:
  - CNN backbone 提取图像特征 (image tokens)
  - 单独的 joint token 和 vertex token 作为 query
  - Cross-attention: joint/vertex tokens 对 image tokens 做 cross-attn
  - 级联三层 decoder，逐步上采样 (195→778 vertices)
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor
from typing import List, Optional


# ─────────────────────────────────────────────────────────
#  Position Encoding (2D sinusoidal, flattened to 1D)
# ─────────────────────────────────────────────────────────

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, H, W) → returns (B, H*W, C) position embedding"""
        B, C, H, W = x.shape
        y_embed = torch.arange(H, device=x.device, dtype=torch.float32).unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=x.device, dtype=torch.float32).unsqueeze(0).expand(H, W)

        dim_t = torch.arange(self.num_pos_feats, device=x.device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=-1).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=-1).flatten(2)

        pos = torch.cat([pos_y, pos_x], dim=-1).flatten(0, 1)  # (H*W, C)
        return pos.unsqueeze(0).expand(B, -1, -1)               # (B, H*W, C)


# ─────────────────────────────────────────────────────────
#  Multi-Head Attention helpers
# ─────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Query 对 Key/Value 做 cross-attention."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm   = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, query: Tensor, key_value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        x, _ = self.attn(query, key_value, key_value,
                          key_padding_mask=key_padding_mask,
                          attn_mask=attn_mask)
        return self.norm(query + self.drop(x))


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return self.norm(x + self.drop(out))


class FFN(nn.Module):
    def __init__(self, d_model: int, dim_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.net(x))


# ─────────────────────────────────────────────────────────
#  FastMETRO Decoder Block
#  (joint tokens + vertex tokens cross-attend to image)
# ─────────────────────────────────────────────────────────

class FastMETRODecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        # joint branch
        self.j_self  = SelfAttention(d_model, nhead, dropout)
        self.j_cross = CrossAttention(d_model, nhead, dropout)
        self.j_ffn   = FFN(d_model, dim_ff, dropout)

        # vertex branch (cross-attends to joints too)
        self.v_self      = SelfAttention(d_model, nhead, dropout)
        self.v_cross_img = CrossAttention(d_model, nhead, dropout)
        self.v_cross_jnt = CrossAttention(d_model, nhead, dropout)
        self.v_ffn       = FFN(d_model, dim_ff, dropout)

    def forward(self, joint_tokens: Tensor, vertex_tokens: Tensor,
                img_tokens: Tensor) -> tuple:
        # joint tokens attend to image
        j = self.j_self(joint_tokens)
        j = self.j_cross(j, img_tokens)
        j = self.j_ffn(j)

        # vertex tokens attend to image + joints
        v = self.v_self(vertex_tokens)
        v = self.v_cross_img(v, img_tokens)
        v = self.v_cross_jnt(v, j)
        v = self.v_ffn(v)

        return j, v


# ─────────────────────────────────────────────────────────
#  FastMETRO Hand Network
# ─────────────────────────────────────────────────────────

JOINT_NUM   = 21    # MANO hand joints
VERTEX_NUM  = 778   # MANO full mesh vertices
VERTEX_SUB  = 195   # sub-sampled mesh (195 → 778 via upsampling matrix)

class FastMETRO_Hand_Network(nn.Module):
    """
    Full model:
        backbone → feature map
        → 3-stage decoder (cross-attn joint & vertex tokens ↔ image tokens)
        → head MLPs → (cam, joints_3d, vertices_sub, vertices_full)
    """
    def __init__(self,
                 backbone: nn.Module,
                 d_model_stages: List[int],   # e.g. [512, 256, 128]
                 nhead: int = 8,
                 dropout: float = 0.1,
                 img_feat_dim: int = 2048,     # backbone output channels
                 n_decoder_layers: int = 1):
        super().__init__()
        self.backbone = backbone
        self.n_stages = len(d_model_stages)

        # Per-stage: project image features into d_model space
        self.img_proj = nn.ModuleList()
        self.pos_enc  = nn.ModuleList()
        # Per-stage joint/vertex learnable queries
        self.joint_queries  = nn.ParameterList()
        self.vertex_queries = nn.ParameterList()
        # Per-stage decoders
        self.decoders   = nn.ModuleList()
        # Between-stage joint/vertex token projections
        self.jnt_proj = nn.ModuleList()
        self.vtx_proj = nn.ModuleList()

        prev_j_dim = None
        prev_v_dim = None
        prev_img   = img_feat_dim

        for i, d in enumerate(d_model_stages):
            self.img_proj.append(nn.Linear(prev_img, d))
            self.pos_enc.append(PositionEmbeddingSine(d // 2))

            self.joint_queries.append(nn.Parameter(torch.zeros(1, JOINT_NUM, d)))
            self.vertex_queries.append(nn.Parameter(torch.zeros(1, VERTEX_SUB, d)))
            nn.init.normal_(self.joint_queries[-1],  std=0.02)
            nn.init.normal_(self.vertex_queries[-1], std=0.02)

            blocks = nn.ModuleList([
                FastMETRODecoderBlock(d, nhead, d * 4, dropout)
                for _ in range(n_decoder_layers)])
            self.decoders.append(blocks)

            if i > 0:
                self.jnt_proj.append(nn.Linear(prev_j_dim, d))
                self.vtx_proj.append(nn.Linear(prev_v_dim, d))

            prev_img   = d
            prev_j_dim = d
            prev_v_dim = d

        last_d = d_model_stages[-1]

        # Output heads
        self.cam_head   = nn.Sequential(nn.Linear(last_d, 3))   # scale, tx, ty (weak perspective)
        self.joint_head = nn.Sequential(nn.Linear(last_d, 3))   # 3D joint positions
        self.vert_head  = nn.Sequential(nn.Linear(last_d, 3))   # sub-mesh vertex positions

        # Pool tokens for camera (mean of joint tokens)
        self._init_weights()

    def _init_weights(self):
        for m in [self.cam_head, self.joint_head, self.vert_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _extract_image_tokens(self, x: Tensor, stage_idx: int) -> tuple:
        """Flatten backbone feature map and project."""
        B, C, H, W = x.shape
        pos = self.pos_enc[stage_idx](x)                    # (B, H*W, d)
        img = x.flatten(2).permute(0, 2, 1)                # (B, H*W, C)
        img = self.img_proj[stage_idx](img) + pos           # (B, H*W, d)
        return img

    def forward(self, images: Tensor, mano_model, mesh_sampler,
                meta_masks=None, is_train: bool = True):
        B = images.shape[0]

        # ── backbone (feature pyramid / single scale) ────────
        feat = self.backbone(images)  # (B, C, H, W)

        prev_j = None
        prev_v = None
        img_tokens = feat   # will be projected in each stage

        for i in range(self.n_stages):
            d = self.joint_queries[i].shape[-1]

            img_tok = self._extract_image_tokens(img_tokens, i)  # (B, HW, d)

            # Init joint/vertex queries (learned embeddings)
            jq = self.joint_queries[i].expand(B, -1, -1).clone()
            vq = self.vertex_queries[i].expand(B, -1, -1).clone()

            # Add residual from previous stage (if available)
            if i > 0:
                jq = jq + self.jnt_proj[i - 1](prev_j)
                vq = vq + self.vtx_proj[i - 1](prev_v)

            for block in self.decoders[i]:
                jq, vq = block(jq, vq, img_tok)

            prev_j = jq
            prev_v = vq

            # Feed projected features into next stage
            img_tokens = img_tok.mean(dim=1, keepdim=True).expand(-1, img_tok.shape[1], -1)
            img_tokens = img_tokens.permute(0, 2, 1).view(B, d, 1, img_tok.shape[1])

        # ── heads ────────────────────────────────────────────
        pred_3d_joints  = self.joint_head(jq)          # (B, 21, 3)
        pred_vertices_sub = self.vert_head(vq)          # (B, 195, 3)

        # Camera: mean-pool joint tokens
        pred_cam = self.cam_head(jq.mean(dim=1))        # (B, 3)

        # Upsample sub-mesh → full mesh
        pred_vertices = mesh_sampler.upsample(pred_vertices_sub)  # (B, 778, 3)

        return pred_cam, pred_3d_joints, pred_vertices_sub, pred_vertices


# ─────────────────────────────────────────────────────────
#  Backbone builders
# ─────────────────────────────────────────────────────────

def build_backbone_resnet50(pretrained: bool = True) -> tuple:
    """Returns (backbone_module, output_channels).
    Set pretrained=False to skip downloading weights (useful for testing)."""
    if pretrained:
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
    r50 = tv_models.resnet50(weights=weights)
    # Remove avg-pool and fc; keep up to layer4
    backbone = nn.Sequential(
        r50.conv1, r50.bn1, r50.relu, r50.maxpool,
        r50.layer1, r50.layer2, r50.layer3, r50.layer4)
    out_channels = 2048
    return backbone, out_channels


def build_backbone_hrnet_w64(hrnet_ckpt: Optional[str] = None) -> tuple:
    """Lightweight stub — replace with actual HRNet if needed."""
    try:
        from src.modeling.hrnet import build_hrnet_w64
        backbone, out_channels = build_hrnet_w64(hrnet_ckpt)
    except ImportError:
        print("[WARN] HRNet not found; falling back to ResNet-50.")
        backbone, out_channels = build_backbone_resnet50()
    return backbone, out_channels


# ─────────────────────────────────────────────────────────
#  Public build function
# ─────────────────────────────────────────────────────────

def build_fastmetro(args, input_feat_dim: List[int], hidden_feat_dim: List[int]) -> FastMETRO_Hand_Network:
    """
    args.arch:       'resnet50' | 'hrnet-w64'
    args.model_name: 'FastMETRO-S' | 'FastMETRO-L'
    input_feat_dim:  e.g. [2051, 512, 128]  (first value must be img feature dim + 3)
    hidden_feat_dim: e.g. [1024, 256, 64]   (maps to nhead; FastMETRO uses 8 heads)
    """
    if args.arch == "hrnet-w64":
        backbone, img_feat_dim = build_backbone_hrnet_w64(
            getattr(args, "hrnet_checkpoint", None))
    else:
        pretrained = getattr(args, "pretrained_backbone", True)
        backbone, img_feat_dim = build_backbone_resnet50(pretrained=pretrained)

    # d_model per stage = hidden_feat_dim values
    d_model_stages = hidden_feat_dim  # [1024, 256, 64] for FastMETRO-L

    # nhead: FastMETRO-S uses 4, FastMETRO-L uses 8
    nhead = 8 if "L" in args.model_name else 4

    model = FastMETRO_Hand_Network(
        backbone     = backbone,
        d_model_stages = d_model_stages,
        nhead        = nhead,
        dropout      = args.drop_out,
        img_feat_dim = img_feat_dim,
    )
    return model
