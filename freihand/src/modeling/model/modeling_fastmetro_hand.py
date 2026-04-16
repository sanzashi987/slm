# ----------------------------------------------------------------------------------------------
# FastMETRO - Minimal FreiHAND implementation
# Derived from: kaist-ami/FastMETRO (MIT License)
# Only the Hand network is kept. Data paths are passed in via args (no hardcoded './src/...').
# ----------------------------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import os
import torch
import numpy as np
from torch import nn

from freihand.src.modeling.mano_utils import Mesh

from .transformer import build_transformer
from .position_encoding import build_position_encoding


class FastMETRO_Hand_Network(nn.Module):
    """FastMETRO for 3D hand mesh reconstruction from a single RGB image."""

    def __init__(self, args, backbone, mesh_sampler:Mesh, num_joints=21, num_vertices=195):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices

        # Transformer layer counts according to model size
        if "FastMETRO-S" in args.model_name:
            num_enc_layers, num_dec_layers = 1, 1
        elif "FastMETRO-M" in args.model_name:
            num_enc_layers, num_dec_layers = 2, 2
        elif "FastMETRO-L" in args.model_name:
            num_enc_layers, num_dec_layers = 3, 3
        else:
            raise ValueError(f"Invalid model_name: {args.model_name}")

        # Two-stage transformer: high-dim -> low-dim (progressive dim reduction)
        self.transformer_config_1 = {
            "model_dim": args.model_dim_1,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": args.feedforward_dim_1,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }
        self.transformer_config_2 = {
            "model_dim": args.model_dim_2,
            "dropout": args.transformer_dropout,
            "nhead": args.transformer_nhead,
            "feedforward_dim": args.feedforward_dim_2,
            "num_enc_layers": num_enc_layers,
            "num_dec_layers": num_dec_layers,
            "pos_type": args.pos_type,
        }
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)

        # Dim reduction between two transformers
        d1 = self.transformer_config_1["model_dim"]
        d2 = self.transformer_config_2["model_dim"]
        self.dim_reduce_enc_cam = nn.Linear(d1, d2)
        self.dim_reduce_enc_img = nn.Linear(d1, d2)
        self.dim_reduce_dec = nn.Linear(d1, d2)

        # Learnable tokens
        self.cam_token_embed = nn.Embedding(1, d1)
        self.joint_token_embed = nn.Embedding(self.num_joints, d1)
        self.vertex_token_embed = nn.Embedding(self.num_vertices, d1)

        # Positional encodings (2D sine)
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1["pos_type"], hidden_dim=d1)
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2["pos_type"], hidden_dim=d2)

        # Output heads
        self.xyz_regressor = nn.Linear(d2, 3)
        self.cam_predictor = nn.Linear(d2, 3)

        # Reduce backbone feature channels down to transformer dim
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, d1, kernel_size=1)

        # Attention mask (from MANO mesh adjacency) - only vertex-vertex region is masked
        data_dir = getattr(args, "model_data_dir", "src/modeling/data")
        adj_indices = torch.load(os.path.join(data_dir, "mano_195_adjmat_indices.pt"))
        adj_values = torch.load(os.path.join(data_dir, "mano_195_adjmat_values.pt"))
        adj_size = torch.load(os.path.join(data_dir, "mano_195_adjmat_size.pt"))
        adjacency = torch.sparse_coo_tensor(adj_indices, adj_values, size=adj_size).to_dense()

        zeros_1 = torch.zeros((num_vertices, num_joints), dtype=torch.bool)
        zeros_2 = torch.zeros((num_joints, num_joints + num_vertices), dtype=torch.bool)
        temp_mask_1 = (adjacency == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
        self.register_buffer("attention_mask", attention_mask, persistent=False)

    def forward(self, images):
        device = images.device
        bs = images.size(0)

        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, bs, 1)       # 1 x B x d1
        jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0) \
                         .unsqueeze(1).repeat(1, bs, 1)                              # (J+V) x B x d1

        # CNN backbone features
        img_features = self.backbone(images)                                         # B x C x h x w
        _, _, h, w = img_features.shape
        img_features = self.conv_1x1(img_features).flatten(2).permute(2, 0, 1)       # (h*w) x B x d1

        # Positional encodings
        pos_enc_1 = self.position_encoding_1(bs, h, w, device).flatten(2).permute(2, 0, 1)
        pos_enc_2 = self.position_encoding_2(bs, h, w, device).flatten(2).permute(2, 0, 1)

        # First transformer (high dim)
        cam_feat_1, enc_img_1, jv_feat_1 = self.transformer_1(
            img_features, cam_token, jv_tokens, pos_enc_1, attention_mask=self.attention_mask
        )

        # Dim reduction
        r_cam = self.dim_reduce_enc_cam(cam_feat_1)
        r_img = self.dim_reduce_enc_img(enc_img_1)
        r_jv = self.dim_reduce_dec(jv_feat_1)

        # Second transformer (low dim)
        cam_feat_2, _, jv_feat_2 = self.transformer_2(
            r_img, r_cam, r_jv, pos_enc_2, attention_mask=self.attention_mask
        )

        # Predictions
        pred_cam = self.cam_predictor(cam_feat_2).view(bs, 3)                        # B x 3
        pred_3d = self.xyz_regressor(jv_feat_2.transpose(0, 1))                      # B x (J+V) x 3
        pred_3d_joints = pred_3d[:, :self.num_joints, :]
        pred_3d_vertices_coarse = pred_3d[:, self.num_joints:, :]
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_coarse)  # B x 778 x 3

        return {
            "pred_cam": pred_cam,
            "pred_3d_joints": pred_3d_joints,
            "pred_3d_vertices_coarse": pred_3d_vertices_coarse,
            "pred_3d_vertices_fine": pred_3d_vertices_fine,
        }
