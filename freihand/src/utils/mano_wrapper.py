"""
MANO 模型封装
依赖 manopth (FastMETRO 仓库中已作为 git submodule 包含):
  git submodule update --init --recursive
  pip install -e manopth/

MANO 权重需要从官网下载 (需注册):
  https://mano.is.tue.mpg.de/
  下载 mano_v1_2.zip, 解压后将 MANO_RIGHT.pkl 放入 --mano_dir
"""

import os
import numpy as np
import torch
import torch.nn as nn


class MANOWrapper(nn.Module):
    """
    轻量包装, 暴露:
      .layer(pose, shape) -> (vertices, joints)   pose=(B,48), shape=(B,10)
      .get_3d_joints(vertices) -> joints (B,21,3)
      .face  -> numpy (F, 3) face indices
    """

    # MANO joint regressor: maps 778 vertices → 21 joints
    # (loaded from MANO model file)
    J_REGRESSOR = None

    def __init__(self, mano_dir: str = "./models/mano", use_pca: bool = False):
        super().__init__()
        self.mano_dir = mano_dir

        try:
            from manopth.manolayer import ManoLayer
            self._mano = ManoLayer(
                mano_root   = mano_dir,
                use_pca     = use_pca,
                ncomps      = 45,      # PCA components (ignored if use_pca=False)
                flat_hand_mean = False,
                side        = "right",
            )
            self.layer = self._mano_forward
            self.face  = self._mano.th_faces.numpy()  # (F, 3)
            self._has_mano = True

            # Joint regressor from MANO
            J_reg = self._mano.th_J_regressor.numpy()  # sparse (21, 778)
            self.register_buffer("j_regressor",
                                 torch.from_numpy(J_reg.toarray()
                                                  if hasattr(J_reg, "toarray")
                                                  else J_reg).float())
        except ImportError:
            print("[WARN] manopth not found. Using stub MANO (outputs zeros).")
            print("       Install: pip install -e manopth/")
            self._has_mano = False
            self.face = np.zeros((1538, 3), dtype=np.int64)  # placeholder face count
            # Stub regressor
            J = np.zeros((21, 778), dtype=np.float32)
            self.register_buffer("j_regressor", torch.from_numpy(J))

    def _mano_forward(self, pose: torch.Tensor, shape: torch.Tensor):
        """Calls manopth and returns (vertices, joints) in meters."""
        verts, joints = self._mano(pose, shape)
        return verts / 1000.0, joints / 1000.0  # mm → m

    def layer(self, pose, shape):
        """Stub forward when manopth not available."""
        B = pose.shape[0]
        verts  = torch.zeros(B, 778, 3, device=pose.device)
        joints = torch.zeros(B, 21, 3, device=pose.device)
        return verts, joints

    def get_3d_joints(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Regress 21 hand joints from 778 mesh vertices.
        vertices: (B, 778, 3)
        returns:  (B, 21, 3)
        """
        # j_regressor: (21, 778)
        return torch.einsum("jv,bvd->bjd", self.j_regressor, vertices)

    def forward(self, pose, shape):
        return self.layer(pose, shape)
