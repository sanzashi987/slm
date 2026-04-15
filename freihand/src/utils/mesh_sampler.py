"""
Mesh Sampler (手部网格下采样 / 上采样)
778顶点 MANO mesh → 195 sub-sampled vertices

原始 FastMETRO / METRO 使用 graphcmr 生成的预计算采样矩阵.
本实现提供:
  1. 尝试从 .npz 文件加载预计算矩阵 (推荐)
  2. 如文件不存在, 使用基于欧氏距离的简单 FPS 近似 (快速 MVP 替代)

推荐做法: 从 FastMETRO 仓库或 METRO 仓库下载预计算矩阵:
  src/modeling/data/mano_195_adj_matrix.npz  (sub→full 的稀疏上采样矩阵)
"""

import os
import numpy as np
import torch
import torch.nn as nn


VERTEX_NUM_FULL = 778
VERTEX_NUM_SUB  = 195


class Mesh:
    """
    提供 downsample / upsample 两个操作.
    如果找到预计算矩阵文件就加载, 否则构造近似映射.
    """

    # 找矩阵文件的候选路径
    _SEARCH_PATHS = [
        "src/modeling/data/mesh_downsampling.npz",
        "models/mesh_downsampling.npz",
        os.path.join(os.path.dirname(__file__), "../modeling/data/mesh_downsampling.npz"),
    ]

    def __init__(self):
        self._D = None   # downsample matrix (VERTEX_NUM_FULL → VERTEX_NUM_SUB)
        self._U = None   # upsample matrix   (VERTEX_NUM_SUB  → VERTEX_NUM_FULL)
        self._device = torch.device("cpu")

        for p in self._SEARCH_PATHS:
            if os.path.exists(p):
                self._load_matrices(p)
                break

        if self._D is None:
            print("[Mesh] Pre-computed mesh matrices not found. "
                  "Building approximate sampling (less accurate).")
            self._build_approx()

    def _load_matrices(self, path: str):
        data = np.load(path, allow_pickle=True)
        # Expected keys: 'down' shape (195, 778), 'up' shape (778, 195)
        # (sparse matrices saved as dense or scipy sparse)
        D = data["down"]
        U = data["up"]
        if hasattr(D, "toarray"):
            D = D.toarray()
        if hasattr(U, "toarray"):
            U = U.toarray()
        self._D = torch.from_numpy(D.astype(np.float32))  # (195, 778)
        self._U = torch.from_numpy(U.astype(np.float32))  # (778, 195)
        print(f"[Mesh] Loaded mesh matrices from {path}: "
              f"D={tuple(self._D.shape)}, U={tuple(self._U.shape)}")

    def _build_approx(self):
        """
        Fallback: build a simple uniformly-spaced index map.
        downsample: pick VERTEX_NUM_SUB evenly-spaced indices from 778 vertices
        upsample:   nearest-neighbour from sub-mesh to full mesh
        This is NOT as accurate as the graph-based matrices but works for code verification.
        """
        rng = np.random.default_rng(42)
        idx_down = np.sort(rng.choice(VERTEX_NUM_FULL, VERTEX_NUM_SUB, replace=False))
        self._idx_down = idx_down  # (195,) indices into full mesh

        # For upsample: each full vertex maps to nearest sub-vertex (identity for sub-indices)
        # Simple approach: nearest by linear index
        idx_up = np.zeros(VERTEX_NUM_FULL, dtype=np.int64)
        for i in range(VERTEX_NUM_FULL):
            idx_up[i] = int(np.argmin(np.abs(idx_down - i)))
        self._idx_up = idx_up  # (778,) indices into sub-mesh
        self._approx = True

    @property
    def _is_approx(self):
        return self._D is None

    def downsample(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        vertices: (B, 778, 3)
        returns:  (B, 195, 3)
        """
        if self._is_approx:
            return vertices[:, self._idx_down, :]
        D = self._D.to(vertices.device)
        # (195, 778) × (778, 3) = (195, 3) per sample
        return torch.einsum("ij,bjk->bik", D, vertices)

    def upsample(self, sub_vertices: torch.Tensor) -> torch.Tensor:
        """
        sub_vertices: (B, 195, 3)
        returns:      (B, 778, 3)
        """
        if self._is_approx:
            return sub_vertices[:, self._idx_up, :]
        U = self._U.to(sub_vertices.device)
        return torch.einsum("ij,bjk->bik", U, sub_vertices)
