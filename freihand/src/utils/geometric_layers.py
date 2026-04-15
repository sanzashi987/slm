"""
几何工具函数
"""
import torch
from torch import Tensor


def orthographic_projection(points_3d: Tensor, camera: Tensor) -> Tensor:
    """
    弱透视 (orthographic) 投影: 将 3D 点投影到 2D 图像平面.

    Args:
        points_3d : (B, N, 3) 3D 点, 坐标已相对于 wrist root 归一化
        camera    : (B, 3)    [scale, tx, ty]

    Returns:
        (B, N, 2) 2D 投影坐标, 范围约 [-1, 1]
    """
    scale = camera[:, 0:1].unsqueeze(1)   # (B, 1, 1)
    trans = camera[:, 1:3].unsqueeze(1)   # (B, 1, 2)

    # Project XY, ignore Z for weak-perspective
    projected = scale * points_3d[:, :, :2] + trans
    return projected
