# ----------------------------------------------------------------------------------------------
# Minimal MANO wrapper + mesh sampling
# Derived from: kaist-ami/FastMETRO (MIT License) & microsoft/MeshTransformer
# Requires `manopth` (https://github.com/hassony2/manopth) AND MANO_RIGHT.pkl (download from MANO).
# ----------------------------------------------------------------------------------------------

import os
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn

from manopth.manolayer import ManoLayer


HAND_JOINT_NAMES = (
    "Wrist",
    "Thumb_1", "Thumb_2", "Thumb_3", "Thumb_4",
    "Index_1", "Index_2", "Index_3", "Index_4",
    "Middle_1", "Middle_2", "Middle_3", "Middle_4",
    "Ring_1", "Ring_2", "Ring_3", "Ring_4",
    "Pinky_1", "Pinky_2", "Pinky_3", "Pinky_4",
)
WRIST_IDX = HAND_JOINT_NAMES.index("Wrist")


class MANO(nn.Module):
    """MANO right-hand layer wrapper (flat_hand_mean=False, use_pca=False)."""

    def __init__(self, mano_dir):
        super().__init__()
        self.mano_dir = mano_dir
        self.layer = ManoLayer(mano_root=mano_dir, flat_hand_mean=False, use_pca=False)
        self.vertex_num = 778
        self.joint_num = 21
        self.face = self.layer.th_faces.numpy()

        # Extended joint regressor: includes fingertips via direct vertex lookup (GraphCMR/METRO style).
        jr = self.layer.th_J_regressor.numpy()
        fingertip_idx = {745: "thumb", 317: "index", 445: "middle", 556: "ring", 673: "pinky"}
        onehots = []
        for v_idx in fingertip_idx.keys():
            oh = np.zeros((1, jr.shape[1]), dtype=np.float32)
            oh[0, v_idx] = 1.0
            onehots.append(oh)
        jr = np.concatenate([jr] + onehots, axis=0)
        # Re-order to match HAND_JOINT_NAMES (Wrist, Thumb_1..4, Index_1..4, ...)
        reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        jr = jr[reorder, :]
        self.register_buffer("joint_regressor_torch", torch.from_numpy(jr).float())

    def get_3d_joints(self, vertices):
        """vertices: B x 778 x 3  ->  B x 21 x 3"""
        return torch.einsum("bik,ji->bjk", [vertices, self.joint_regressor_torch])


class _SparseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        with torch.autocast(device_type="cuda", enabled=False):
            result = torch.sparse.mm(sparse, dense.float())
        return result.to(dense.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (sparse,) = ctx.saved_tensors
        if ctx.req_grad:
            with torch.autocast(device_type="cuda", enabled=False):
                grad_input = torch.sparse.mm(sparse.t(), grad_output.float())
            grad_input = grad_input.to(grad_output.dtype)
        return None, grad_input


def _spmm(sparse, dense):
    return _SparseMM.apply(sparse, dense)


def _scipy_to_pytorch(A, U, D):
    ptU, ptD = [], []
    for u in U:
        u = scipy.sparse.coo_matrix(u)
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse_coo_tensor(i, v, u.shape))
    for d in D:
        d = scipy.sparse.coo_matrix(d)
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse_coo_tensor(i, v, d.shape))
    return ptU, ptD


class Mesh(object):
    """Graph downsampling / upsampling for MANO mesh (778 <-> 195)."""

    def __init__(self, sampling_npz_path, num_downsampling=1, device=torch.device("cuda")):
        data = np.load(sampling_npz_path, encoding="latin1", allow_pickle=True)
        U, D = _scipy_to_pytorch(data["A"], data["U"], data["D"])
        # self._U = [u.to(device) for u in U]
        # self._D = [d.to(device) for d in D]
        self._U = [u.to(device=device, dtype=torch.float32) for u in U]
        self._D = [d.to(device=device, dtype=torch.float32) for d in D]
        self.num_downsampling = num_downsampling

    @staticmethod
    def _batched_spmm(sparse, x):
        """Apply a sparse matrix (M x N) to a batched dense (B x N x C) -> (B x M x C)
        as a single sparse @ dense matmul (no python loop over batch)."""
        if x.ndim < 3:
            return _spmm(sparse, x)
        B, N, C = x.shape
        # (B, N, C) -> (N, B*C)
        x_flat = x.permute(1, 0, 2).reshape(N, B * C)
        y_flat = _spmm(sparse, x_flat)                   # (M, B*C)
        M = y_flat.shape[0]
        # (M, B*C) -> (B, M, C)
        return y_flat.reshape(M, B, C).permute(1, 0, 2).contiguous()

    def downsample(self, x, n1=0, n2=None):
        if n2 is None:
            n2 = self.num_downsampling
        for i in range(n1, n2):
            x = self._batched_spmm(self._D[i], x)
        return x

    def upsample(self, x, n1=1, n2=0):
        for i in reversed(range(n2, n1)):
            x = self._batched_spmm(self._U[i], x)
        return x
