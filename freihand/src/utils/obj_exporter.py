"""
OBJ / MTL 导出工具
将 MANO 手部网格的顶点 + 面索引导出为标准 Wavefront OBJ + MTL 格式.

使用示例:
    from src.utils.obj_exporter import export_obj_mtl, export_batch_obj_mtl

    # 单个样本
    export_obj_mtl(
        vertices  = pred_v.cpu().numpy(),   # (778, 3)
        faces     = mano.face,              # (F, 3) 0-indexed int
        obj_path  = "output/sample_000.obj",
        mtl_path  = "output/sample_000.mtl",
        color_rgb = (0.8, 0.7, 0.6),        # 皮肤色
    )

    # 批量导出
    export_batch_obj_mtl(vertices_list, faces, out_dir, names)
"""

import os
import os.path as op
import numpy as np
from typing import Optional, List, Tuple


# MANO 手部关节颜色 (皮肤色 RGB)
DEFAULT_COLOR = (0.87, 0.72, 0.53)


def export_obj_mtl(
    vertices : np.ndarray,          # (V, 3)  float, meters
    faces    : np.ndarray,          # (F, 3)  int,   0-indexed
    obj_path : str,
    mtl_path : Optional[str] = None,
    color_rgb: Tuple[float, float, float] = DEFAULT_COLOR,
    uv_coords: Optional[np.ndarray] = None,   # (V, 2) texture UV (optional)
) -> None:
    """
    导出单个网格为 OBJ + MTL 文件.

    MTL 使用 Phong 着色模型 (无纹理贴图, 仅颜色).
    如需纹理贴图, 传入 uv_coords 并在 MTL 中指定 map_Kd.

    OBJ 文件格式:
        # 注释
        mtllib <name>.mtl
        usemtl hand_material
        v  x y z        <- 顶点坐标 (m)
        vt u v          <- UV 坐标 (可选)
        f  v1 v2 v3     <- 三角面 (1-indexed)
    """
    os.makedirs(op.dirname(op.abspath(obj_path)), exist_ok=True)

    # ── 决定 MTL 文件名 ────────────────────────────────────
    if mtl_path is None:
        mtl_path = op.splitext(obj_path)[0] + ".mtl"
    mtl_name = op.basename(mtl_path)
    mat_name = "hand_material"

    # ── 写 OBJ ─────────────────────────────────────────────
    with open(obj_path, "w") as f:
        f.write("# FastMETRO Hand Mesh Export\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write(f"usemtl {mat_name}\n\n")

        # Vertices
        f.write("# Vertices\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # UV coordinates (optional)
        if uv_coords is not None:
            f.write("\n# UV Coordinates\n")
            for uv in uv_coords:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        # Faces (1-indexed)
        f.write("\n# Faces\n")
        if uv_coords is not None:
            # f v/vt for each vertex
            for tri in faces:
                i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
        else:
            for tri in faces:
                i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
                f.write(f"f {i0} {i1} {i2}\n")

    # ── 写 MTL ─────────────────────────────────────────────
    r, g, b = color_rgb
    with open(mtl_path, "w") as f:
        f.write("# FastMETRO Hand Mesh Material\n")
        f.write(f"newmtl {mat_name}\n")
        f.write(f"Ka {r:.4f} {g:.4f} {b:.4f}\n")   # ambient
        f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")   # diffuse
        f.write(f"Ks 0.2000 0.2000 0.2000\n")       # specular
        f.write(f"Ns 10.0\n")                        # shininess
        f.write("illum 2\n")                         # Phong shading
        f.write("d 1.0\n")                           # opacity


def export_batch_obj_mtl(
    vertices_list: List[np.ndarray],      # list of (V, 3)
    faces        : np.ndarray,            # (F, 3) shared topology (MANO)
    out_dir      : str,
    names        : Optional[List[str]] = None,
    color_rgb    : Tuple[float, float, float] = DEFAULT_COLOR,
) -> List[str]:
    """
    批量导出, 返回写出的 OBJ 路径列表.

    Parameters
    ----------
    vertices_list : predicted vertices per sample
    faces         : MANO face array (shared for all samples)
    out_dir       : output directory
    names         : optional filename stems (without extension)
    """
    os.makedirs(out_dir, exist_ok=True)
    exported = []

    for i, verts in enumerate(vertices_list):
        stem     = names[i] if names else f"{i:06d}"
        obj_path = op.join(out_dir, f"{stem}.obj")
        mtl_path = op.join(out_dir, f"{stem}.mtl")
        export_obj_mtl(verts, faces, obj_path, mtl_path, color_rgb)
        exported.append(obj_path)

    return exported


# ─────────────────────────────────────────────────────────
#  Quick sanity check (run as __main__)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("Testing OBJ/MTL export with random mesh...")
    V = 778
    F = 1538
    verts = np.random.randn(V, 3).astype(np.float32) * 0.1
    faces = np.random.randint(0, V, size=(F, 3))

    with tempfile.TemporaryDirectory() as td:
        obj_p = op.join(td, "test_hand.obj")
        mtl_p = op.join(td, "test_hand.mtl")
        export_obj_mtl(verts, faces, obj_p, mtl_p)

        # Read back and verify
        with open(obj_p) as f:
            lines = f.readlines()
        v_lines = [l for l in lines if l.startswith("v ")]
        f_lines = [l for l in lines if l.startswith("f ")]
        assert len(v_lines) == V, f"Expected {V} vertices, got {len(v_lines)}"
        assert len(f_lines) == F, f"Expected {F} faces, got {len(f_lines)}"
        print(f"  OBJ: {len(v_lines)} vertices, {len(f_lines)} faces — OK")

        with open(mtl_p) as f:
            mtl = f.read()
        assert "newmtl" in mtl and "Kd" in mtl
        print("  MTL: valid — OK")

    print("Export test passed!")
