# ----------------------------------------------------------------------------------------------
# Wavefront OBJ / MTL exporter for hand meshes.
# Writes .obj + .mtl with a simple flesh-colored material so it's viewable immediately
# in MeshLab / Blender / Preview without further editing.
# ----------------------------------------------------------------------------------------------

import os
import numpy as np


_DEFAULT_MTL_NAME = "hand_material"
_DEFAULT_MTL = f"""# Auto-generated MTL
newmtl {_DEFAULT_MTL_NAME}
Ka 0.30 0.22 0.20
Kd 0.85 0.65 0.55
Ks 0.15 0.15 0.15
Ns 25.0
d  1.0
illum 2
"""


def save_mesh_obj(vertices, faces, out_path, material_name=_DEFAULT_MTL_NAME, write_mtl=True):
    """
    Save a triangle mesh as Wavefront OBJ (+ optional MTL side-file).

    Args:
        vertices:  (N, 3) numpy array or torch tensor (float).
        faces:     (F, 3) numpy array or torch tensor (int, 0-indexed).
        out_path:  Path ending in '.obj'.  MTL will be written next to it with same stem.
        material_name: material ID referenced by the OBJ.
        write_mtl: if True, also produce an MTL file.
    """
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3

    out_path = os.fspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stem, _ = os.path.splitext(out_path)
    mtl_path = stem + ".mtl"

    if write_mtl:
        with open(mtl_path, "w") as f:
            f.write(_DEFAULT_MTL)

    with open(out_path, "w") as f:
        if write_mtl:
            f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        f.write(f"usemtl {material_name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # OBJ faces are 1-indexed
        for face in faces + 1:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    return out_path, (mtl_path if write_mtl else None)
