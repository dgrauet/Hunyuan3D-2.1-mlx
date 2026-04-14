"""Pure-Python port of meshVerticeInpaint (mesh-aware texture inpainting).

The PyTorch pipeline uses a C++ extension (mesh_inpaint_processor.cpp) that
first propagates colors across the mesh's vertex graph, *then* falls back to
cv2.inpaint. Without step 1, cv2.inpaint must fill huge UV-atlas gaps using
only 2D neighborhood info and produces noisy bleeding.

This port implements the "smooth" variant (the PyTorch default):
  1. Build vertex adjacency graph from face connectivity.
  2. Seed each vertex's color from the texel it lands on (if that texel is
     already painted by back-projection).
  3. For each uncolored vertex, average colors from colored neighbors using
     inverse-square-distance weights. Iterate 2 times.
  4. Stamp each newly-colored vertex's color back into its UV-space texel.

The remaining gaps are left to cv2.inpaint. With the mesh-aware step done
first, the residual gaps are small and cv2.inpaint produces clean results.
"""

from typing import Tuple

import numpy as np


def _calculate_uv_coords(vtx_uv: np.ndarray, vtx_uv_idx: int,
                          tex_h: int, tex_w: int) -> Tuple[int, int]:
    """Convert a UV coord to (row, col) texel indices.

    MeshRenderMLX.set_mesh already flips V (uv[:, 1] = 1 - uv[:, 1]) so the
    texel row is a straight ``v * (h - 1)``, matching how back_project writes
    into the atlas.
    """
    uv_v = int(round(vtx_uv[vtx_uv_idx, 0] * (tex_w - 1)))
    uv_u = int(round(vtx_uv[vtx_uv_idx, 1] * (tex_h - 1)))
    return uv_u, uv_v


def _build_vertex_graph(pos_idx: np.ndarray, vtx_num: int) -> list:
    """For each vertex, list of connected vertex indices (via face edges)."""
    G = [[] for _ in range(vtx_num)]
    for i in range(pos_idx.shape[0]):
        for k in range(3):
            v0 = int(pos_idx[i, k])
            v1 = int(pos_idx[i, (k + 1) % 3])
            G[v0].append(v1)
    return G


def mesh_vertex_inpaint(
    texture: np.ndarray,
    mask: np.ndarray,
    vtx_pos: np.ndarray,
    vtx_uv: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate colors across the mesh to fill UV gaps.

    Args:
        texture: (H, W, C) float32 in [0, 1].
        mask: (H, W) uint8, 255 = painted, 0 = unpainted.
        vtx_pos: (V, 3) float32 vertex positions.
        vtx_uv: (V_uv, 2) float32 UV coords.
        pos_idx: (F, 3) int face->vertex indices.
        uv_idx: (F, 3) int face->uv indices.

    Returns:
        Tuple (new_texture, new_mask).
    """
    tex = texture.copy()
    msk = mask.copy()
    H, W, C = tex.shape
    V = vtx_pos.shape[0]

    G = _build_vertex_graph(pos_idx, V)

    # Seed vertex colors from existing texels
    vtx_color = np.zeros((V, C), dtype=np.float32)
    vtx_mask = np.zeros(V, dtype=np.float32)
    uncolored = []
    seen = set()
    for i in range(pos_idx.shape[0]):
        for k in range(3):
            v = int(pos_idx[i, k])
            uv = int(uv_idx[i, k])
            uu, vv = _calculate_uv_coords(vtx_uv, uv, H, W)
            if msk[uu, vv] > 0:
                vtx_color[v] = tex[uu, vv]
                vtx_mask[v] = 1.0
            elif v not in seen:
                uncolored.append(v)
                seen.add(v)

    # Smooth propagation: 2 iterations of averaging from colored neighbors
    smooth_count = 2
    last_uncolored = -1
    while smooth_count > 0:
        cur_uncolored = 0
        for v in uncolored:
            if vtx_mask[v] > 0:
                continue
            p0 = vtx_pos[v]
            acc = np.zeros(C, dtype=np.float32)
            tw = 0.0
            for vn in G[v]:
                if vtx_mask[vn] > 0:
                    d = np.linalg.norm(vtx_pos[vn] - p0)
                    w = 1.0 / max(d, 1e-4)
                    w = w * w
                    acc += vtx_color[vn] * w
                    tw += w
            if tw > 0:
                vtx_color[v] = acc / tw
                vtx_mask[v] = 1.0
            else:
                cur_uncolored += 1

        if last_uncolored == cur_uncolored:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored = cur_uncolored

    # Stamp vertex colors back into UV-space texels
    for i in range(pos_idx.shape[0]):
        for k in range(3):
            v = int(pos_idx[i, k])
            uv = int(uv_idx[i, k])
            if vtx_mask[v] >= 1.0:
                uu, vv = _calculate_uv_coords(vtx_uv, uv, H, W)
                tex[uu, vv] = vtx_color[v]
                msk[uu, vv] = 255

    return tex, msk
