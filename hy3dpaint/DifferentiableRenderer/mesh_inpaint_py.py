"""Pure-Python port of meshVerticeInpaint (mesh-aware texture inpainting).

The PyTorch pipeline uses a C++ extension (mesh_inpaint_processor.cpp) that
first propagates colors across the mesh's vertex graph, *then* falls back to
cv2.inpaint. Without step 1, cv2.inpaint must fill huge UV-atlas gaps using
only 2D neighborhood info and produces noisy bleeding.

Algorithm:
  1. Build vertex adjacency graph from face connectivity.
  2. Seed each vertex's color from the texel it lands on (if that texel is
     already painted by back-projection).
  3. For each uncolored vertex, average colors from colored neighbors using
     inverse-square-distance weights. Iterate until convergence.
  4. Rasterize each face into UV space and fill every interior texel with
     barycentric-interpolated vertex color (this is the dense step that the
     vertex-only stamping was missing).

The remaining gaps are left to cv2.inpaint.
"""

from typing import Tuple

import numpy as np


def _build_vertex_graph(pos_idx: np.ndarray, vtx_num: int) -> list:
    """For each vertex, list of connected vertex indices (via face edges)."""
    G = [[] for _ in range(vtx_num)]
    for i in range(pos_idx.shape[0]):
        for k in range(3):
            v0 = int(pos_idx[i, k])
            v1 = int(pos_idx[i, (k + 1) % 3])
            G[v0].append(v1)
    return G


def _seed_vertex_colors(
    texture: np.ndarray, mask: np.ndarray, vtx_uv: np.ndarray,
    pos_idx: np.ndarray, uv_idx: np.ndarray, V: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read each vertex's color from its painted UV texels.

    A 3D vertex can show up in several UV islands (UV unwrapping splits along
    seams). Each island gives a potentially different color. We average all
    painted readings per 3D vertex instead of taking the last write — last-
    write-wins introduces visible stripes on faces where one UV vertex was
    seeded from view A and another from view B with slightly different
    diffusion output.
    """
    H, W, C = texture.shape

    flat_uv_idx = uv_idx.reshape(-1)
    uv_xy = vtx_uv[flat_uv_idx]
    cols = np.clip(np.round(uv_xy[:, 0] * (W - 1)).astype(np.int64), 0, W - 1)
    rows = np.clip(np.round(uv_xy[:, 1] * (H - 1)).astype(np.int64), 0, H - 1)

    flat_pos = pos_idx.reshape(-1)
    painted = mask[rows, cols] > 0
    colors = texture[rows, cols]

    valid_v = flat_pos[painted]
    valid_c = colors[painted]

    vtx_color = np.zeros((V, C), dtype=np.float32)
    vtx_count = np.zeros(V, dtype=np.float32)
    np.add.at(vtx_color, valid_v, valid_c)
    np.add.at(vtx_count, valid_v, 1.0)

    vtx_mask = (vtx_count > 0).astype(np.float32)
    nonzero = vtx_count > 0
    vtx_color[nonzero] /= vtx_count[nonzero, None]
    return vtx_color, vtx_mask


def _propagate_colors(
    vtx_color: np.ndarray, vtx_mask: np.ndarray,
    vtx_pos: np.ndarray, G: list,
) -> None:
    """Smooth uncolored vertices using inverse-square-dist weighted neighbors.

    Modifies vtx_color, vtx_mask in place.
    """
    V, C = vtx_color.shape
    smooth_count = 2
    last_uncolored = -1
    uncolored = np.where(vtx_mask == 0)[0].tolist()

    while smooth_count > 0 and uncolored:
        next_uncolored = []
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
                next_uncolored.append(v)

        if len(next_uncolored) == last_uncolored:
            smooth_count -= 1
        else:
            smooth_count += 1
        last_uncolored = len(next_uncolored)
        uncolored = next_uncolored


def _rasterize_face_with_colors(
    tex: np.ndarray, msk: np.ndarray,
    uv0: np.ndarray, uv1: np.ndarray, uv2: np.ndarray,
    c0: np.ndarray, c1: np.ndarray, c2: np.ndarray,
) -> None:
    """Rasterize a triangle in UV space and fill texels with bary-interpolated colors.

    All UV coords are in [0, 1]. Modifies tex, msk in place.
    """
    H, W, _ = tex.shape

    # Convert UVs to pixel coordinates (col = u*W, row = v*H, V already flipped)
    p0 = np.array([uv0[0] * (W - 1), uv0[1] * (H - 1)], dtype=np.float32)
    p1 = np.array([uv1[0] * (W - 1), uv1[1] * (H - 1)], dtype=np.float32)
    p2 = np.array([uv2[0] * (W - 1), uv2[1] * (H - 1)], dtype=np.float32)

    # Bounding box (clipped to image)
    x_min = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
    x_max = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), W - 1)
    y_min = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
    y_max = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), H - 1)
    if x_max < x_min or y_max < y_min:
        return

    # Edge-function rasterization with barycentric coords
    xs, ys = np.meshgrid(
        np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1),
        indexing="xy",
    )
    xs = xs.astype(np.float32) + 0.5  # pixel centers
    ys = ys.astype(np.float32) + 0.5

    # Barycentric coordinates
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(denom) < 1e-9:
        return
    inv_denom = 1.0 / denom
    w0 = ((p1[1] - p2[1]) * (xs - p2[0]) + (p2[0] - p1[0]) * (ys - p2[1])) * inv_denom
    w1 = ((p2[1] - p0[1]) * (xs - p2[0]) + (p0[0] - p2[0]) * (ys - p2[1])) * inv_denom
    w2 = 1.0 - w0 - w1

    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    # Interpolate colors
    rows_local, cols_local = np.where(inside)
    rows_global = rows_local + y_min
    cols_global = cols_local + x_min
    bw0 = w0[inside][:, None]
    bw1 = w1[inside][:, None]
    bw2 = w2[inside][:, None]
    interp = bw0 * c0 + bw1 * c1 + bw2 * c2

    tex[rows_global, cols_global] = interp
    msk[rows_global, cols_global] = 255


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
        vtx_uv: (V_uv, 2) float32 UV coords (V already flipped by set_mesh).
        pos_idx: (F, 3) int face->vertex indices.
        uv_idx: (F, 3) int face->uv indices.

    Returns:
        Tuple (new_texture, new_mask).
    """
    tex = texture.copy()
    msk = mask.copy()
    V = vtx_pos.shape[0]

    G = _build_vertex_graph(pos_idx, V)
    vtx_color, vtx_mask = _seed_vertex_colors(
        tex, msk, vtx_uv, pos_idx, uv_idx, V,
    )
    _propagate_colors(vtx_color, vtx_mask, vtx_pos, G)

    # Densely fill each face's UV island with bary-interpolated colors
    for f in range(pos_idx.shape[0]):
        v0, v1, v2 = int(pos_idx[f, 0]), int(pos_idx[f, 1]), int(pos_idx[f, 2])
        if vtx_mask[v0] < 1.0 or vtx_mask[v1] < 1.0 or vtx_mask[v2] < 1.0:
            continue
        u0, u1, u2 = int(uv_idx[f, 0]), int(uv_idx[f, 1]), int(uv_idx[f, 2])
        _rasterize_face_with_colors(
            tex, msk,
            vtx_uv[u0], vtx_uv[u1], vtx_uv[u2],
            vtx_color[v0], vtx_color[v1], vtx_color[v2],
        )

    return tex, msk
