"""Camera utilities for MLX mesh rendering.

Direct port of camera_utils.py from PyTorch to MLX.
Projection matrices stay as numpy (computed once, small).
transform_pos operates on MLX arrays.
"""

import math

import mlx.core as mx
import numpy as np


def transform_pos(mtx, pos, keepdim=False):
    """Transform positions by a 4x4 matrix.

    Args:
        mtx: (4, 4) numpy or mx.array transformation matrix.
        pos: (N, 3) or (N, 4) mx.array positions.
        keepdim: if False, prepend batch dim -> (1, N, 4).

    Returns:
        (1, N, 4) or (N, 4) mx.array of transformed positions.
    """
    if isinstance(mtx, np.ndarray):
        t_mtx = mx.array(mtx)
    else:
        t_mtx = mtx

    if pos.shape[-1] == 3:
        ones = mx.ones((pos.shape[0], 1), dtype=pos.dtype)
        posw = mx.concatenate([pos, ones], axis=1)
    else:
        posw = pos

    result = posw @ t_mtx.T
    if keepdim:
        return result
    return mx.expand_dims(result, axis=0)


def get_mv_matrix(elev, azim, camera_distance, center=None):
    """Build world-to-camera matrix from spherical coordinates.

    Returns:
        (4, 4) numpy float32 array.
    """
    elev = -elev
    azim += 90

    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)

    camera_position = np.array([
        camera_distance * math.cos(elev_rad) * math.cos(azim_rad),
        camera_distance * math.cos(elev_rad) * math.sin(azim_rad),
        camera_distance * math.sin(elev_rad),
    ])

    if center is None:
        center = np.array([0, 0, 0])
    else:
        center = np.array(center)

    lookat = center - camera_position
    lookat = lookat / np.linalg.norm(lookat)

    up = np.array([0, 0, 1.0])
    right = np.cross(lookat, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, lookat)
    up = up / np.linalg.norm(up)

    c2w = np.concatenate(
        [np.stack([right, up, -lookat], axis=-1), camera_position[:, None]],
        axis=-1,
    )

    w2c = np.zeros((4, 4), dtype=np.float32)
    w2c[:3, :3] = c2w[:3, :3].T
    w2c[:3, 3:] = -c2w[:3, :3].T @ c2w[:3, 3:]
    w2c[3, 3] = 1.0
    return w2c


def get_orthographic_projection_matrix(
    left=-1, right=1, bottom=-1, top=1, near=0, far=2
):
    """Orthographic projection matrix.

    Returns:
        (4, 4) numpy float32 array.
    """
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2 / (right - left)
    m[1, 1] = 2 / (top - bottom)
    m[2, 2] = -2 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def get_perspective_projection_matrix(fovy, aspect_wh, near, far):
    """Perspective projection matrix.

    Returns:
        (4, 4) numpy float32 array.
    """
    fovy_rad = math.radians(fovy)
    return np.array(
        [
            [1.0 / (math.tan(fovy_rad / 2.0) * aspect_wh), 0, 0, 0],
            [0, 1.0 / math.tan(fovy_rad / 2.0), 0, 0],
            [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
