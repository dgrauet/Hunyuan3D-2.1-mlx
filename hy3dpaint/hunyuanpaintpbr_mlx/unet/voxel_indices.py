"""Multi-resolution discrete voxel index computation for 3D RoPE.

MLX port of `calc_multires_voxel_idxs` and `compute_discrete_voxel_indice`
from hunyuanpaintpbr/unet/modules.py. Used by the multiview attention to
inject 3D-aware positional encoding so the model knows which texels across
views correspond to the same world-space point.
"""

from typing import Dict, List

import mlx.core as mx
import numpy as np


def compute_discrete_voxel_indice(
    position: np.ndarray,
    grid_resolution: int = 8,
    voxel_resolution: int = 128,
) -> mx.array:
    """Quantize a 4D position-map tensor to discrete voxel indices.

    Args:
        position: (B, N, 3, H, W) float32 numpy array with values in [0, 1].
            Background pixels are encoded as 1.0 in every channel.
        grid_resolution: spatial downsampling factor (must divide H and W).
        voxel_resolution: number of bins per axis.

    Returns:
        (B, N, 3, grid_res, grid_res) int32 mx.array of voxel bin indices.
    """
    B, N, _, H, W = position.shape
    assert H % grid_resolution == 0 and W % grid_resolution == 0

    # Background: pixels where all 3 channels equal 1.0
    valid_mask = (position != 1.0).all(axis=2, keepdims=True)
    valid_mask = np.broadcast_to(valid_mask, position.shape)
    pos = position.copy()
    pos[~valid_mask] = 0.0

    grid_h = H // grid_resolution
    grid_w = W // grid_resolution

    # (B, N, 3, num_h, grid_h, num_w, grid_w)
    pos_r = pos.reshape(B, N, 3, grid_resolution, grid_h, grid_resolution, grid_w)
    valid_r = valid_mask.reshape(B, N, 3, grid_resolution, grid_h, grid_resolution, grid_w)

    # Sum over (grid_h, grid_w) per cell
    grid_position = pos_r.sum(axis=(4, 6))           # (B, N, 3, num_h, num_w)
    count_masked = valid_r.sum(axis=(4, 6))          # same shape

    grid_position = grid_position / np.clip(count_masked, 1, None)
    voxel_mask_thres = (H // grid_resolution) * (W // grid_resolution) // 16
    grid_position[count_masked < voxel_mask_thres] = 0.0

    # Quantize to [0, voxel_resolution - 1]
    grid_position = np.clip(grid_position, 0, 1)
    voxel_indices = np.round(grid_position * (voxel_resolution - 1)).astype(np.int32)
    return mx.array(voxel_indices)


def calc_multires_voxel_idxs(
    position_maps: np.ndarray,
    grid_resolutions: List[int] = (64, 32, 16, 8),
    voxel_resolutions: List[int] = (512, 256, 128, 64),
) -> Dict[int, dict]:
    """Build a multi-res voxel index dictionary keyed by sequence length.

    Args:
        position_maps: (B, N, 3, H, W) float32 numpy array in [0, 1].

    Returns:
        Dict ``{seq_len: {"voxel_indices": (B, seq_len, 3) int32 mx.array,
                           "voxel_resolution": int}}``
        where seq_len = N * grid_res * grid_res.
    """
    out: Dict[int, dict] = {}
    for grid_res, vox_res in zip(grid_resolutions, voxel_resolutions):
        idx = compute_discrete_voxel_indice(position_maps, grid_res, vox_res)
        # (B, N, 3, gr, gr) -> (B, N*gr*gr, 3)
        idx_np = np.array(idx)
        B, N, _, gr, _ = idx_np.shape
        idx_flat = idx_np.transpose(0, 1, 3, 4, 2).reshape(B, N * gr * gr, 3)
        out[N * gr * gr] = {
            "voxel_indices": mx.array(idx_flat),
            "voxel_resolution": vox_res,
        }
    return out
