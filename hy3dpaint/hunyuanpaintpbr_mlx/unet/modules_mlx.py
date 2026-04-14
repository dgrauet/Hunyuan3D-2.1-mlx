"""2.5D transformer block and UNet wrapper for HunyuanPaintPBR.

Basic2p5DTransformerBlock: replaces standard BasicTransformerBlock,
adding Material-Dimension (MDA), Multiview (MA), Reference (RA),
and DINO cross-attention.

UNet2p5DConditionModelMLX: wraps UNet2DConditionModel, manages
dual-stream reference processing and condition routing.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attn_processor_mlx import (
    ImageProjModel,
    PoseRoPEAttnProcessor,
    RefAttnProcessor,
    SelfAttnProcessor,
    scaled_dot_product_attention,
    reshape_for_attention,
    reshape_from_attention,
)
from .blocks_mlx import (
    Attention,
    BasicTransformerBlock,
    FeedForward,
    Transformer2DModel,
)


# ---------------------------------------------------------------------------
# Voxel index computation for RoPE
# ---------------------------------------------------------------------------

def calc_voxel_indices(
    position_maps: mx.array,
    grid_resolution: int,
    voxel_resolution: int,
) -> dict:
    """Compute quantized voxel indices from position maps.

    Args:
        position_maps: (B, N_views, H, W, 3) position maps in [0, 1].
        grid_resolution: spatial resolution to downsample to.
        voxel_resolution: quantization levels for voxel grid.

    Returns:
        dict with 'voxel_indices' (B*N, grid_res^2, 3) and 'voxel_resolution'.
    """
    B, N, H, W, C = position_maps.shape

    # Downsample to grid_resolution using simple strided indexing
    stride_h = max(1, H // grid_resolution)
    stride_w = max(1, W // grid_resolution)
    downsampled = position_maps[:, :, ::stride_h, ::stride_w, :]
    # Crop to exact grid_resolution
    downsampled = downsampled[:, :, :grid_resolution, :grid_resolution, :]

    # Quantize to [0, voxel_resolution)
    voxel_idx = (downsampled * (voxel_resolution - 1)).astype(mx.int32)
    voxel_idx = mx.clip(voxel_idx, 0, voxel_resolution - 1)

    # Reshape: (B, N, G, G, 3) → (B*N, G*G, 3)
    voxel_idx = voxel_idx.reshape(B * N, grid_resolution * grid_resolution, 3)

    return {
        "voxel_indices": voxel_idx,
        "voxel_resolution": voxel_resolution,
    }


def calc_multires_voxel_indices(
    position_maps: mx.array,
    grid_resolutions: List[int] = (64, 32, 16, 8),
    voxel_resolutions: List[int] = (512, 256, 128, 64),
) -> Dict[int, dict]:
    """Compute voxel indices at multiple resolutions.

    Returns:
        Dict mapping seq_len (grid_res^2) to voxel index dict.
    """
    result = {}
    for grid_res, vox_res in zip(grid_resolutions, voxel_resolutions):
        seq_len = grid_res * grid_res
        result[seq_len] = calc_voxel_indices(
            position_maps, grid_res, vox_res
        )
    return result


# ---------------------------------------------------------------------------
# Basic2p5DTransformerBlock
# ---------------------------------------------------------------------------

class Basic2p5DTransformerBlock(nn.Module):
    """Enhanced transformer block with 4 attention mechanisms.

    Replaces BasicTransformerBlock in the SD UNet. Adds:
    - MDA: Material-Dimension self-Attention (per-material Q/K/V)
    - MA: Multiview Attention with 3D RoPE
    - RA: Reference Attention (cross-attend to cached reference features)
    - DINO: Cross-attention to DINOv2 vision features

    Processing order per block:
    1. Self-attention (MDA if enabled, else standard)
    2. Reference attention (RA)
    3. Multiview attention (MA)
    4. Cross-attention with text embeddings
    5. DINO cross-attention
    6. Feed-forward
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1024,
        pbr_settings: Tuple[str, ...] = ("albedo", "mr"),
        use_mda: bool = True,
        use_ma: bool = True,
        use_ra: bool = True,
        use_dino: bool = True,
        layer_name: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.pbr_settings = pbr_settings
        self.n_pbr = len(pbr_settings)
        self.use_mda = use_mda
        self.use_ma = use_ma
        self.use_ra = use_ra
        self.use_dino = use_dino
        self.layer_name = layer_name

        # Base self-attention + norms + FFN
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

        # MDA: Material-dimension self-attention
        if use_mda:
            self.mda_processor = SelfAttnProcessor(
                query_dim=dim,
                num_heads=num_attention_heads,
                pbr_settings=pbr_settings,
            )

        # MA: Multiview attention with RoPE
        if use_ma:
            self.norm_ma = nn.LayerNorm(dim)
            self.attn_multiview = PoseRoPEAttnProcessor(
                query_dim=dim,
                num_heads=num_attention_heads,
            )

        # RA: Reference attention
        if use_ra:
            self.norm_ra = nn.LayerNorm(dim)
            self.attn_refview = RefAttnProcessor(
                query_dim=dim,
                num_heads=num_attention_heads,
                pbr_settings=pbr_settings,
            )

        # DINO: Vision feature cross-attention
        if use_dino:
            self.norm_dino = nn.LayerNorm(dim)
            self.attn_dino = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        temb: Optional[mx.array] = None,
        mode: str = "r",
        n_views: int = 6,
        condition_embed_dict: Optional[Dict] = None,
        dino_hidden_states: Optional[mx.array] = None,
        position_voxel_indices: Optional[Dict] = None,
        mva_scale: float = 1.0,
        ref_scale: float = 1.0,
        **kwargs,
    ) -> mx.array:
        """
        Args:
            hidden_states: (B*N_pbr*N_gen, L, C) token features.
            encoder_hidden_states: (B*N_pbr*N_gen, L_text, C) text embeddings.
            mode: "w" = write reference features to cache, "r" = read and use them.
            n_views: number of generated views.
            condition_embed_dict: cache for reference attention features.
            dino_hidden_states: (B, N_dino_tokens, C) DINO features.
            position_voxel_indices: dict for RoPE computation.
            mva_scale: scaling factor for multiview attention residual.
            ref_scale: scaling factor for reference attention residual.

        Returns:
            (B*N_pbr*N_gen, L, C)
        """
        BNpN = hidden_states.shape[0]
        B = BNpN // (self.n_pbr * n_views)

        # --- Step 1: Self-attention (MDA or standard) ---
        norm_hs = self.norm1(hidden_states)

        if self.use_mda and mode != "w":
            # Reshape to (B, N_pbr, N_gen, L, C) for material-wise attention
            L, C = norm_hs.shape[1], norm_hs.shape[2]
            mda_input = norm_hs.reshape(B, self.n_pbr, n_views, L, C)
            attn_out = self.mda_processor(
                mda_input,
                self.attn1.to_q, self.attn1.to_k,
                self.attn1.to_v, self.attn1.to_out,
                n_views=n_views,
            )
            attn_out = attn_out.reshape(BNpN, L, C)
        else:
            attn_out = self.attn1(norm_hs)

        hidden_states = attn_out + hidden_states

        # --- Step 2: Reference attention (RA) ---
        if self.use_ra and condition_embed_dict is not None:
            if mode == "w":
                # Write mode: cache current features for later use
                condition_embed_dict[self.layer_name] = hidden_states
            elif mode == "r" and self.layer_name in condition_embed_dict:
                # Read mode: cross-attend to cached reference features
                ref_features = condition_embed_dict[self.layer_name]
                norm_hs = self.norm_ra(hidden_states)

                L, C = norm_hs.shape[1], norm_hs.shape[2]
                # Reshape: (B*Np*Ng, L, C) → (B, Ng*L, C) using first material
                query_flat = norm_hs.reshape(B, self.n_pbr, n_views, L, C)
                query_flat = query_flat[:, 0].reshape(B, n_views * L, C)

                ra_out = self.attn_refview(
                    query_flat, ref_features,
                )  # (B, N_pbr, Ng*L, C)

                ra_out = ra_out.reshape(B, self.n_pbr, n_views, L, C)
                ra_out = ra_out.reshape(BNpN, L, C)
                hidden_states = ref_scale * ra_out + hidden_states

        # --- Step 3: Multiview attention (MA) ---
        if self.use_ma and n_views > 1 and mode != "w":
            norm_hs = self.norm_ma(hidden_states)
            L, C = norm_hs.shape[1], norm_hs.shape[2]

            # Reshape: (B*Np*Ng, L, C) → (B*Np, Ng*L, C)
            ma_input = norm_hs.reshape(B * self.n_pbr, n_views * L, C)

            # Get position indices for this sequence length
            pos_idx = None
            if position_voxel_indices is not None:
                seq_len = L  # per-view sequence length
                if seq_len in position_voxel_indices:
                    pos_idx = position_voxel_indices[seq_len]

            ma_out = self.attn_multiview(ma_input, position_indices=pos_idx)
            ma_out = ma_out.reshape(BNpN, L, C)
            hidden_states = mva_scale * ma_out + hidden_states

        # --- Step 4: Text cross-attention ---
        hidden_states = (
            self.attn2(self.norm2(hidden_states), encoder_hidden_states)
            + hidden_states
        )

        # --- Step 5: DINO cross-attention ---
        if self.use_dino and dino_hidden_states is not None and mode != "w":
            norm_hs = self.norm_dino(hidden_states)

            # Broadcast DINO features to all materials and views
            # dino: (B, N_dino, C) → (B*Np*Ng, N_dino, C)
            dino_expanded = mx.broadcast_to(
                dino_hidden_states[:, None, None, :, :],
                (B, self.n_pbr, n_views, dino_hidden_states.shape[1], C),
            ).reshape(BNpN, -1, C)

            dino_out = self.attn_dino(norm_hs, dino_expanded)
            hidden_states = dino_out + hidden_states

        # --- Step 6: Feed-forward ---
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Enhanced Transformer2DModel with 2.5D blocks
# ---------------------------------------------------------------------------

class Transformer2p5DModel(nn.Module):
    """Spatial transformer using Basic2p5DTransformerBlock.

    Drop-in replacement for Transformer2DModel in the UNet.
    """

    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 320,
        num_layers: int = 1,
        cross_attention_dim: int = 1024,
        norm_num_groups: int = 32,
        pbr_settings: Tuple[str, ...] = ("albedo", "mr"),
        block_prefix: str = "",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = [
            Basic2p5DTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                pbr_settings=pbr_settings,
                layer_name=f"{block_prefix}_{i}",
            )
            for i in range(num_layers)
        ]

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        temb: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """(B, H, W, C) → (B, H, W, C) with 2.5D attention."""
        residual = hidden_states
        B, H, W, C = hidden_states.shape

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(B, H * W, C)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, temb, **kwargs
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(B, H, W, C)
        return hidden_states + residual
