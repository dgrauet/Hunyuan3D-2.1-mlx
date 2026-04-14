"""MLX attention processors for HunyuanPaintPBR UNet.

Ports: RotaryEmbedding, AttnCore, SelfAttnProcessor2_0,
PoseRoPEAttnProcessor2_0, RefAttnProcessor2_0.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------


class RotaryEmbedding:
    """1D and 3D rotary position embeddings for attention."""

    @staticmethod
    def get_1d_rotary_pos_embed(
        dim: int, pos: mx.array, theta: float = 10000.0
    ) -> Tuple[mx.array, mx.array]:
        """Compute 1D rotary embeddings.

        Args:
            dim: Embedding dimension (head_dim).
            pos: (L,) position indices.
            theta: Base frequency.

        Returns:
            (cos, sin) each of shape (L, dim).
        """
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        # (dim//2,)
        angles = pos[:, None].astype(mx.float32) * freqs[None, :]  # (L, dim//2)
        cos = mx.cos(angles)
        sin = mx.sin(angles)
        # Repeat-interleave to full dim: [c0,c0,c1,c1,...]
        cos = mx.repeat(cos, 2, axis=-1)  # (L, dim)
        sin = mx.repeat(sin, 2, axis=-1)
        return cos, sin

    @staticmethod
    def get_3d_rotary_pos_embed(
        position: mx.array, embed_dim: int = 96, voxel_resolution: int = 128
    ) -> Tuple[mx.array, mx.array]:
        """Compute 3D rotary embeddings from voxel coordinates.

        Args:
            position: (B, L, 3) int voxel indices.
            embed_dim: Total embedding dimension (split across XYZ).
            voxel_resolution: Max voxel coordinate value.

        Returns:
            (cos, sin) each of shape (B, L, embed_dim).
        """
        dim_xy = embed_dim // 8 * 3  # 36 for embed_dim=96
        dim_z = embed_dim // 8 * 2   # 24

        grid = mx.arange(voxel_resolution).astype(mx.float32)
        xy_cos, xy_sin = RotaryEmbedding.get_1d_rotary_pos_embed(dim_xy, grid)
        z_cos, z_sin = RotaryEmbedding.get_1d_rotary_pos_embed(dim_z, grid)
        # Each: (voxel_resolution, dim_xy or dim_z)

        # Gather per-position embeddings
        px = position[..., 0]  # (B, L)
        py = position[..., 1]
        pz = position[..., 2]

        cos = mx.concatenate([xy_cos[px], xy_cos[py], z_cos[pz]], axis=-1)
        sin = mx.concatenate([xy_sin[px], xy_sin[py], z_sin[pz]], axis=-1)
        return cos, sin  # (B, L, embed_dim)

    @staticmethod
    def apply_rotary_emb(
        x: mx.array, freqs_cis: Tuple[mx.array, mx.array]
    ) -> mx.array:
        """Apply rotary embeddings to a tensor.

        Args:
            x: (B, num_heads, L, head_dim)
            freqs_cis: (cos, sin) each broadcastable to x.

        Returns:
            Rotated tensor, same shape as x.
        """
        cos, sin = freqs_cis
        # cos/sin: (B, L, head_dim) or (L, head_dim) — broadcast to (B, 1, L, head_dim)
        if cos.ndim == 2:
            cos = cos[None, None, :, :]  # (1, 1, L, D)
            sin = sin[None, None, :, :]
        elif cos.ndim == 3:
            cos = cos[:, None, :, :]     # (B, 1, L, D)
            sin = sin[:, None, :, :]

        # Rotate pairs: (x0, x1) -> (x0*cos - x1*sin, x1*cos + x0*sin)
        x_even = x[..., 0::2]  # (B, H, L, D//2)
        x_odd = x[..., 1::2]
        cos_half = cos[..., 0::2]
        sin_half = sin[..., 0::2]

        out_even = x_even * cos_half - x_odd * sin_half
        out_odd = x_odd * cos_half + x_even * sin_half

        # Interleave back: stack on last dim then flatten
        return mx.stack([out_even, out_odd], axis=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# ImageProjModel
# ---------------------------------------------------------------------------


class ImageProjModel(nn.Module):
    """Projects DINO features to cross-attention dimension.

    Input (B, N*L, clip_dim) → Output (B, N*L*extra_tokens, cross_dim).
    """

    def __init__(
        self,
        cross_attention_dim: int = 768,
        clip_embeddings_dim: int = 1536,
        clip_extra_context_tokens: int = 4,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(
            clip_embeddings_dim,
            clip_extra_context_tokens * cross_attention_dim,
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def __call__(self, image_embeds: mx.array) -> mx.array:
        """
        Args:
            image_embeds: (B, N*L, clip_dim) or (B*N*L, clip_dim)
        Returns:
            (B, N*L*extra_tokens, cross_dim)
        """
        has_batch = image_embeds.ndim == 3
        if has_batch:
            b, nl, _ = image_embeds.shape
            embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
        else:
            embeds = image_embeds

        tokens = self.proj(embeds)  # (B*N*L, extra * cross_dim)
        tokens = tokens.reshape(-1, self.clip_extra_context_tokens,
                                self.cross_attention_dim)
        tokens = self.norm(tokens)  # (B*N*L, extra, cross_dim)

        if has_batch:
            tokens = tokens.reshape(b, nl * self.clip_extra_context_tokens,
                                    self.cross_attention_dim)
        return tokens


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    attn_mask: Optional[mx.array] = None,
    scale: Optional[float] = None,
) -> mx.array:
    """Scaled dot-product attention.

    Args:
        query: (B, H, L_q, D)
        key:   (B, H, L_k, D)
        value: (B, H, L_v, D)  (L_v == L_k)
        attn_mask: optional (B, H, L_q, L_k) or broadcastable.
        scale: scaling factor (default 1/sqrt(D)).

    Returns:
        (B, H, L_q, D)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    scores = (query @ key.transpose(0, 1, 3, 2)) * scale  # (B, H, L_q, L_k)

    if attn_mask is not None:
        scores = scores + attn_mask

    weights = mx.softmax(scores, axis=-1)
    return weights @ value


# ---------------------------------------------------------------------------
# Attention Utilities
# ---------------------------------------------------------------------------


def reshape_for_attention(
    x: mx.array, num_heads: int
) -> mx.array:
    """Reshape (B, L, C) → (B, num_heads, L, head_dim)."""
    B, L, C = x.shape
    head_dim = C // num_heads
    return x.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)


def reshape_from_attention(
    x: mx.array,
) -> mx.array:
    """Reshape (B, num_heads, L, head_dim) → (B, L, C)."""
    B, H, L, D = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, L, H * D)


# ---------------------------------------------------------------------------
# Attention Processors
# ---------------------------------------------------------------------------


class LinearProjection(nn.Module):
    """Linear layer for Q/K/V projections."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


class SelfAttnProcessor(nn.Module):
    """Material-dimension self-attention.

    Maintains separate Q/K/V/out projections per PBR material.
    """

    def __init__(
        self,
        query_dim: int = 768,
        num_heads: int = 8,
        pbr_settings: tuple = ("albedo", "mr"),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.pbr_settings = pbr_settings

        # Per-material projections (albedo uses the base attn1 projections,
        # additional materials get their own)
        self.extra_qkv = {}
        for token in pbr_settings[1:]:  # skip first (uses base)
            self.extra_qkv[token] = {
                "to_q": nn.Linear(query_dim, query_dim),
                "to_k": nn.Linear(query_dim, query_dim),
                "to_v": nn.Linear(query_dim, query_dim),
                "to_out": nn.Linear(query_dim, query_dim),
            }
            # Register as submodules
            for name, mod in self.extra_qkv[token].items():
                setattr(self, f"{name}_{token}", mod)

    def __call__(
        self,
        hidden_states: mx.array,
        to_q: nn.Linear,
        to_k: nn.Linear,
        to_v: nn.Linear,
        to_out: nn.Linear,
        n_views: int = 6,
    ) -> mx.array:
        """
        Args:
            hidden_states: (B, N_pbr, N_gen, L, C) — batched per material.
            to_q/k/v/out: base projections (used for first material).
            n_views: number of generated views.

        Returns:
            (B, N_pbr, N_gen, L, C)
        """
        B, N_pbr, N_gen, L, C = hidden_states.shape
        results = []

        for i, token in enumerate(self.pbr_settings):
            hs = hidden_states[:, i]  # (B, N_gen, L, C)
            hs = hs.reshape(B * N_gen, L, C)

            if i == 0:
                q, k, v, out_proj = to_q, to_k, to_v, to_out
            else:
                q = getattr(self, f"to_q_{token}")
                k = getattr(self, f"to_k_{token}")
                v = getattr(self, f"to_v_{token}")
                out_proj = getattr(self, f"to_out_{token}")

            query = reshape_for_attention(q(hs), self.num_heads)
            key = reshape_for_attention(k(hs), self.num_heads)
            value = reshape_for_attention(v(hs), self.num_heads)

            attn_out = scaled_dot_product_attention(query, key, value)
            attn_out = reshape_from_attention(attn_out)
            attn_out = out_proj(attn_out)

            results.append(attn_out.reshape(B, 1, N_gen, L, C))

        return mx.concatenate(results, axis=1)


class PoseRoPEAttnProcessor(nn.Module):
    """Multiview self-attention with 3D rotary position embeddings."""

    def __init__(self, query_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(query_dim, query_dim)
        self.to_v = nn.Linear(query_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        position_indices: Optional[dict] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: (B*N_pbr, N_gen*L, C) — views concatenated.
            position_indices: dict with voxel_indices and voxel_resolution.

        Returns:
            (B*N_pbr, N_gen*L, C)
        """
        B, NL, C = hidden_states.shape

        query = reshape_for_attention(self.to_q(hidden_states), self.num_heads)
        key = reshape_for_attention(self.to_k(hidden_states), self.num_heads)
        value = reshape_for_attention(self.to_v(hidden_states), self.num_heads)

        if position_indices is not None:
            vi = position_indices.get("voxel_indices")
            vr = position_indices.get("voxel_resolution", 128)
            if vi is not None:
                rope = RotaryEmbedding.get_3d_rotary_pos_embed(
                    vi, self.head_dim, vr
                )
                query = RotaryEmbedding.apply_rotary_emb(query, rope)
                key = RotaryEmbedding.apply_rotary_emb(key, rope)

        attn_out = scaled_dot_product_attention(query, key, value)
        attn_out = reshape_from_attention(attn_out)
        return self.to_out(attn_out)


class RefAttnProcessor(nn.Module):
    """Reference cross-attention with shared Q/K, material-specific V."""

    def __init__(
        self,
        query_dim: int = 768,
        num_heads: int = 8,
        pbr_settings: tuple = ("albedo", "mr"),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.pbr_settings = pbr_settings
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(query_dim, query_dim)

        # Per-material value + output projections
        for token in pbr_settings:
            setattr(self, f"to_v_{token}", nn.Linear(query_dim, query_dim))
            setattr(self, f"to_out_{token}", nn.Linear(query_dim, query_dim))

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        """
        Args:
            hidden_states: (B, N_gen*L, C) — generated features (query).
            encoder_hidden_states: (B, N_ref*L, C) — reference features (key/value).

        Returns:
            (B, N_pbr, N_gen*L, C) — per-material outputs.
        """
        query = reshape_for_attention(self.to_q(hidden_states), self.num_heads)
        key = reshape_for_attention(self.to_k(encoder_hidden_states), self.num_heads)

        results = []
        for token in self.pbr_settings:
            to_v = getattr(self, f"to_v_{token}")
            to_out = getattr(self, f"to_out_{token}")
            value = reshape_for_attention(to_v(encoder_hidden_states), self.num_heads)

            attn_out = scaled_dot_product_attention(query, key, value)
            attn_out = reshape_from_attention(attn_out)
            results.append(to_out(attn_out))

        return mx.stack(results, axis=1)  # (B, N_pbr, N_gen*L, C)
