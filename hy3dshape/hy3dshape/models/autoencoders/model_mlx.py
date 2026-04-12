"""ShapeVAE decoder in MLX.

MLX port of ShapeVAE from model.py and attention_blocks.py.
Decodes latent vectors to SDF values for marching cubes mesh extraction.
"""

import math
from typing import Optional, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_ops.encoding import FourierEmbedder


# --------------------------------------------------------------------------- #
# Self-Attention with fused QKV (matching PyTorch in_proj_weight/bias layout)
# --------------------------------------------------------------------------- #


class SelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection.

    Uses a single linear layer for Q, K, V (c_qkv: width -> 3*width).

    Args:
        width: Hidden dimension.
        heads: Number of attention heads.
        qkv_bias: Whether to use bias.
        qk_norm: Whether to apply QK normalization.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5

        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)

        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.c_qkv(x)
        qkv = qkv.reshape(B, N, self.heads, 3 * self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=-1)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # (B, N, H, D) -> (B, H, N, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.c_proj(out)


class VAMLP(nn.Module):
    """MLP for the VAE transformer.

    Args:
        width: Input/output dimension.
        expand_ratio: Expansion ratio.
    """

    def __init__(self, width: int, expand_ratio: int = 4):
        super().__init__()
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(width * expand_ratio, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):
    """Self-attention + FFN with pre-norm residuals.

    Args:
        width: Hidden dimension.
        heads: Number of attention heads.
        qkv_bias: Whether to use bias.
        qk_norm: Whether to apply QK normalization.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = SelfAttention(width, heads, qkv_bias, qk_norm)
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = VAMLP(width)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Stack of ResidualAttentionBlocks.

    Args:
        n_ctx: Context length (number of latent tokens).
        width: Hidden dimension.
        layers: Number of transformer layers.
        heads: Number of attention heads.
        qkv_bias: Whether to use bias.
        qk_norm: Whether to apply QK normalization.
    """

    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = [
            ResidualAttentionBlock(width, heads, qkv_bias, qk_norm)
            for _ in range(layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.resblocks:
            x = block(x)
        return x


# --------------------------------------------------------------------------- #
# Cross-Attention Decoder (queries 3D points against latent features)
# --------------------------------------------------------------------------- #


class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries attend to latent features.

    Args:
        width: Hidden dimension for queries.
        heads: Number of attention heads.
        data_width: Dimension of the data (latent features).
        mlp_expand_ratio: Expansion ratio for MLP.
        qkv_bias: Whether to use bias.
        qk_norm: Whether to apply QK normalization.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        data_width: int = None,
        mlp_expand_ratio: int = 4,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        data_width = data_width or width
        self.head_dim = width // heads
        self.heads = heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.ln_2 = nn.LayerNorm(data_width, eps=1e-6)
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)

        self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6) if qk_norm else None

        # FFN
        self.ln_3 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = VAMLP(width, mlp_expand_ratio)

    def __call__(self, x: mx.array, data: mx.array) -> mx.array:
        """
        Args:
            x: (B, Nq, width) query embeddings.
            data: (B, Nkv, data_width) latent features.
        Returns:
            (B, Nq, width) updated queries.
        """
        B, Nq, _ = x.shape
        _, Nkv, _ = data.shape

        # Cross-attention
        q = self.c_q(self.ln_1(x))
        kv = self.c_kv(self.ln_2(data))
        kv = kv.reshape(B, Nkv, self.heads, 2 * self.head_dim)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, Nq, self.heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, Nq, -1)
        x = x + self.c_proj(out)

        # FFN
        x = x + self.mlp(self.ln_3(x))
        return x


class CrossAttentionDecoder(nn.Module):
    """Geo decoder: Fourier-encoded 3D points cross-attend to latent features.

    Args:
        fourier_embedder: FourierEmbedder instance.
        out_channels: Output channels (1 for SDF).
        num_latents: Number of latent tokens.
        width: Hidden dimension.
        heads: Number of attention heads.
        mlp_expand_ratio: MLP expansion ratio.
        downsample_ratio: Ratio for downsampling latent features.
        enable_ln_post: Whether to apply LayerNorm after cross-attention.
        qkv_bias: Whether to use bias.
        qk_norm: Whether to apply QK normalization.
    """

    def __init__(
        self,
        fourier_embedder: FourierEmbedder,
        out_channels: int = 1,
        num_latents: int = 4096,
        width: int = 768,
        heads: int = 12,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
    ):
        super().__init__()
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.enable_ln_post = enable_ln_post

        self.query_proj = nn.Linear(fourier_embedder.out_dim, width)
        if downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)

        if not enable_ln_post:
            qk_norm = False

        self.cross_attn_decoder = CrossAttentionBlock(
            width=width,
            heads=heads,
            data_width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        if enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)

    def __call__(
        self,
        queries: mx.array = None,
        query_embeddings: mx.array = None,
        latents: mx.array = None,
    ) -> mx.array:
        """
        Args:
            queries: (B, N, 3) 3D query points.
            query_embeddings: Pre-computed query embeddings (optional).
            latents: (B, num_latents, width) latent features.

        Returns:
            (B, N, out_channels) SDF predictions.
        """
        if query_embeddings is None:
            query_embeddings = self.query_proj(
                self.fourier_embedder(queries).astype(latents.dtype)
            )

        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)

        x = self.cross_attn_decoder(query_embeddings, latents)

        if self.enable_ln_post:
            x = self.ln_post(x)

        return self.output_proj(x)


# --------------------------------------------------------------------------- #
# ShapeVAE Decoder (top-level)
# --------------------------------------------------------------------------- #


def generate_dense_grid_points(bbox_min, bbox_max, octree_resolution, indexing="ij"):
    """Generate dense 3D grid points for volume decoding."""
    num_cells = octree_resolution
    x = np.linspace(bbox_min[0], bbox_max[0], num_cells + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], num_cells + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], num_cells + 1, dtype=np.float32)
    xs, ys, zs = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [num_cells + 1, num_cells + 1, num_cells + 1]
    length = bbox_max - bbox_min
    return xyz, grid_size, length


class ShapeVAEDecoder(nn.Module):
    """Top-level ShapeVAE decoder.

    Decodes latent vectors through self-attention transformer, then queries
    SDF values via cross-attention geo decoder.

    Args:
        num_latents: Number of latent tokens.
        embed_dim: Latent embedding dimension.
        width: Transformer hidden dimension.
        heads: Number of attention heads.
        num_decoder_layers: Number of self-attention layers.
        num_freqs: Number of Fourier frequencies.
        include_pi: Whether to include pi in Fourier encoding.
        qkv_bias: Whether to use bias in attention.
        qk_norm: Whether to apply QK normalization.
        geo_decoder_downsample_ratio: Downsample ratio for geo decoder.
        geo_decoder_mlp_expand_ratio: MLP expansion ratio for geo decoder.
        geo_decoder_ln_post: Whether to apply LayerNorm after geo decoder.
        scale_factor: Scale factor for latents.
        label_type: Type of label for geo decoder.
    """

    def __init__(
        self,
        num_latents: int = 4096,
        embed_dim: int = 64,
        width: int = 768,
        heads: int = 12,
        num_decoder_layers: int = 12,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        scale_factor: float = 1.0,
        label_type: str = "binary",
        # Ignored kwargs for compatibility
        num_encoder_layers: int = 8,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        drop_path_rate: float = 0.0,
        use_ln_post: bool = True,
        ckpt_path=None,
        **kwargs,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)
        self.width = width

        # Latent to transformer space
        self.post_kl = nn.Linear(embed_dim, width)

        # Self-attention transformer
        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        # Geo decoder (cross-attention SDF predictor)
        fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs,
            include_pi=include_pi,
            input_dim=3,
            include_input=True,
        )
        geo_width = width // geo_decoder_downsample_ratio
        geo_heads = heads // geo_decoder_downsample_ratio

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=geo_width,
            heads=geo_heads,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=geo_decoder_ln_post,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

    def decode_latents(self, latents: mx.array) -> mx.array:
        """Decode latents through post_kl and transformer.

        Args:
            latents: (B, num_latents, embed_dim) latent vectors.

        Returns:
            (B, num_latents, width) decoded features.
        """
        latents = latents * (1.0 / self.scale_factor)
        x = self.post_kl(latents)
        x = self.transformer(x)
        return x

    def query_sdf(self, points: mx.array, features: mx.array) -> mx.array:
        """Query SDF values at 3D points.

        Args:
            points: (B, N, 3) query coordinates.
            features: (B, num_latents, width) decoded features.

        Returns:
            (B, N, 1) SDF values.
        """
        return self.geo_decoder(queries=points, latents=features)

    def decode_to_mesh(
        self,
        latents: mx.array,
        bounds: float = 1.01,
        octree_resolution: int = 256,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
    ):
        """Decode latents to a trimesh.Trimesh via marching cubes.

        Args:
            latents: (B, num_latents, embed_dim) latent vectors.
            bounds: Bounding box extent.
            octree_resolution: Grid resolution for marching cubes.
            num_chunks: Batch size for SDF queries.
            mc_level: Marching cubes iso-level.

        Returns:
            trimesh.Trimesh mesh object.
        """
        import trimesh

        # Decode latents to features
        features = self.decode_latents(latents)
        # Force materialization of features
        _force_eval(features)

        # Generate dense grid
        bbox_min = np.array([-bounds, -bounds, -bounds])
        bbox_max = np.array([bounds, bounds, bounds])
        xyz_samples, grid_size, _ = generate_dense_grid_points(
            bbox_min, bbox_max, octree_resolution, indexing="ij"
        )
        xyz_samples = mx.array(xyz_samples.reshape(-1, 3))

        # Query SDF in chunks
        all_logits = []
        total_points = xyz_samples.shape[0]
        for start in range(0, total_points, num_chunks):
            chunk = xyz_samples[start : start + num_chunks]
            chunk = chunk[None, :, :]  # (1, chunk_size, 3)
            chunk = chunk.astype(features.dtype)
            logits = self.query_sdf(chunk, features)
            _force_eval(logits)
            all_logits.append(np.array(logits))

        grid_logits = np.concatenate(all_logits, axis=1)
        grid_logits = grid_logits.reshape(grid_size).astype(np.float32)

        # Marching cubes (CPU)
        try:
            import mcubes

            vertices, faces = mcubes.marching_cubes(grid_logits, mc_level)
        except ImportError:
            from skimage.measure import marching_cubes

            vertices, faces, _, _ = marching_cubes(grid_logits, level=mc_level)

        # Scale vertices to bounding box
        vertices = vertices / octree_resolution * (bbox_max - bbox_min) + bbox_min

        # Flip face winding to match original
        faces = faces[:, ::-1]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh

    @classmethod
    def from_pretrained(cls, weights_path: str, config: dict) -> "ShapeVAEDecoder":
        """Load from safetensors weights.

        Args:
            weights_path: Path to vae.safetensors.
            config: Dict with model config.

        Returns:
            Loaded ShapeVAEDecoder.
        """
        model = cls(**config)
        weights = mx.load(weights_path)
        model.load_weights(list(weights.items()))
        # Force materialization of lazy parameters
        _force_eval(model.parameters())
        return model


def _force_eval(*args):
    """Trigger MLX lazy evaluation. This is the standard MLX pattern
    for materializing computation graphs."""
    mx.eval(*args)  # noqa: S307 - mx.eval is MLX's computation trigger, not Python's eval
