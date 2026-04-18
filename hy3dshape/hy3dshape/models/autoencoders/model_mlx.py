"""ShapeVAE decoder in MLX.

MLX port of ShapeVAE from model.py and attention_blocks.py.
Decodes latent vectors to SDF values for marching cubes mesh extraction.
"""

import math
from typing import Optional, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_arsenal.encoding import FourierEmbedder


# --------------------------------------------------------------------------- #
# Self-Attention with fused QKV (matching PyTorch in_proj_weight/bias layout)
# --------------------------------------------------------------------------- #


class _QKNormGroup(nn.Module):
    """Container for q_norm and k_norm to match checkpoint key hierarchy.

    Checkpoint stores QK norms under ``attn.attention.q_norm`` /
    ``attn.attention.k_norm``, so this module is stored as ``self.attention``
    inside ``SelfAttention``.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)


class SelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection.

    Uses a single linear layer for Q, K, V (c_qkv: width -> 3*width).
    Checkpoint has no bias on c_qkv.

    QK norms are stored under ``self.attention`` sub-module to match
    checkpoint key hierarchy (``attn.attention.q_norm``).

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
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim ** -0.5

        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)

        # Store under ``attention`` to match checkpoint key ``attn.attention.q_norm``
        self.attention = _QKNormGroup(self.head_dim) if qk_norm else None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.c_qkv(x)
        qkv = qkv.reshape(B, N, self.heads, 3 * self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=-1)

        if self.attention is not None:
            q = self.attention.q_norm(q)
            k = self.attention.k_norm(k)

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
        qkv_bias: bool = False,
        qk_norm: bool = True,
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


class _CrossAttnProjections(nn.Module):
    """Container for cross-attention projections to match checkpoint hierarchy.

    Checkpoint stores projections under ``cross_attn_decoder.attn.c_q``,
    ``cross_attn_decoder.attn.c_kv``, ``cross_attn_decoder.attn.c_proj``,
    and ``cross_attn_decoder.attn.attention.q_norm/k_norm``.

    Args:
        width: Query dimension.
        data_width: Data (key/value) dimension.
        heads: Number of attention heads.
        qkv_bias: Whether to use bias on c_q and c_kv.
        qk_norm: Whether to apply QK normalization.
    """

    def __init__(
        self,
        width: int,
        data_width: int,
        heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = _QKNormGroup(width // heads) if qk_norm else None


class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries attend to latent features.

    Projections are grouped under ``self.attn`` to match checkpoint key
    hierarchy (``cross_attn_decoder.attn.c_q``, etc.).

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
        qkv_bias: bool = False,
        qk_norm: bool = True,
    ):
        super().__init__()
        data_width = data_width or width
        self.head_dim = width // heads
        self.heads = heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention projections grouped under ``attn`` for key matching
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)
        self.ln_2 = nn.LayerNorm(data_width, eps=1e-6)
        self.attn = _CrossAttnProjections(width, data_width, heads, qkv_bias, qk_norm)

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
        q = self.attn.c_q(self.ln_1(x))
        kv = self.attn.c_kv(self.ln_2(data))
        kv = kv.reshape(B, Nkv, self.heads, 2 * self.head_dim)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, Nq, self.heads, self.head_dim)
        if self.attn.attention is not None:
            q = self.attn.attention.q_norm(q)
            k = self.attn.attention.k_norm(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, Nq, -1)
        x = x + self.attn.c_proj(out)

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
        qkv_bias: bool = False,
        qk_norm: bool = True,
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
        include_pi: bool = False,
        qkv_bias: bool = False,
        qk_norm: bool = True,
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

        # Encoder output projection (not used during decode, but weights
        # exist in the checkpoint so we need a matching parameter slot)
        self.pre_kl = nn.Linear(width, embed_dim * 2)

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

    def _query_sdf_volume(
        self,
        xyz_samples: np.ndarray,
        features: mx.array,
        num_chunks: int = 10000,
    ) -> np.ndarray:
        """Query SDF values at a set of 3D points in chunks.

        Args:
            xyz_samples: (N, 3) query coordinates as numpy array.
            features: (B, num_latents, width) decoded features.
            num_chunks: Batch size for SDF queries.

        Returns:
            (N,) SDF values as float32 numpy array.
        """
        all_logits = []
        total_points = xyz_samples.shape[0]
        xyz_mx = mx.array(xyz_samples)
        for start in range(0, total_points, num_chunks):
            chunk = xyz_mx[start : start + num_chunks]
            chunk = chunk[None, :, :]  # (1, chunk_size, 3)
            chunk = chunk.astype(features.dtype)
            logits = self.query_sdf(chunk, features)
            _force_eval(logits)
            all_logits.append(np.array(logits[0, :, 0]).astype(np.float32))
        return np.concatenate(all_logits, axis=0)

    def _extract_near_surface_mask(self, volume: np.ndarray, mc_level: float) -> np.ndarray:
        """Find voxels near the iso-surface by checking sign changes with neighbors.

        Matches the PyTorch ``extract_near_surface_volume_fn``.

        Args:
            volume: 3D SDF grid (D, D, D).
            mc_level: Iso-level offset.

        Returns:
            Binary mask (D, D, D) of int32 where near-surface voxels are 1.
        """
        val = volume + mc_level
        valid_mask = (val > -9000).astype(np.int32)

        sign = np.sign(val.astype(np.float32))

        # Check each of 6 neighbors for sign change
        same_sign = np.ones(val.shape, dtype=bool)
        for axis in range(3):
            for shift in [1, -1]:
                neighbor = np.roll(val, -shift, axis=axis).astype(np.float32)
                # Replicate padding: overwrite the rolled boundary with the original
                slc_src = [slice(None)] * 3
                slc_dst = [slice(None)] * 3
                if shift > 0:
                    slc_src[axis] = slice(-1, None)
                    slc_dst[axis] = slice(-1, None)
                else:
                    slc_src[axis] = slice(0, 1)
                    slc_dst[axis] = slice(0, 1)
                neighbor[tuple(slc_dst)] = val[tuple(slc_src)].astype(np.float32)
                # Replace invalid neighbors with original value
                invalid = neighbor <= -9000
                neighbor[invalid] = val[invalid].astype(np.float32)
                same_sign &= (np.sign(neighbor) == sign)

        mask = (~same_sign).astype(np.int32)
        return mask * valid_mask

    def _dilate_3d(self, volume: np.ndarray) -> np.ndarray:
        """3x3x3 dilation via scipy (matches PyTorch Conv3d with all-ones kernel).

        Args:
            volume: 3D array.

        Returns:
            Dilated 3D array.
        """
        from scipy.ndimage import maximum_filter
        return maximum_filter(volume.astype(np.float64), size=3).astype(volume.dtype)

    def decode_to_mesh(
        self,
        latents: mx.array,
        bounds: float = 1.01,
        octree_resolution: int = 256,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        min_resolution: int = 63,
    ):
        """Decode latents to a trimesh.Trimesh via hierarchical volume decoding.

        Uses coarse-to-fine multi-resolution decoding (matching PyTorch
        ``HierarchicalVolumeDecoding``) to find near-surface regions first,
        then refines only those regions at higher resolutions. This avoids
        grid-pattern noise that the vanilla single-resolution approach produces.

        Args:
            latents: (B, num_latents, embed_dim) latent vectors.
            bounds: Bounding box extent.
            octree_resolution: Final grid resolution for marching cubes.
            num_chunks: Batch size for SDF queries.
            mc_level: Marching cubes iso-level.
            min_resolution: Minimum resolution for coarsest level.

        Returns:
            trimesh.Trimesh mesh object.
        """
        import trimesh
        from skimage.measure import marching_cubes

        # Decode latents to features
        features = self.decode_latents(latents)
        _force_eval(features)

        # Compute bounding box
        if isinstance(bounds, (int, float)):
            bounds = float(bounds)
            bbox_min = np.array([-bounds, -bounds, -bounds], dtype=np.float32)
            bbox_max = np.array([bounds, bounds, bounds], dtype=np.float32)
        bbox_size = bbox_max - bbox_min

        # Build resolution pyramid (matching PyTorch HierarchicalVolumeDecoding)
        resolutions = []
        r = octree_resolution
        if r < min_resolution:
            resolutions.append(r)
        while r >= min_resolution:
            resolutions.append(r)
            r = r // 2
        resolutions.reverse()

        # --- Coarsest level: query entire grid ---
        coarse_res = resolutions[0]
        xyz_samples, grid_size, _ = generate_dense_grid_points(
            bbox_min, bbox_max, coarse_res, indexing="ij"
        )
        xyz_flat = xyz_samples.reshape(-1, 3)

        print(f"Hierarchical Volume Decoding [r{coarse_res + 1}]: {xyz_flat.shape[0]} points")
        sdf_vals = self._query_sdf_volume(xyz_flat, features, num_chunks)
        grid_logits = sdf_vals.reshape(grid_size)

        # --- Refine at each higher resolution ---
        for level_idx, res_now in enumerate(resolutions[1:]):
            grid_size_now = np.array([res_now + 1] * 3)
            resolution_step = bbox_size / res_now

            # Initialize next level with -10000 (unqueried sentinel)
            next_logits = np.full(tuple(grid_size_now), -10000.0, dtype=np.float32)

            # Find near-surface voxels at current level
            curr_mask = self._extract_near_surface_mask(grid_logits, mc_level)
            # Also include voxels with small absolute SDF
            curr_mask = curr_mask + (np.abs(grid_logits) < 0.95).astype(np.int32)

            # Dilation: expand for intermediate levels, not for final
            if res_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1

            for _ in range(expand_num):
                curr_mask = self._dilate_3d(curr_mask)

            # Map coarse voxels to fine grid (2x upscale)
            cidx = np.where(curr_mask > 0)
            next_index = np.zeros(tuple(grid_size_now), dtype=np.float32)
            # Clamp indices to stay within bounds of the next grid
            fine_x = np.clip(cidx[0] * 2, 0, grid_size_now[0] - 1)
            fine_y = np.clip(cidx[1] * 2, 0, grid_size_now[1] - 1)
            fine_z = np.clip(cidx[2] * 2, 0, grid_size_now[2] - 1)
            next_index[fine_x, fine_y, fine_z] = 1

            # Dilate the seed points to cover neighborhood
            for _ in range(2 - expand_num):
                next_index = self._dilate_3d(next_index)

            # Get coordinates of points to query
            nidx = np.where(next_index > 0)
            next_points = np.stack(nidx, axis=1).astype(np.float32)
            next_points = next_points * resolution_step + bbox_min

            print(f"Hierarchical Volume Decoding [r{res_now + 1}]: {next_points.shape[0]} points "
                  f"(of {np.prod(grid_size_now)} total)")

            sdf_vals = self._query_sdf_volume(next_points, features, num_chunks)
            next_logits[nidx] = sdf_vals
            grid_logits = next_logits

        # Replace sentinel values with NaN (skimage marching_cubes handles NaN)
        grid_logits[grid_logits == -10000.0] = float('nan')

        # Marching cubes (matching original: skimage with lewiner method)
        vertices, faces, _, _ = marching_cubes(
            grid_logits, level=mc_level, method="lewiner"
        )

        # Scale vertices to bounding box coordinates
        # grid_size = [R+1, R+1, R+1] matching the original _compute_box_stat
        grid_size_arr = np.array([octree_resolution + 1] * 3, dtype=np.float32)
        vertices = vertices / grid_size_arr * bbox_size + bbox_min

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh

    @classmethod
    def from_pretrained(cls, weights_path: str, config: dict) -> "ShapeVAEDecoder":
        """Load from safetensors weights.

        Handles key remapping:
        - Strips ``vae.`` prefix
        - Skips encoder weights (``encoder.*``) not needed for decoding

        Args:
            weights_path: Path to vae.safetensors.
            config: Dict with model config.

        Returns:
            Loaded ShapeVAEDecoder.
        """
        model = cls(**config)
        raw_weights = mx.load(weights_path)

        remapped = {}
        for key, value in raw_weights.items():
            k = key
            # Strip top-level prefix
            if k.startswith("vae."):
                k = k[len("vae."):]

            # Skip encoder weights (not needed for decode-only)
            if k.startswith("encoder."):
                continue

            remapped[k] = value

        # strict=False: fourier_embedder.frequencies is computed, not in checkpoint
        model.load_weights(list(remapped.items()), strict=False)
        # Force materialization of lazy parameters
        _force_eval(model.parameters())
        return model


def _force_eval(*args):
    """Trigger MLX lazy evaluation. This is the standard MLX pattern
    for materializing computation graphs."""
    mx.eval(*args)  # noqa: S307 - mx.eval is MLX's computation trigger, not Python's eval
