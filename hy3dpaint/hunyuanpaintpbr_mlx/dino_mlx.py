"""DINOv2 Vision Transformer for MLX.

Extracts patch embeddings for conditioning the diffusion model.
Architecture: ViT-giant (1536-dim, 40 heads, 40 layers, patch_size=14).
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PatchEmbedding(nn.Module):
    """Image to patch embedding via convolution."""

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1536,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        """(B, H, W, 3) → (B, num_patches, embed_dim)"""
        x = self.proj(x)  # (B, H/P, W/P, D)
        B = x.shape[0]
        return x.reshape(B, -1, x.shape[-1])  # (B, N, D)


class ViTAttention(nn.Module):
    """Multi-head self-attention for ViT."""

    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class ViTMLP(nn.Module):
    """SwiGLU MLP for DINOv2.

    fc1 (weights_in) projects to 2*hidden_dim, splits into gate + value.
    fc2 (weights_out) projects gated output back to dim.
    """

    def __init__(self, dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim <= 0:
            hidden_dim = int(dim * 4 * 2 / 3)
        hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.fc1(x)
        gate, value = mx.split(h, 2, axis=-1)
        return self.fc2(nn.silu(gate) * value)


class ViTBlock(nn.Module):
    """Transformer block with optional layer scaling (DINOv2)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # HF DINOv2 uses layer_norm_eps=1e-6; MLX default is 1e-5. Over 40
        # blocks the 10x eps discrepancy drives the output to diverge
        # super-exponentially from HF (measured: block 0 diff 0.04 ->
        # block 39 diff 435 before the final LayerNorm clips it).
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = ViTAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        # DINOv2 SwiGLU: hidden = dim * 4 * 2/3 (gated), default 0 computes this
        self.mlp = ViTMLP(dim)
        self.layer_scale1 = {"lambda1": mx.ones((dim,))}
        self.layer_scale2 = {"lambda1": mx.ones((dim,))}

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.layer_scale1["lambda1"] * self.attn(self.norm1(x))
        x = x + self.layer_scale2["lambda1"] * self.mlp(self.norm2(x))
        return x


class DINOv2MLX(nn.Module):
    """DINOv2 ViT-giant feature extractor.

    Frozen model — used only for inference (no gradients).

    Config for dinov2-giant:
        embed_dim=1536, num_heads=24, depth=40, mlp_ratio=4.0,
        patch_size=14, image_size=518
    """

    def __init__(
        self,
        embed_dim: int = 1536,
        num_heads: int = 24,
        depth: int = 40,
        mlp_ratio: float = 4.0,
        patch_size: int = 14,
        image_size: int = 518,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size

        self.patch_embed = PatchEmbedding(
            image_size, patch_size, 3, embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.pos_embed = mx.zeros((1, num_patches + 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = [
            ViTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ]

    def _interpolate_pos_embed(self, num_patches_in: int) -> mx.array:
        """Interpolate pos_embed to match ``num_patches_in`` patches.

        Matches HF DINOv2's ``interpolate_pos_encoding=True`` bicubic
        resizing with ``antialias=True`` — we run the resize in numpy via
        scipy so the result is numerically close to HF (MLX has no native
        bicubic upsampling). Cached so we only pay the cost once.
        """
        trained_patches = self.pos_embed.shape[1] - 1
        if num_patches_in == trained_patches:
            return self.pos_embed

        cache = getattr(self, "_pos_embed_cache", {})
        if num_patches_in in cache:
            return cache[num_patches_in]

        import math
        import torch
        import torch.nn.functional as F

        g_old = int(round(math.sqrt(trained_patches)))
        g_new = int(round(math.sqrt(num_patches_in)))
        assert g_old * g_old == trained_patches
        assert g_new * g_new == num_patches_in

        pos_np = np.asarray(self.pos_embed).astype(np.float32)
        cls_np = pos_np[:, :1, :]
        patch_np = pos_np[:, 1:, :].reshape(1, g_old, g_old, -1)  # NHWC
        # Route through PyTorch's F.interpolate(mode="bicubic",
        # antialias=True) which is what HF DINOv2 uses internally. MLX has
        # no bicubic upsampling kernel yet, so this one cross-library call
        # at load time is the easiest way to get bit-close to HF.
        pt = torch.from_numpy(patch_np).permute(0, 3, 1, 2)  # (1, D, g_old, g_old)
        pt_new = F.interpolate(
            pt, size=(g_new, g_new), mode="bicubic", antialias=True,
            align_corners=False,
        )
        patch_new = pt_new.permute(0, 2, 3, 1).numpy().reshape(1, g_new * g_new, -1)
        new_pos = mx.array(np.concatenate([cls_np, patch_new], axis=1))
        cache[num_patches_in] = new_pos
        self._pos_embed_cache = cache
        return new_pos

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Extract patch features.

        Args:
            pixel_values: (B, H, W, 3) float32 normalized images.

        Returns:
            (B, num_patches + 1, embed_dim) features including CLS token.
        """
        B, H, W, _ = pixel_values.shape

        x = self.patch_embed(pixel_values)  # (B, N, D)
        N = x.shape[1]

        cls = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls, x], axis=1)  # (B, N+1, D)

        # Interpolate the trained pos_embed to match current patch count.
        pos = self._interpolate_pos_embed(N)
        x = x + pos

        for block in self.blocks:
            x = block(x)

        return self.norm(x)  # (B, N+1, D)

    def extract_features(
        self, images: mx.array, batch_size: int = 1
    ) -> mx.array:
        """Extract and reshape features for diffusion conditioning.

        Args:
            images: (B, N_views, H, W, 3) or (B*N, H, W, 3).

        Returns:
            (batch_size, N_views * (num_patches+1), embed_dim)
        """
        if images.ndim == 5:
            B, N, H, W, C = images.shape
            images = images.reshape(B * N, H, W, C)
        else:
            N = images.shape[0] // batch_size

        features = self(images)  # (B*N, L, D)
        L = features.shape[1]
        features = features.reshape(batch_size, N * L, self.embed_dim)
        return features


def preprocess_for_dino(
    image_np: np.ndarray,
    image_size: int = 518,
) -> mx.array:
    """Preprocess a numpy image for DINOv2.

    Uses 518x518 (the native resolution the checkpoint's pos_embed is
    trained for). PT's AutoImageProcessor default is 224x224, which
    requires interpolating pos_embed and introduces numerical drift that
    amplifies super-exponentially over the 40 ViT blocks (we measured
    mean_rel ~130% on the final features vs HF at 224, vs ~2% at 518).

    ImageProjModel downstream handles the 5.3x more tokens (1370 vs 257)
    correctly — it's a per-token Linear projection. attn_dino in the
    main UNet cross-attends to whatever tokens we provide; more tokens
    give finer spatial detail for reference conditioning.

    Args:
        image_np: (H, W, 3) uint8 or float32 image.

    Returns:
        (1, image_size, image_size, 3) mx.array normalized.
    """
    from PIL import Image

    if isinstance(image_np, np.ndarray):
        if image_np.dtype == np.uint8:
            image_np = image_np.astype(np.float32) / 255.0
        img = Image.fromarray((image_np * 255).astype(np.uint8))
    else:
        img = image_np

    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    return mx.array(arr[None, ...])  # (1, H, W, 3)
