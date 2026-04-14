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
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
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
        self.norm = nn.LayerNorm(embed_dim)

        self.blocks = [
            ViTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ]

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Extract patch features.

        Args:
            pixel_values: (B, H, W, 3) float32 normalized images.

        Returns:
            (B, num_patches + 1, embed_dim) features including CLS token.
        """
        B = pixel_values.shape[0]

        x = self.patch_embed(pixel_values)  # (B, N, D)

        cls = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls, x], axis=1)  # (B, N+1, D)
        x = x + self.pos_embed

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
