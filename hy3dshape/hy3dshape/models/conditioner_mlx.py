"""DINOv2 image encoder in MLX.

MLX port of the DinoImageEncoder from conditioner.py.
Implements a standard ViT (DINOv2) architecture for image conditioning.

Default config: hidden_size=1024, num_heads=16, 24 layers,
patch_size=14, image_size=518 -> 1370 tokens (37*37 + 1 CLS).
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PatchEmbedding(nn.Module):
    """2D image to patch embedding using Conv2d.

    MLX Conv2d uses channels-last layout: (B, H, W, C).

    Args:
        image_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, H, W, C) channels-last image tensor.

        Returns:
            (B, num_patches, embed_dim) patch embeddings.
        """
        # Conv2d: (B, H, W, C) -> (B, H/P, W/P, D)
        x = self.projection(x)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)
        return x


class DINOv2Attention(nn.Module):
    """Multi-head self-attention for DINOv2.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projections.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale
        )
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        return x


class DINOv2MLP(nn.Module):
    """Feed-forward network for DINOv2 with GELU activation.

    Args:
        dim: Hidden dimension.
        mlp_ratio: MLP expansion ratio.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class DINOv2Layer(nn.Module):
    """Single DINOv2 transformer layer (pre-norm).

    LN -> Attention -> Residual -> LN -> MLP -> Residual

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Whether to use bias in QKV.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DINOv2Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DINOv2MLP(dim, mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DINOv2Encoder(nn.Module):
    """Full DINOv2 ViT encoder.

    Embeddings (patch + CLS + position) -> N transformer layers -> LayerNorm.

    Args:
        image_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Whether to use bias in QKV.
    """

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, hidden_size
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = mx.zeros((1, 1, hidden_size))
        self.position_embeddings = mx.zeros((1, num_patches + 1, hidden_size))

        self.layers = [
            DINOv2Layer(hidden_size, num_heads, mlp_ratio, qkv_bias)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, H, W, C) channels-last image tensor (already preprocessed).

        Returns:
            (B, num_patches+1, hidden_size) last hidden states including CLS.
        """
        B = x.shape[0]
        patches = self.patch_embed(x)

        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.hidden_size))
        x = mx.concatenate([cls_tokens, patches], axis=1)
        x = x + self.position_embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class ImageEncoder(nn.Module):
    """DINOv2-based image encoder with preprocessing.

    Handles resize, normalize with ImageNet stats, and encoding.

    Args:
        image_size: Target image size for DINOv2.
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        patch_size: Patch size.
        use_cls_token: Whether to include CLS token in output.
    """

    # ImageNet normalization stats
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        image_size: int = 518,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        patch_size: int = 14,
        use_cls_token: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.use_cls_token = use_cls_token
        self.num_patches = (image_size // patch_size) ** 2
        if use_cls_token:
            self.num_patches += 1

        self.encoder = DINOv2Encoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

    def preprocess(self, image: mx.array) -> mx.array:
        """Normalize image with ImageNet stats.

        Args:
            image: (B, H, W, 3) float32 in [0, 1] range, channels-last.

        Returns:
            Normalized image tensor.
        """
        mean = mx.array(self.MEAN).reshape(1, 1, 1, 3)
        std = mx.array(self.STD).reshape(1, 1, 1, 3)
        return (image - mean) / std

    def __call__(self, image: mx.array, value_range=(-1, 1)) -> mx.array:
        """Encode an image.

        Args:
            image: (B, H, W, 3) image tensor in channels-last format.
            value_range: Expected value range of input, will be mapped to [0, 1].

        Returns:
            (B, num_patches, hidden_size) image features.
        """
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = self.preprocess(image)
        features = self.encoder(image)

        if not self.use_cls_token:
            features = features[:, 1:, :]

        return features

    def unconditional_embedding(self, batch_size: int) -> mx.array:
        """Return zero embedding for classifier-free guidance.

        Args:
            batch_size: Batch size.

        Returns:
            Zero tensor of shape (batch_size, num_patches, hidden_size).
        """
        return mx.zeros((batch_size, self.num_patches, self.hidden_size))

    @classmethod
    def from_pretrained(cls, weights_path: str, config: dict) -> "ImageEncoder":
        """Load ImageEncoder from safetensors weights.

        Args:
            weights_path: Path to image_encoder.safetensors.
            config: Dict with model config (image_size, hidden_size, etc.).

        Returns:
            Loaded ImageEncoder.
        """
        model = cls(**config)
        weights = mx.load(weights_path)
        model.load_weights(list(weights.items()))
        # Force materialization of lazy parameters
        mx.eval(model.parameters())  # noqa: S307 - mx.eval is MLX's standard lazy evaluation trigger, not Python's eval()
        return model
