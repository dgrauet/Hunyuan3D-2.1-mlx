"""DINOv2 image encoder in MLX.

MLX port of the DinoImageEncoder from conditioner.py.
Implements a standard ViT (DINOv2) architecture for image conditioning.

Default config: hidden_size=1024, num_heads=16, 24 layers,
patch_size=14, image_size=518 -> 1370 tokens (37*37 + 1 CLS).

Weight key mapping (HuggingFace DINOv2 checkpoint -> MLX module):
  image_encoder.embeddings.cls_token                              -> embeddings.cls_token
  image_encoder.embeddings.mask_token                             -> embeddings.mask_token
  image_encoder.embeddings.patch_embeddings.projection.weight     -> embeddings.patch_embeddings.projection.weight  (transposed OIHW->OHWI)
  image_encoder.embeddings.position_embeddings                    -> embeddings.position_embeddings
  image_encoder.encoder.layer.N.attention.attention.query/key/value -> encoder.layer.N.attention.attention.query/key/value
  image_encoder.encoder.layer.N.attention.output.dense            -> encoder.layer.N.attention.output.dense
  image_encoder.encoder.layer.N.layer_scale1/2.lambda1            -> encoder.layer.N.layer_scale1/2.lambda1
  image_encoder.encoder.layer.N.norm1/2                           -> encoder.layer.N.norm1/2
  image_encoder.encoder.layer.N.mlp.fc1/fc2                       -> encoder.layer.N.mlp.fc1/fc2
  image_encoder.layernorm                                         -> layernorm
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PatchEmbeddings(nn.Module):
    """2D image to patch embedding using Conv2d.

    MLX Conv2d uses channels-last layout: (B, H, W, C).
    Attribute name matches HF: ``patch_embeddings.projection``.

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


class Embeddings(nn.Module):
    """DINOv2 embeddings: patch + CLS + position.

    Attribute names match HF DINOv2:
      - cls_token, mask_token, patch_embeddings, position_embeddings

    Args:
        image_size: Input image size.
        patch_size: Patch size.
        in_channels: Number of input channels.
        hidden_size: Embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 518,
        patch_size: int = 14,
        in_channels: int = 3,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            image_size, patch_size, in_channels, hidden_size
        )
        num_patches = self.patch_embeddings.num_patches
        self.cls_token = mx.zeros((1, 1, hidden_size))
        self.mask_token = mx.zeros((1, hidden_size))
        self.position_embeddings = mx.zeros((1, num_patches + 1, hidden_size))
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array) -> mx.array:
        B = x.shape[0]
        patches = self.patch_embeddings(x)
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.hidden_size))
        x = mx.concatenate([cls_tokens, patches], axis=1)
        x = x + self.position_embeddings
        return x


class DINOv2SelfAttention(nn.Module):
    """DINOv2 self-attention with separate Q, K, V projections.

    Attribute names match HF: ``attention.query``, ``attention.key``,
    ``attention.value``.  Wrapped inside a parent that adds ``output.dense``.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projections.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)

    def __call__(self, x: mx.array):
        B, N, C = x.shape
        q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        return x


class DINOv2AttentionOutput(nn.Module):
    """Output projection for DINOv2 attention.

    Attribute name matches HF: ``output.dense``.

    Args:
        dim: Hidden dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dense = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense(x)


class DINOv2Attention(nn.Module):
    """Full DINOv2 attention block: self-attention + output projection.

    Attribute hierarchy matches HF:
      ``attention.attention.query/key/value`` and ``attention.output.dense``.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projections.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.attention = DINOv2SelfAttention(dim, num_heads, qkv_bias)
        self.output = DINOv2AttentionOutput(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output(self.attention(x))


class LayerScale(nn.Module):
    """Per-channel learnable scale (DINOv2 LayerScale).

    Stores a single parameter ``lambda1`` matching the HF checkpoint key
    ``layer_scale1.lambda1`` / ``layer_scale2.lambda1``.

    Args:
        dim: Hidden dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.lambda1 = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.lambda1


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
    """Single DINOv2 transformer layer (pre-norm) with LayerScale.

    LN -> Attention -> LayerScale -> Residual -> LN -> MLP -> LayerScale -> Residual

    Attribute names match HF checkpoint:
      ``attention`` (DINOv2Attention), ``layer_scale1``, ``layer_scale2``,
      ``norm1``, ``norm2``, ``mlp``.

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
        self.attention = DINOv2Attention(dim, num_heads, qkv_bias)
        self.layer_scale1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DINOv2MLP(dim, mlp_ratio)
        self.layer_scale2 = LayerScale(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder: stack of transformer layers.

    Attribute name ``layer`` (list) matches HF ``encoder.layer.N``.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Whether to use bias in QKV.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.layer = [
            DINOv2Layer(hidden_size, num_heads, mlp_ratio, qkv_bias)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for blk in self.layer:
            x = blk(x)
        return x


class ImageEncoder(nn.Module):
    """DINOv2-based image encoder with preprocessing.

    Module hierarchy matches HF DINOv2 checkpoint key structure:
      - ``embeddings`` (cls_token, mask_token, patch_embeddings, position_embeddings)
      - ``encoder.layer.N`` (transformer blocks)
      - ``layernorm`` (final layer norm)

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
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.use_cls_token = use_cls_token
        self.num_patches = (image_size // patch_size) ** 2
        if use_cls_token:
            self.num_patches += 1

        # Attribute names match HF DINOv2 checkpoint
        self.embeddings = Embeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            hidden_size=hidden_size,
        )
        self.encoder = DINOv2Encoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

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
        x = self.embeddings(image)
        x = self.encoder(x)
        x = self.layernorm(x)

        if not self.use_cls_token:
            x = x[:, 1:, :]

        return x

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

        Handles key remapping from HF DINOv2 checkpoint format:
        - Strips ``image_encoder.`` prefix
        - Transposes Conv2d weight from PyTorch OIHW to MLX OHWI

        Args:
            weights_path: Path to image_encoder.safetensors.
            config: Dict with model config (image_size, hidden_size, etc.).

        Returns:
            Loaded ImageEncoder.
        """
        model = cls(**config)
        raw_weights = mx.load(weights_path)

        remapped = {}
        for key, value in raw_weights.items():
            # Strip top-level prefix
            k = key
            if k.startswith("image_encoder."):
                k = k[len("image_encoder."):]

            # Transpose Conv2d patch embedding weight: PyTorch (O,I,H,W) -> MLX (O,H,W,I)
            if k == "embeddings.patch_embeddings.projection.weight" and value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)

            remapped[k] = value

        model.load_weights(list(remapped.items()))
        # Force materialization of lazy parameters
        mx.eval(model.parameters())  # noqa: S307 - mx.eval is MLX's standard lazy evaluation trigger, not Python's eval()
        return model
