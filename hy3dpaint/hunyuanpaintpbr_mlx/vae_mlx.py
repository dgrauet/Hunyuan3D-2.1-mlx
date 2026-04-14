"""MLX port of Stable Diffusion 2.1 AutoencoderKL (VAE).

Encodes images to latent space and decodes latents back to images.
All tensors use NHWC layout: (B, H, W, C).
"""

import math
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------


class ResnetBlock2D(nn.Module):
    """Residual block: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv + skip."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return h + residual


class AttentionBlock(nn.Module):
    """Single-head self-attention for VAE mid block.

    input -> GroupNorm -> reshape (B, H*W, C) -> Q,K,V -> attention -> proj_out -> reshape -> + input
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)
        self.channels = channels

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        B, H, W, C = x.shape

        h = self.group_norm(x)
        h = h.reshape(B, H * W, C)

        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)

        # Single-head attention: (B, L, C)
        scale = 1.0 / math.sqrt(C)
        scores = (q @ k.transpose(0, 2, 1)) * scale  # (B, L, L)
        weights = mx.softmax(scores, axis=-1)
        attn_out = weights @ v  # (B, L, C)

        attn_out = self.to_out(attn_out)
        attn_out = attn_out.reshape(B, H, W, C)

        return attn_out + residual


class Downsample2D(nn.Module):
    """Spatial downsample via stride-2 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Spatial upsample via nearest-neighbor interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder Blocks
# ---------------------------------------------------------------------------


class DownEncoderBlock2D(nn.Module):
    """Encoder block: N ResnetBlocks + optional Downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(ch_in, out_channels))

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = [Downsample2D(out_channels)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                x = ds(x)
        return x


class UpDecoderBlock2D(nn.Module):
    """Decoder block: N ResnetBlocks + optional Upsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            ch_in = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(ch_in, out_channels))

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = [Upsample2D(out_channels)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            for us in self.upsamplers:
                x = us(x)
        return x


class MidBlock2D(nn.Module):
    """Mid block: ResnetBlock -> Attention -> ResnetBlock."""

    def __init__(self, channels: int):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(channels, channels),
            ResnetBlock2D(channels, channels),
        ]
        self.attentions = [AttentionBlock(channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """VAE Encoder: image (B, H, W, 3) -> latent parameters (B, H/8, W/8, 8)."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = []
        for i, out_ch in enumerate(block_out_channels):
            in_ch = block_out_channels[i - 1] if i > 0 else block_out_channels[0]
            add_downsample = i < len(block_out_channels) - 1
            self.down_blocks.append(
                DownEncoderBlock2D(in_ch, out_ch, num_layers=2, add_downsample=add_downsample)
            )

        self.mid_block = MidBlock2D(block_out_channels[-1])

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1])
        # 2 * latent_channels for mean + logvar
        self.conv_out = nn.Conv2d(block_out_channels[-1], 2 * latent_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv_in(x)

        for block in self.down_blocks:
            h = block(h)

        h = self.mid_block(h)

        h = self.conv_norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """VAE Decoder: latent (B, H/8, W/8, 4) -> image (B, H, W, 3)."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
    ):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))

        self.conv_in = nn.Conv2d(latent_channels, reversed_channels[0], kernel_size=3, padding=1)

        self.mid_block = MidBlock2D(reversed_channels[0])

        self.up_blocks = []
        for i, out_ch in enumerate(reversed_channels):
            in_ch = reversed_channels[i - 1] if i > 0 else reversed_channels[0]
            add_upsample = i < len(reversed_channels) - 1
            self.up_blocks.append(
                UpDecoderBlock2D(in_ch, out_ch, num_layers=3, add_upsample=add_upsample)
            )

        self.conv_norm_out = nn.GroupNorm(32, reversed_channels[-1])
        self.conv_out = nn.Conv2d(reversed_channels[-1], out_channels, kernel_size=3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        h = self.conv_in(z)

        h = self.mid_block(h)

        for block in self.up_blocks:
            h = block(h)

        h = self.conv_norm_out(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h


# ---------------------------------------------------------------------------
# AutoencoderKL
# ---------------------------------------------------------------------------


class AutoencoderKLMLX(nn.Module):
    """Stable Diffusion 2.1 VAE ported to MLX.

    All tensors are NHWC: (B, H, W, C).
    Encodes images in [-1, 1] to latents, decodes latents back to images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.encoder = Encoder(in_channels, latent_channels, block_out_channels)
        self.decoder = Decoder(out_channels, latent_channels, block_out_channels)
        # Post-quantization conv pair (identity-initialized in SD)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(self, x: mx.array) -> mx.array:
        """Encode image to latent mean (for inference, no sampling).

        Args:
            x: (B, H, W, 3) float32 images in [-1, 1].

        Returns:
            (B, H/8, W/8, 4) latent mean, scaled by scaling_factor.
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        # Split into mean and logvar along channel axis
        mean, _logvar = mx.split(h, 2, axis=-1)
        return mean * self.scaling_factor

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to image.

        Args:
            z: (B, H/8, W/8, 4) latents (already scaled by scaling_factor).

        Returns:
            (B, H, W, 3) decoded image.
        """
        z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode then decode (for testing roundtrips)."""
        latent = self.encode(x)
        return self.decode(latent)


# ---------------------------------------------------------------------------
# Weight Conversion: PyTorch -> MLX
# ---------------------------------------------------------------------------


def convert_vae_weights_to_mlx(pytorch_state_dict: dict) -> dict:
    """Convert PyTorch VAE state dict to MLX format.

    Handles:
    - Conv2d weights: (out_ch, in_ch, kH, kW) -> (out_ch, kH, kW, in_ch)
    - Linear weights: transpose
    - GroupNorm weight/bias: unchanged

    Args:
        pytorch_state_dict: PyTorch state dict (already on CPU, as numpy or torch tensors).

    Returns:
        dict of MLX arrays ready for model.load_weights().
    """
    import numpy as np

    mlx_weights = {}

    for key, value in pytorch_state_dict.items():
        # Convert torch tensor to numpy if needed
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value)

        if "conv" in key and "weight" in key and value.ndim == 4:
            # Conv2d: PyTorch (out_ch, in_ch, kH, kW) -> MLX (out_ch, kH, kW, in_ch)
            value = np.transpose(value, (0, 2, 3, 1))
        elif value.ndim == 2 and "norm" not in key:
            # Linear weight: transpose
            value = value.T

        mlx_weights[key] = mx.array(value)

    return mlx_weights


def map_diffusers_vae_keys(diffusers_state_dict: dict) -> dict:
    """Map diffusers VAE key names to AutoencoderKLMLX key names.

    Diffusers uses keys like:
        encoder.down_blocks.0.resnets.0.norm1.weight
        decoder.mid_block.attentions.0.group_norm.weight
        decoder.mid_block.attentions.0.to_q.weight  (as key/query/value)

    This function maps the attention projection keys from diffusers format
    to our format.

    Args:
        diffusers_state_dict: state dict with diffusers key names.

    Returns:
        dict with keys matching AutoencoderKLMLX parameter names.
    """
    mapped = {}

    for key, value in diffusers_state_dict.items():
        new_key = key

        # Diffusers attention uses 'key' for to_k, 'query' for to_q, etc.
        new_key = new_key.replace(".query.", ".to_q.")
        new_key = new_key.replace(".key.", ".to_k.")
        new_key = new_key.replace(".value.", ".to_v.")
        new_key = new_key.replace(".proj_attn.", ".to_out.")

        # Diffusers conv_shortcut is named nin_shortcut in some older checkpoints
        new_key = new_key.replace(".nin_shortcut.", ".conv_shortcut.")

        mapped[new_key] = value

    return mapped
