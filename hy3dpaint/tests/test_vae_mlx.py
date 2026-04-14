"""Tests for MLX VAE (AutoencoderKL) port."""

import sys
import os

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hunyuanpaintpbr_mlx.vae_mlx import (
    AttentionBlock,
    AutoencoderKLMLX,
    Decoder,
    DownEncoderBlock2D,
    Encoder,
    ResnetBlock2D,
    UpDecoderBlock2D,
    convert_vae_weights_to_mlx,
)


# ---------------------------------------------------------------------------
# ResnetBlock2D
# ---------------------------------------------------------------------------


class TestResnetBlock2D:
    def test_same_channels(self):
        block = ResnetBlock2D(128, 128)
        x = mx.random.normal((1, 16, 16, 128))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 128)

    def test_different_channels(self):
        block = ResnetBlock2D(128, 256)
        x = mx.random.normal((1, 16, 16, 128))
        out = block(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 256)
        assert block.conv_shortcut is not None

    def test_output_finite(self):
        block = ResnetBlock2D(64, 64)
        x = mx.random.normal((1, 8, 8, 64))
        out = block(x)
        mx.eval(out)
        assert not mx.isnan(out).any().item()


# ---------------------------------------------------------------------------
# AttentionBlock
# ---------------------------------------------------------------------------


class TestAttentionBlock:
    def test_shape(self):
        attn = AttentionBlock(256)
        x = mx.random.normal((1, 8, 8, 256))
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 256)

    def test_residual_connection(self):
        attn = AttentionBlock(64, num_groups=32)
        x = mx.zeros((1, 4, 4, 64))
        out = attn(x)
        mx.eval(out)
        # With zero input the residual should keep output near zero
        assert out.shape == (1, 4, 4, 64)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_output_shape(self):
        enc = Encoder(in_channels=3, latent_channels=4)
        x = mx.random.normal((1, 64, 64, 3))
        out = enc(x)
        mx.eval(out)
        # 8 channels (mean + logvar), spatial / 8
        assert out.shape == (1, 8, 8, 8)

    def test_output_finite(self):
        enc = Encoder(in_channels=3, latent_channels=4)
        x = mx.random.normal((1, 32, 32, 3))
        out = enc(x)
        mx.eval(out)
        assert not mx.isnan(out).any().item()


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(out_channels=3, latent_channels=4)
        z = mx.random.normal((1, 8, 8, 4))
        out = dec(z)
        mx.eval(out)
        assert out.shape == (1, 64, 64, 3)

    def test_output_finite(self):
        dec = Decoder(out_channels=3, latent_channels=4)
        z = mx.random.normal((1, 4, 4, 4))
        out = dec(z)
        mx.eval(out)
        assert not mx.isnan(out).any().item()


# ---------------------------------------------------------------------------
# AutoencoderKLMLX
# ---------------------------------------------------------------------------


class TestAutoEncoderKLMLX:
    def test_encode_shape_64(self):
        """(1, 64, 64, 3) -> (1, 8, 8, 4)"""
        vae = AutoencoderKLMLX()
        x = mx.random.normal((1, 64, 64, 3))
        latent = vae.encode(x)
        mx.eval(latent)
        assert latent.shape == (1, 8, 8, 4)

    def test_decode_shape_64(self):
        """(1, 8, 8, 4) -> (1, 64, 64, 3)"""
        vae = AutoencoderKLMLX()
        z = mx.random.normal((1, 8, 8, 4))
        out = vae.decode(z)
        mx.eval(out)
        assert out.shape == (1, 64, 64, 3)

    def test_roundtrip_shape_64(self):
        """Encode then decode preserves spatial dimensions."""
        vae = AutoencoderKLMLX()
        x = mx.random.normal((1, 64, 64, 3))
        reconstructed = vae(x)
        mx.eval(reconstructed)
        assert reconstructed.shape == (1, 64, 64, 3)

    def test_encode_shape_32(self):
        """(1, 32, 32, 3) -> (1, 4, 4, 4)"""
        vae = AutoencoderKLMLX()
        x = mx.random.normal((1, 32, 32, 3))
        latent = vae.encode(x)
        mx.eval(latent)
        assert latent.shape == (1, 4, 4, 4)

    def test_decode_shape_32(self):
        """(1, 4, 4, 4) -> (1, 32, 32, 3)"""
        vae = AutoencoderKLMLX()
        z = mx.random.normal((1, 4, 4, 4))
        out = vae.decode(z)
        mx.eval(out)
        assert out.shape == (1, 32, 32, 3)

    def test_encode_shape_128(self):
        """(1, 128, 128, 3) -> (1, 16, 16, 4)"""
        vae = AutoencoderKLMLX()
        x = mx.random.normal((1, 128, 128, 3))
        latent = vae.encode(x)
        mx.eval(latent)
        assert latent.shape == (1, 16, 16, 4)

    def test_decode_shape_128(self):
        """(1, 16, 16, 4) -> (1, 128, 128, 3)"""
        vae = AutoencoderKLMLX()
        z = mx.random.normal((1, 16, 16, 4))
        out = vae.decode(z)
        mx.eval(out)
        assert out.shape == (1, 128, 128, 3)

    def test_roundtrip_shape_128(self):
        """Encode then decode at 128x128."""
        vae = AutoencoderKLMLX()
        x = mx.random.normal((1, 128, 128, 3))
        reconstructed = vae(x)
        mx.eval(reconstructed)
        assert reconstructed.shape == (1, 128, 128, 3)

    def test_output_finite(self):
        vae = AutoencoderKLMLX()
        x = mx.random.normal((1, 32, 32, 3))
        latent = vae.encode(x)
        decoded = vae.decode(latent)
        mx.eval(decoded)
        assert not mx.isnan(latent).any().item()
        assert not mx.isnan(decoded).any().item()

    def test_scaling_factor(self):
        vae = AutoencoderKLMLX()
        assert vae.scaling_factor == 0.18215

    def test_batch_size_2(self):
        vae = AutoencoderKLMLX()
        x = mx.random.normal((2, 32, 32, 3))
        latent = vae.encode(x)
        mx.eval(latent)
        assert latent.shape == (2, 4, 4, 4)
        decoded = vae.decode(latent)
        mx.eval(decoded)
        assert decoded.shape == (2, 32, 32, 3)


# ---------------------------------------------------------------------------
# Weight Conversion
# ---------------------------------------------------------------------------


class TestWeightConversion:
    def test_conv_weight_transpose(self):
        """Conv2d weights: (out, in, kH, kW) -> (out, kH, kW, in)."""
        pt_weights = {
            "encoder.conv_in.weight": np.random.randn(128, 3, 3, 3).astype(np.float32),
            "encoder.conv_in.bias": np.random.randn(128).astype(np.float32),
        }
        mlx_weights = convert_vae_weights_to_mlx(pt_weights)
        assert mlx_weights["encoder.conv_in.weight"].shape == (128, 3, 3, 3)
        assert mlx_weights["encoder.conv_in.bias"].shape == (128,)

    def test_linear_weight_transpose(self):
        """Linear weights should be transposed."""
        pt_weights = {
            "mid_block.attentions.0.to_q.weight": np.random.randn(512, 512).astype(np.float32),
            "mid_block.attentions.0.to_q.bias": np.random.randn(512).astype(np.float32),
        }
        mlx_weights = convert_vae_weights_to_mlx(pt_weights)
        assert mlx_weights["mid_block.attentions.0.to_q.weight"].shape == (512, 512)

    def test_groupnorm_unchanged(self):
        """GroupNorm weight/bias should not be modified."""
        pt_weights = {
            "encoder.conv_norm_out.weight": np.ones(512, dtype=np.float32),
            "encoder.conv_norm_out.bias": np.zeros(512, dtype=np.float32),
        }
        mlx_weights = convert_vae_weights_to_mlx(pt_weights)
        np.testing.assert_array_equal(
            np.array(mlx_weights["encoder.conv_norm_out.weight"]),
            pt_weights["encoder.conv_norm_out.weight"],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
