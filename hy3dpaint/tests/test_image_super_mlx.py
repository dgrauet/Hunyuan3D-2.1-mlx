"""Tests for MLX RRDBNet (RealESRGAN) super-resolution model."""

import os
import sys

import mlx.core as mx
import numpy as np
import pytest

# Add hy3dpaint to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.image_super_utils_mlx import (
    RRDBNetMLX,
    ResidualDenseBlock,
    RRDB,
    _nearest_upsample_2x,
    imageSuperNetMLX,
)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------


class TestResidualDenseBlock:
    """Tests for the ResidualDenseBlock module."""

    def test_creation(self):
        rdb = ResidualDenseBlock(num_feat=64, num_grow_ch=32)
        assert rdb.conv1 is not None
        assert rdb.conv5 is not None

    def test_forward_shape(self):
        rdb = ResidualDenseBlock(num_feat=64, num_grow_ch=32)
        x = mx.zeros((1, 16, 16, 64))
        out = rdb(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 64), f"Expected (1,16,16,64), got {out.shape}"

    def test_residual_connection(self):
        """Output should differ from input (conv weights are random)."""
        rdb = ResidualDenseBlock(num_feat=64, num_grow_ch=32)
        x = mx.ones((1, 4, 4, 64))
        out = rdb(x)
        mx.eval(out)
        # The residual path adds x, so output should not be zero
        assert not mx.allclose(out, mx.zeros_like(out)).item()


class TestRRDB:
    """Tests for the RRDB module."""

    def test_creation(self):
        rrdb = RRDB(num_feat=64, num_grow_ch=32)
        assert len(rrdb.rdb1.conv1.weight.shape) == 4

    def test_forward_shape(self):
        rrdb = RRDB(num_feat=64, num_grow_ch=32)
        x = mx.zeros((1, 8, 8, 64))
        out = rrdb(x)
        mx.eval(out)
        assert out.shape == (1, 8, 8, 64), f"Expected (1,8,8,64), got {out.shape}"


class TestRRDBNetMLX:
    """Tests for the full RRDBNet model."""

    def test_creation_default(self):
        model = RRDBNetMLX()
        assert model.scale == 4
        assert len(model.body) == 23

    def test_creation_custom(self):
        model = RRDBNetMLX(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=4, num_grow_ch=16, scale=4)
        assert len(model.body) == 4

    def test_forward_shape_scale4(self):
        """Input 64x64x3 should produce 256x256x3 with scale=4."""
        model = RRDBNetMLX(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=2, num_grow_ch=16, scale=4)
        x = mx.zeros((1, 64, 64, 3))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 256, 256, 3), f"Expected (1,256,256,3), got {out.shape}"

    def test_forward_shape_scale2(self):
        """Input 64x64x3 should produce 128x128x3 with scale=2."""
        model = RRDBNetMLX(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=2, num_grow_ch=16, scale=2)
        x = mx.zeros((1, 64, 64, 3))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 128, 128, 3), f"Expected (1,128,128,3), got {out.shape}"

    def test_forward_nonzero_input(self):
        """Model should produce non-zero output for non-zero input."""
        model = RRDBNetMLX(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=2, num_grow_ch=16, scale=4)
        x = mx.ones((1, 16, 16, 3)) * 0.5
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 64, 64, 3)
        assert not mx.allclose(out, mx.zeros_like(out)).item()


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestNearestUpsample:
    """Tests for the nearest-neighbor upsampling function."""

    def test_2x_upsample_shape(self):
        x = mx.zeros((1, 8, 8, 3))
        out = _nearest_upsample_2x(x)
        mx.eval(out)
        assert out.shape == (1, 16, 16, 3)

    def test_2x_upsample_values(self):
        """Each pixel should be replicated in a 2x2 block."""
        x = mx.array([[[[1.0, 2.0, 3.0]]]])  # (1, 1, 1, 3)
        out = _nearest_upsample_2x(x)
        mx.eval(out)
        assert out.shape == (1, 2, 2, 3)
        expected = mx.array([[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                              [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]])
        assert mx.allclose(out, expected).item()


class TestPixelUnshuffle:
    """Tests for the pixel_unshuffle static method."""

    def test_unshuffle_scale2(self):
        x = mx.zeros((1, 8, 8, 3))
        out = RRDBNetMLX._pixel_unshuffle(x, 2)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 12)

    def test_unshuffle_scale4(self):
        x = mx.zeros((1, 16, 16, 3))
        out = RRDBNetMLX._pixel_unshuffle(x, 4)
        mx.eval(out)
        assert out.shape == (1, 4, 4, 48)


# ---------------------------------------------------------------------------
# Weight loading test (conditional on weight file availability)
# ---------------------------------------------------------------------------


WEIGHTS_PATH = os.environ.get("REALESRGAN_MLX_WEIGHTS", None)


@pytest.mark.skipif(WEIGHTS_PATH is None, reason="Set REALESRGAN_MLX_WEIGHTS env var to test weight loading")
class TestWeightLoading:
    """Tests that require converted MLX weights."""

    def test_load_weights(self):
        wrapper = imageSuperNetMLX(WEIGHTS_PATH)
        assert wrapper.model is not None

    def test_inference_with_weights(self):
        from PIL import Image

        wrapper = imageSuperNetMLX(WEIGHTS_PATH)
        # Create a small test image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        out = wrapper(img)
        assert out.size == (128, 128), f"Expected (128,128), got {out.size}"
