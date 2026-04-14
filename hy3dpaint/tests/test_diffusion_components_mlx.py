"""Tests for MLX diffusion model components (Phase 1 + Phase 2)."""

import sys
import os

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hunyuanpaintpbr_mlx.unet.attn_processor_mlx import (
    ImageProjModel,
    PoseRoPEAttnProcessor,
    RefAttnProcessor,
    RotaryEmbedding,
    SelfAttnProcessor,
    reshape_for_attention,
    reshape_from_attention,
    scaled_dot_product_attention,
)
from hunyuanpaintpbr_mlx.scheduler_mlx import (
    SchedulerConfig,
    UniPCMultistepSchedulerMLX,
)


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------

class TestRotaryEmbedding:
    def test_1d_shape(self):
        pos = mx.arange(32)
        cos, sin = RotaryEmbedding.get_1d_rotary_pos_embed(96, pos)
        assert cos.shape == (32, 96)
        assert sin.shape == (32, 96)

    def test_1d_values_normalized(self):
        pos = mx.arange(16)
        cos, sin = RotaryEmbedding.get_1d_rotary_pos_embed(64, pos)
        mx.synchronize()
        # cos^2 + sin^2 should be 1 for each position
        magnitude = np.array(cos * cos + sin * sin)
        np.testing.assert_allclose(magnitude, 1.0, atol=1e-5)

    def test_3d_shape(self):
        position = mx.zeros((2, 100, 3), dtype=mx.int32)
        cos, sin = RotaryEmbedding.get_3d_rotary_pos_embed(position, 96, 128)
        assert cos.shape == (2, 100, 96)
        assert sin.shape == (2, 100, 96)

    def test_apply_rotary_preserves_shape(self):
        x = mx.random.normal((2, 8, 64, 96))
        pos = mx.arange(64)
        cos, sin = RotaryEmbedding.get_1d_rotary_pos_embed(96, pos)
        result = RotaryEmbedding.apply_rotary_emb(x, (cos, sin))
        assert result.shape == x.shape


# ---------------------------------------------------------------------------
# ImageProjModel
# ---------------------------------------------------------------------------

class TestImageProjModel:
    def test_output_shape(self):
        model = ImageProjModel(
            cross_attention_dim=768,
            clip_embeddings_dim=1536,
            clip_extra_context_tokens=4,
        )
        x = mx.random.normal((1, 100, 1536))
        out = model(x)
        mx.synchronize()
        assert out.shape == (1, 400, 768)  # 100 * 4 = 400

    def test_2d_input(self):
        model = ImageProjModel(
            cross_attention_dim=768,
            clip_embeddings_dim=1536,
            clip_extra_context_tokens=4,
        )
        x = mx.random.normal((50, 1536))
        out = model(x)
        mx.synchronize()
        assert out.shape == (50, 4, 768)


# ---------------------------------------------------------------------------
# Attention utilities
# ---------------------------------------------------------------------------

class TestAttentionUtils:
    def test_reshape_roundtrip(self):
        x = mx.random.normal((2, 64, 768))
        reshaped = reshape_for_attention(x, 8)
        assert reshaped.shape == (2, 8, 64, 96)  # 768/8 = 96
        back = reshape_from_attention(reshaped)
        assert back.shape == (2, 64, 768)
        mx.synchronize()
        np.testing.assert_allclose(np.array(back), np.array(x), atol=1e-5)

    def test_sdpa_shape(self):
        q = mx.random.normal((2, 8, 64, 96))
        k = mx.random.normal((2, 8, 32, 96))
        v = mx.random.normal((2, 8, 32, 96))
        out = scaled_dot_product_attention(q, k, v)
        mx.synchronize()
        assert out.shape == (2, 8, 64, 96)

    def test_sdpa_self_attention(self):
        q = mx.random.normal((1, 1, 4, 8))
        out = scaled_dot_product_attention(q, q, q)
        mx.synchronize()
        assert out.shape == (1, 1, 4, 8)


# ---------------------------------------------------------------------------
# Attention Processors
# ---------------------------------------------------------------------------

class TestSelfAttnProcessor:
    def test_output_shape(self):
        import mlx.nn as nn
        proc = SelfAttnProcessor(query_dim=768, num_heads=8, pbr_settings=("albedo", "mr"))
        to_q = nn.Linear(768, 768)
        to_k = nn.Linear(768, 768)
        to_v = nn.Linear(768, 768)
        to_out = nn.Linear(768, 768)
        hs = mx.random.normal((1, 2, 6, 64, 768))
        out = proc(hs, to_q, to_k, to_v, to_out, n_views=6)
        mx.synchronize()
        assert out.shape == (1, 2, 6, 64, 768)


class TestPoseRoPEAttnProcessor:
    def test_output_shape(self):
        proc = PoseRoPEAttnProcessor(query_dim=768, num_heads=8)
        hs = mx.random.normal((2, 384, 768))  # 2 materials, 6*64=384 tokens
        out = proc(hs)
        mx.synchronize()
        assert out.shape == (2, 384, 768)

    def test_with_rope(self):
        proc = PoseRoPEAttnProcessor(query_dim=768, num_heads=8)
        hs = mx.random.normal((2, 384, 768))
        pos_idx = {
            "voxel_indices": mx.zeros((2, 384, 3), dtype=mx.int32),
            "voxel_resolution": 128,
        }
        out = proc(hs, position_indices=pos_idx)
        mx.synchronize()
        assert out.shape == (2, 384, 768)


class TestRefAttnProcessor:
    def test_output_shape(self):
        proc = RefAttnProcessor(query_dim=768, num_heads=8, pbr_settings=("albedo", "mr"))
        hs = mx.random.normal((1, 384, 768))
        ref = mx.random.normal((1, 64, 768))
        out = proc(hs, ref)
        mx.synchronize()
        assert out.shape == (1, 2, 384, 768)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TestUniPCScheduler:
    def test_set_timesteps(self):
        sched = UniPCMultistepSchedulerMLX()
        sched.set_timesteps(15)
        assert len(sched.timesteps) == 15
        assert sched.timesteps[0] > sched.timesteps[-1]  # descending

    def test_step_shape(self):
        sched = UniPCMultistepSchedulerMLX()
        sched.set_timesteps(15)
        sample = mx.random.normal((4, 4, 64, 64))
        noise_pred = mx.random.normal((4, 4, 64, 64))
        t = sched.timesteps[0]
        result = sched.step(noise_pred, t, sample)
        mx.synchronize()
        assert result.shape == (4, 4, 64, 64)

    def test_multi_step_denoising(self):
        sched = UniPCMultistepSchedulerMLX()
        sched.set_timesteps(5)
        sample = mx.random.normal((1, 4, 8, 8))
        for t in sched.timesteps:
            noise = mx.random.normal(sample.shape)
            sample = sched.step(noise, int(t), sample)
        mx.synchronize()
        assert sample.shape == (1, 4, 8, 8)
        # After denoising, values should be finite
        assert not mx.isnan(sample).any().item()

    def test_scale_model_input_identity(self):
        sched = UniPCMultistepSchedulerMLX()
        sample = mx.ones((1, 4, 8, 8))
        result = sched.scale_model_input(sample, 500)
        np.testing.assert_array_equal(np.array(result), np.array(sample))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
