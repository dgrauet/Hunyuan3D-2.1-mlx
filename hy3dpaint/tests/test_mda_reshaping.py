"""Tests for MDA batch ordering and chunk scatter logic.

Verifies that:
1. Chunk indices map correctly between the chunk-local ordering
   (albedo views first, then MR views) and the full-batch ordering
   (all albedo views, then all MR views).
2. Scatter-back produces the same result as a single non-chunked pass.
3. MDA index splitting within a chunk is correct.
4. Multiview attention reshape groups views per material correctly.
"""

import mlx.core as mx
import numpy as np
import pytest


# -- Helpers reproducing the inference.py chunk logic --

def build_chunk_indices(n_views: int, n_pbr: int, chunk_size: int):
    """Return list of (chunk_idx, n_chunk) per chunk."""
    chunks = []
    for v_start in range(0, n_views, chunk_size):
        v_end = min(v_start + chunk_size, n_views)
        n_chunk = v_end - v_start
        chunk_idx = list(range(v_start, v_end)) + [
            n_views + j for j in range(v_start, v_end)
        ]
        chunks.append((chunk_idx, n_chunk))
    return chunks


class TestChunkIndexOrdering:
    """Verify chunk_idx maps chunk-local positions to correct full-batch positions."""

    def test_6_views_chunks_of_2(self):
        """6 views, chunk_size=2: 3 chunks of 4 samples each."""
        n_views, n_pbr, chunk_size = 6, 2, 2
        chunks = build_chunk_indices(n_views, n_pbr, chunk_size)

        assert len(chunks) == 3
        # Chunk 0: albedo views 0,1 + MR views 0,1
        assert chunks[0] == ([0, 1, 6, 7], 2)
        # Chunk 1: albedo views 2,3 + MR views 2,3
        assert chunks[1] == ([2, 3, 8, 9], 2)
        # Chunk 2: albedo views 4,5 + MR views 4,5
        assert chunks[2] == ([4, 5, 10, 11], 2)

    def test_2_views_single_chunk(self):
        """2 views, chunk_size=2: single chunk covers everything."""
        n_views, n_pbr, chunk_size = 2, 2, 2
        chunks = build_chunk_indices(n_views, n_pbr, chunk_size)
        assert len(chunks) == 1
        assert chunks[0] == ([0, 1, 2, 3], 2)

    def test_all_indices_covered(self):
        """Every position in the full batch is written to exactly once."""
        for n_views in (2, 4, 6, 8):
            n_pbr, chunk_size = 2, 2
            chunks = build_chunk_indices(n_views, n_pbr, chunk_size)
            all_idx = []
            for chunk_idx, _ in chunks:
                all_idx.extend(chunk_idx)
            assert sorted(all_idx) == list(range(n_pbr * n_views)), (
                f"n_views={n_views}: indices {sorted(all_idx)} != {list(range(n_pbr * n_views))}"
            )

    def test_odd_views_chunks_of_2(self):
        """5 views, chunk_size=2: last chunk has 1 view."""
        n_views, n_pbr, chunk_size = 5, 2, 2
        chunks = build_chunk_indices(n_views, n_pbr, chunk_size)
        assert len(chunks) == 3
        # Last chunk: 1 view
        assert chunks[2] == ([4, 9], 1)
        all_idx = []
        for chunk_idx, _ in chunks:
            all_idx.extend(chunk_idx)
        assert sorted(all_idx) == list(range(n_pbr * n_views))


class TestScatterBack:
    """Verify that scatter-back with indexed assignment matches unchunked."""

    def test_scatter_matches_full(self):
        """Simulate noise prediction and verify scatter order matches direct."""
        n_views, n_pbr, chunk_size = 6, 2, 2
        n_total = n_pbr * n_views  # 12
        h, w, c = 2, 2, 1  # tiny spatial dims for test

        # Create unique "noise predictions" for each sample
        # In the full (unchunked) case, sample i gets value i
        full_noise = mx.arange(n_total).astype(mx.float32).reshape(n_total, 1, 1, 1)
        full_noise = mx.broadcast_to(full_noise, (n_total, h, w, c))

        # Simulate chunked scatter-back
        scattered = mx.zeros((n_total, h, w, c))
        chunks = build_chunk_indices(n_views, n_pbr, chunk_size)
        for chunk_idx, n_chunk in chunks:
            # In real inference, the UNet produces predictions for the chunk.
            # Here we simulate by gathering the correct values from full_noise.
            chunk_noise = full_noise[mx.array(chunk_idx)]
            scattered[mx.array(chunk_idx)] = chunk_noise

        # After scatter, scattered should equal full_noise
        np.testing.assert_array_equal(np.array(scattered), np.array(full_noise))

    def test_scatter_preserves_values(self):
        """Each chunk produces distinct values; verify they land correctly."""
        n_views, n_pbr, chunk_size = 6, 2, 2
        n_total = n_pbr * n_views

        scattered = mx.zeros((n_total, 1))
        chunks = build_chunk_indices(n_views, n_pbr, chunk_size)

        for chunk_idx, n_chunk in chunks:
            # Mark each position with its global index
            vals = mx.array(chunk_idx, dtype=mx.float32).reshape(-1, 1)
            scattered[mx.array(chunk_idx)] = vals

        result = np.array(scattered).flatten()
        expected = np.arange(n_total, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestMDAIndexSplit:
    """Verify MDA albedo/MR index splitting within a chunk."""

    @staticmethod
    def mda_split(B_total, n_views, n_pbr):
        """Reproduce the MDA index logic from BasicTransformerBlock."""
        B = B_total // (n_pbr * n_views)
        albedo_idx = list(range(0, B * n_views))
        mr_idx = list(range(B * n_views, B * n_pbr * n_views))
        return albedo_idx, mr_idx

    def test_chunk_of_2_views(self):
        """Chunk with 2 views, 2 materials: 4 samples."""
        # Chunk layout: [albedo_v0, albedo_v1, mr_v0, mr_v1]
        albedo_idx, mr_idx = self.mda_split(B_total=4, n_views=2, n_pbr=2)
        assert albedo_idx == [0, 1]
        assert mr_idx == [2, 3]

    def test_chunk_of_1_view(self):
        """Chunk with 1 view, 2 materials: 2 samples."""
        albedo_idx, mr_idx = self.mda_split(B_total=2, n_views=1, n_pbr=2)
        assert albedo_idx == [0]
        assert mr_idx == [1]

    def test_no_chunking_6_views(self):
        """Full batch, 6 views, 2 materials: 12 samples."""
        albedo_idx, mr_idx = self.mda_split(B_total=12, n_views=6, n_pbr=2)
        assert albedo_idx == [0, 1, 2, 3, 4, 5]
        assert mr_idx == [6, 7, 8, 9, 10, 11]

    def test_mda_split_matches_chunk_layout(self):
        """For each chunk, MDA split correctly identifies albedo vs MR samples."""
        n_views, n_pbr, chunk_size = 6, 2, 2
        chunks = build_chunk_indices(n_views, n_pbr, chunk_size)

        for chunk_idx, n_chunk in chunks:
            B_total = len(chunk_idx)
            albedo_idx, mr_idx = self.mda_split(B_total, n_chunk, n_pbr)

            # Map chunk-local indices to global indices
            global_albedo = [chunk_idx[i] for i in albedo_idx]
            global_mr = [chunk_idx[i] for i in mr_idx]

            # Albedo samples should be in [0, n_views)
            for g in global_albedo:
                assert 0 <= g < n_views, (
                    f"Albedo sample {g} not in albedo range [0, {n_views})"
                )
            # MR samples should be in [n_views, 2*n_views)
            for g in global_mr:
                assert n_views <= g < 2 * n_views, (
                    f"MR sample {g} not in MR range [{n_views}, {2*n_views})"
                )


class TestMultiviewReshape:
    """Verify multiview attention reshape groups views per material."""

    def test_reshape_groups_correctly(self):
        """Reshape (B_total, L, C) -> (B_mat, n_views*L, C) groups same-material views."""
        n_views, n_pbr = 2, 2
        L, C = 4, 8
        B_total = n_pbr * n_views  # 4

        # Create data where each sample has a unique marker in first element
        data = mx.zeros((B_total, L, C))
        for s in range(B_total):
            data[s] = mx.full((L, C), float(s))

        B_mat = B_total // n_views  # 2
        mv_input = data.reshape(B_mat, n_views * L, C)

        # Group 0 should contain samples 0,1 (albedo views)
        group0 = np.array(mv_input[0])
        assert np.all(group0[:L] == 0.0), "First L tokens should be sample 0 (albedo v0)"
        assert np.all(group0[L:] == 1.0), "Next L tokens should be sample 1 (albedo v1)"

        # Group 1 should contain samples 2,3 (MR views)
        group1 = np.array(mv_input[1])
        assert np.all(group1[:L] == 2.0), "First L tokens should be sample 2 (mr v0)"
        assert np.all(group1[L:] == 3.0), "Next L tokens should be sample 3 (mr v1)"

    def test_reshape_roundtrip(self):
        """Reshape and un-reshape preserves data."""
        n_views, n_pbr = 3, 2
        L, C = 4, 8
        B_total = n_pbr * n_views

        data = mx.random.normal((B_total, L, C))
        B_mat = B_total // n_views
        reshaped = data.reshape(B_mat, n_views * L, C)
        restored = reshaped.reshape(B_total, L, C)

        np.testing.assert_allclose(
            np.array(restored), np.array(data), atol=1e-6
        )
