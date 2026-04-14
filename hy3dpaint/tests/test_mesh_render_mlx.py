"""Tests for MLX mesh renderer and camera utilities."""

import math
import sys
import os

import mlx.core as mx
import numpy as np
import pytest

# Add hy3dpaint to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from DifferentiableRenderer.camera_utils_mlx import (
    get_mv_matrix,
    get_orthographic_projection_matrix,
    get_perspective_projection_matrix,
    transform_pos,
)
from DifferentiableRenderer.mesh_render_mlx import (
    MLXRasterizer,
    MeshRenderMLX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cube_mesh():
    """A unit cube with outward-pointing normals (CCW winding from outside)."""
    verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5],
        [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
    ], dtype=np.float32)
    # Winding reversed vs naive ordering so cross(e1,e2) points outward
    faces = np.array([
        [0,2,1],[0,3,2],  # front  (normal -Z)
        [4,5,6],[4,6,7],  # back   (normal +Z)
        [0,4,7],[0,7,3],  # left   (normal -X)
        [1,2,6],[1,6,5],  # right  (normal +X)
        [3,7,6],[3,6,2],  # top    (normal +Y)
        [0,1,5],[0,5,4],  # bottom (normal -Y)
    ], dtype=np.int32)
    return verts, faces


def _make_single_triangle():
    """A single large triangle in XY plane."""
    verts = np.array([
        [-0.8, -0.8, 0.0],
        [ 0.8, -0.8, 0.0],
        [ 0.0,  0.8, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return verts, faces


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

class TestCameraUtils:
    def test_transform_pos_shape(self):
        mtx = np.eye(4, dtype=np.float32)
        pos = mx.zeros((10, 3))
        result = transform_pos(mtx, pos)
        assert result.shape == (1, 10, 4)

    def test_transform_pos_keepdim(self):
        mtx = np.eye(4, dtype=np.float32)
        pos = mx.zeros((10, 3))
        result = transform_pos(mtx, pos, keepdim=True)
        assert result.shape == (10, 4)

    def test_transform_pos_identity(self):
        mtx = np.eye(4, dtype=np.float32)
        pos = mx.array([[1.0, 2.0, 3.0]])
        result = transform_pos(mtx, pos, keepdim=True)
        mx.synchronize()
        expected = [1.0, 2.0, 3.0, 1.0]
        np.testing.assert_allclose(np.array(result[0]), expected, atol=1e-6)

    def test_transform_pos_with_4d_input(self):
        mtx = np.eye(4, dtype=np.float32)
        pos = mx.array([[1.0, 2.0, 3.0, 1.0]])
        result = transform_pos(mtx, pos, keepdim=True)
        mx.synchronize()
        np.testing.assert_allclose(np.array(result[0]), [1.0, 2.0, 3.0, 1.0], atol=1e-6)

    def test_mv_matrix_shape(self):
        mv = get_mv_matrix(0, 0, 1.45)
        assert mv.shape == (4, 4)
        assert mv.dtype == np.float32

    def test_mv_matrix_is_rigid(self):
        mv = get_mv_matrix(30, 45, 2.0)
        R = mv[:3, :3]
        # R^T @ R should be identity for a rotation matrix
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)

    def test_orthographic_projection(self):
        proj = get_orthographic_projection_matrix()
        assert proj.shape == (4, 4)
        # Origin should map to origin
        result = proj @ np.array([0, 0, -1, 1])
        assert result[3] == 1.0  # w unchanged in ortho

    def test_perspective_projection(self):
        proj = get_perspective_projection_matrix(49.13, 1.0, 0.01, 100.0)
        assert proj.shape == (4, 4)
        assert proj[3, 2] == -1.0  # perspective divide


# ---------------------------------------------------------------------------
# Rasterizer adapter
# ---------------------------------------------------------------------------

class TestMLXRasterizer:
    def test_rasterize_single_triangle(self):
        verts = mx.array([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=mx.float32)
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)

        fi, bary = MLXRasterizer.rasterize(
            mx.expand_dims(verts, 0), faces, (16, 16)
        )
        mx.synchronize()
        assert fi.shape == (16, 16)
        assert bary.shape == (16, 16, 3)
        covered = (fi > 0).astype(mx.int32).sum().item()
        assert covered > 0

    def test_interpolate_shape(self):
        verts = mx.array([
            [-0.5, -0.5, 0.5, 1.0],
            [ 0.5, -0.5, 0.5, 1.0],
            [ 0.0,  0.5, 0.5, 1.0],
        ], dtype=mx.float32)
        faces = mx.array([[0, 1, 2]], dtype=mx.int32)

        fi, bary = MLXRasterizer.rasterize(
            mx.expand_dims(verts, 0), faces, (16, 16)
        )
        colors = mx.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=mx.float32)
        result = MLXRasterizer.interpolate(
            mx.expand_dims(colors, 0), fi, bary, faces
        )
        mx.synchronize()
        assert result.shape == (1, 16, 16, 3)


# ---------------------------------------------------------------------------
# MeshRenderMLX
# ---------------------------------------------------------------------------

class TestMeshRenderMLX:
    def setup_method(self):
        self.renderer = MeshRenderMLX(
            default_resolution=32,
            shader_type="face",
        )
        verts, faces = _make_cube_mesh()
        self.renderer.set_mesh(verts, faces)

    def test_render_normal_shape(self):
        result = self.renderer.render_normal(0, 0)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float32

    def test_render_normal_coverage(self):
        result = self.renderer.render_normal(0, 0, bg_color=(1, 1, 1))
        # The cube should cover a portion of the image
        # Background is (1,1,1) after normalization
        # Non-background pixels should differ from pure white
        bg_pixel = np.array([1.0, 1.0, 1.0])
        not_bg = np.any(np.abs(result - bg_pixel) > 0.01, axis=-1)
        coverage = not_bg.sum()
        total = 32 * 32
        assert coverage > total * 0.05, f"Too little normal coverage: {coverage}/{total}"

    def test_render_position_shape(self):
        result = self.renderer.render_position(0, 0)
        assert result.shape == (32, 32, 3)

    def test_render_position_values_in_range(self):
        result = self.renderer.render_position(0, 0, bg_color=(1, 1, 1))
        # Position values should be in [0, 1] range
        assert result.min() >= -0.01
        assert result.max() <= 1.01

    def test_render_alpha_shape(self):
        result = self.renderer.render_alpha(0, 0)
        # Returns (1, H, W, 1) face indices matching original MeshRender
        assert result.shape == (1, 32, 32, 1)

    def test_render_alpha_face_indices(self):
        result = self.renderer.render_alpha(0, 0)
        # Should contain face indices: 0 = background, >0 = face IDs
        assert result.dtype == np.int64
        assert (result >= 0).all()

    def test_render_alpha_has_coverage(self):
        result = self.renderer.render_alpha(0, 0)
        covered = (result > 0).sum()
        assert covered > 0, "Cube should be visible from front"

    def test_render_from_different_angles(self):
        """Verify rendering from different elevations produces different results."""
        # Use elevation change (not azimuth) since a cube is rotationally
        # symmetric under 90-degree azimuth steps with face shading.
        n1 = self.renderer.render_normal(0, 0)
        n2 = self.renderer.render_normal(45, 0)
        diff = np.abs(n1 - n2).sum()
        assert diff > 1.0, "Different elevations should produce different normals"


class TestMeshRenderMLXTriangle:
    def setup_method(self):
        self.renderer = MeshRenderMLX(
            default_resolution=16,
            shader_type="face",
        )
        verts, faces = _make_single_triangle()
        self.renderer.set_mesh(verts, faces, auto_center=True)

    def test_render_normal_nonzero(self):
        result = self.renderer.render_normal(0, 0)
        assert result.shape == (16, 16, 3)
        # Should have some non-background pixels
        bg = np.array([1.0, 1.0, 1.0])
        not_bg = np.any(np.abs(result - bg) > 0.01, axis=-1)
        assert not_bg.sum() > 0


class TestMeshNormalization:
    def test_auto_center(self):
        renderer = MeshRenderMLX(default_resolution=16)
        verts = np.array([
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.0],
            [11.0, 12.0, 10.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        renderer.set_mesh(verts, faces, auto_center=True)

        # After normalization, mesh should be centered near origin
        vtx = np.array(renderer.vtx_pos)
        center = vtx.mean(axis=0)
        np.testing.assert_allclose(center, [0, 0, 0], atol=0.5)

    def test_no_auto_center(self):
        renderer = MeshRenderMLX(default_resolution=16)
        verts, faces = _make_cube_mesh()
        renderer.set_mesh(verts, faces, auto_center=False)
        assert renderer.scale_factor == 1.0


def _make_cube_with_uvs():
    """A cube with simple UV coordinates."""
    verts, faces = _make_cube_mesh()
    # Simple planar UV mapping (use xy projected to [0,1])
    uv = np.zeros((len(verts), 2), dtype=np.float32)
    uv[:, 0] = (verts[:, 0] - verts[:, 0].min()) / (verts[:, 0].max() - verts[:, 0].min())
    uv[:, 1] = (verts[:, 1] - verts[:, 1].min()) / (verts[:, 1].max() - verts[:, 1].min())
    return verts, faces, uv


# ---------------------------------------------------------------------------
# Texture baking
# ---------------------------------------------------------------------------

class TestExtractTextiles:
    def test_textiles_populated(self):
        renderer = MeshRenderMLX(default_resolution=32, texture_size=32)
        verts, faces, uv = _make_cube_with_uvs()
        renderer.set_mesh(verts, faces, vtx_uv=uv, uv_idx=faces)
        assert renderer.tex_position is not None
        assert renderer.tex_grid is not None
        assert len(renderer.tex_position) > 0

    def test_textiles_shape(self):
        renderer = MeshRenderMLX(default_resolution=32, texture_size=16)
        verts, faces, uv = _make_cube_with_uvs()
        renderer.set_mesh(verts, faces, vtx_uv=uv, uv_idx=faces)
        k = len(renderer.tex_position)
        assert renderer.tex_position.shape == (k, 4)
        assert renderer.tex_normal.shape == (k, 3)
        assert renderer.tex_grid.shape == (k, 2)
        assert renderer.texture_indices.shape == (16, 16)


class TestUvFeatureMap:
    def test_shape(self):
        renderer = MeshRenderMLX(default_resolution=32, texture_size=16)
        verts, faces, uv = _make_cube_with_uvs()
        renderer.set_mesh(verts, faces, vtx_uv=uv, uv_idx=faces)
        feat = mx.ones((len(verts), 3))
        result = renderer.uv_feature_map(feat)
        mx.synchronize()
        assert result.shape == (16, 16, 3)


class TestBackProject:
    def setup_method(self):
        self.renderer = MeshRenderMLX(
            default_resolution=32, texture_size=32, boundary_scale=0,
        )
        verts, faces, uv = _make_cube_with_uvs()
        self.renderer.set_mesh(verts, faces, vtx_uv=uv, uv_idx=faces)

    def test_output_shapes(self):
        # Render a solid red image and back-project
        image = np.ones((32, 32, 3), dtype=np.float32) * np.array([1, 0, 0])
        texture, cos_map, boundary = self.renderer.back_project(image, 0, 0)
        assert texture.shape == (32, 32, 3)
        assert cos_map.shape == (32, 32, 1)
        assert boundary.shape == (32, 32, 1)

    def test_nonzero_texture(self):
        image = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        texture, cos_map, _ = self.renderer.back_project(image, 0, 0)
        # Some texture pixels should be filled
        filled = (cos_map > 0).sum()
        assert filled > 0, "back_project should fill some texture pixels"


class TestBakeTexture:
    def test_multi_view_bake(self):
        renderer = MeshRenderMLX(
            default_resolution=32, texture_size=32, boundary_scale=0,
        )
        verts, faces, uv = _make_cube_with_uvs()
        renderer.set_mesh(verts, faces, vtx_uv=uv, uv_idx=faces)

        # Bake from 2 views with solid colors
        img1 = np.ones((32, 32, 3), dtype=np.float32) * 0.8
        img2 = np.ones((32, 32, 3), dtype=np.float32) * 0.6

        texture, trust = renderer.bake_texture(
            [img1, img2], [0, 0], [0, 90], exp=1,
        )
        assert texture.shape == (32, 32, 3)
        assert trust.shape == (32, 32, 1)
        filled = trust.sum()
        assert filled > 0, "Baked texture should have some coverage"

    def test_coverage_increases_with_views(self):
        renderer = MeshRenderMLX(
            default_resolution=32, texture_size=32, boundary_scale=0,
        )
        verts, faces, uv = _make_cube_with_uvs()
        renderer.set_mesh(verts, faces, vtx_uv=uv, uv_idx=faces)

        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5

        _, trust1 = renderer.bake_texture([img], [0], [0], exp=1)
        _, trust2 = renderer.bake_texture(
            [img, img], [0, 0], [0, 90], exp=1,
        )
        c1 = trust1.sum()
        c2 = trust2.sum()
        assert c2 >= c1, "More views should give equal or more coverage"


class TestFastBakeTexture:
    def test_weighted_merge(self):
        renderer = MeshRenderMLX(texture_size=4)
        # Two views covering different halves: no 99% overlap skip
        t1 = np.ones((4, 4, 3), dtype=np.float32)
        t2 = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        c1 = np.zeros((4, 4, 1), dtype=np.float32)
        c2 = np.zeros((4, 4, 1), dtype=np.float32)
        c1[:2, :] = 1.0  # top half
        c2[2:, :] = 1.0  # bottom half
        merged, trust = renderer.fast_bake_texture([t1, t2], [c1, c2])
        # Top half = 1.0, bottom half = 0.5
        np.testing.assert_allclose(merged[:2], 1.0, atol=1e-6)
        np.testing.assert_allclose(merged[2:], 0.5, atol=1e-6)
        assert trust.all()


# ---------------------------------------------------------------------------
# Pipeline integration (ViewProcessorMLX + MeshRenderMLX)
# ---------------------------------------------------------------------------

class TestMeshManagement:
    def test_get_face_areas(self):
        renderer = MeshRenderMLX(default_resolution=16)
        verts, faces = _make_cube_mesh()
        renderer.set_mesh(verts, faces)
        areas = renderer.get_face_areas()
        assert areas.shape == (12,)
        assert (areas > 0).all()

    def test_get_face_areas_one_indexed(self):
        renderer = MeshRenderMLX(default_resolution=16)
        verts, faces = _make_cube_mesh()
        renderer.set_mesh(verts, faces)
        areas = renderer.get_face_areas(from_one_index=True)
        assert areas.shape == (13,)  # 12 faces + 1 padding
        assert areas[0] == 0.0

    def test_render_alpha_returns_face_indices(self):
        renderer = MeshRenderMLX(default_resolution=32)
        verts, faces = _make_cube_mesh()
        renderer.set_mesh(verts, faces)
        alpha = renderer.render_alpha(0, 0)
        assert alpha.shape == (1, 32, 32, 1)
        assert alpha.dtype == np.int64
        # Should have face indices > 0 for visible faces
        unique = np.unique(alpha)
        assert 0 in unique  # background
        assert len(unique) > 1  # at least one face visible

    def test_set_get_texture(self):
        renderer = MeshRenderMLX(default_resolution=16, texture_size=8)
        tex = np.random.rand(8, 8, 3).astype(np.float32)
        renderer.set_texture(tex)
        got = renderer.get_texture()
        np.testing.assert_array_equal(got, tex)

    def test_get_mesh_roundtrip(self):
        renderer = MeshRenderMLX(default_resolution=16)
        verts, faces = _make_cube_mesh()
        renderer.set_mesh(verts, faces, auto_center=False)
        vtx_out, idx_out, _, _ = renderer.get_mesh(normalize=True)
        # After set_mesh + get_mesh, geometry should survive the roundtrip
        assert vtx_out.shape == verts.shape
        assert idx_out.shape == faces.shape

    def test_render_normal_pil(self):
        from PIL import Image as PILImage
        renderer = MeshRenderMLX(default_resolution=16)
        verts, faces = _make_cube_mesh()
        renderer.set_mesh(verts, faces)
        result = renderer.render_normal(0, 0, return_type="pl")
        assert isinstance(result, PILImage.Image)
        assert result.size == (16, 16)

    def test_set_default_render_resolution(self):
        renderer = MeshRenderMLX(default_resolution=32)
        assert renderer.default_resolution == (32, 32)
        renderer.set_default_render_resolution(64)
        assert renderer.default_resolution == (64, 64)


class TestViewProcessorMLX:
    def setup_method(self):
        from utils.pipeline_utils_mlx import ViewProcessorMLX

        class FakeConfig:
            bake_exp = 4

        self.renderer = MeshRenderMLX(
            default_resolution=32, texture_size=32, boundary_scale=0,
        )
        verts, faces, uv = _make_cube_with_uvs()
        self.renderer.set_mesh(verts, faces, vtx_uv=uv, uv_idx=faces)
        self.vp = ViewProcessorMLX(FakeConfig(), self.renderer)

    def test_render_normal_multiview(self):
        from PIL import Image as PILImage
        normals = self.vp.render_normal_multiview([0, 0], [0, 90])
        assert len(normals) == 2
        assert isinstance(normals[0], PILImage.Image)

    def test_render_position_multiview(self):
        from PIL import Image as PILImage
        positions = self.vp.render_position_multiview([0], [0])
        assert len(positions) == 1
        assert isinstance(positions[0], PILImage.Image)

    def test_bake_from_multiview(self):
        img1 = np.ones((32, 32, 3), dtype=np.float32) * 0.7
        img2 = np.ones((32, 32, 3), dtype=np.float32) * 0.3
        texture, trust = self.vp.bake_from_multiview(
            [img1, img2], [0, 0], [0, 90], [1.0, 1.0],
        )
        assert texture.shape == (32, 32, 3)
        assert trust.shape == (32, 32, 1)

    def test_bake_view_selection(self):
        elevs = [0, 0, 0, 0, 90, -90]
        azims = [0, 90, 180, 270, 0, 180]
        weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        sel_e, sel_a, sel_w = self.vp.bake_view_selection(
            elevs, azims, weights, max_selected_view_num=6,
        )
        assert len(sel_e) == 6
        assert len(sel_a) == 6
        assert len(sel_w) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
