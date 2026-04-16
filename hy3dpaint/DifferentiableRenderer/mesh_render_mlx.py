"""MLX mesh renderer for Hunyuan3D texture pipeline.

Port of MeshRender's rendering and texture baking pipeline to MLX,
using mlx_ops.rasterize as the Metal rasterization backend.
"""

import mlx.core as mx
import numpy as np

from mlx_ops.rasterize import interpolate as mlx_interpolate
from mlx_ops.rasterize import rasterize_triangles

from .camera_utils_mlx import (
    get_mv_matrix,
    get_orthographic_projection_matrix,
    get_perspective_projection_matrix,
    transform_pos,
)


# ---------------------------------------------------------------------------
# Rasterizer adapter
# ---------------------------------------------------------------------------


class MLXRasterizer:
    """Adapter wrapping mlx_ops.rasterize to match MeshRender's interface."""

    @staticmethod
    def rasterize(pos, tri, resolution, clamp_depth=None, use_depth_prior=0):
        """Rasterize clip-space triangles.

        Args:
            pos: (1, N, 4) or (N, 4) mx.array clip-space vertices.
            tri: (F, 3) mx.array int32 face indices.
            resolution: (height, width) tuple.

        Returns:
            findices: (H, W) int32, 1-indexed face IDs.
            barycentric: (H, W, 3) float32.
        """
        vertices = pos[0] if pos.ndim == 3 else pos
        height, width = int(resolution[0]), int(resolution[1])

        depth_prior = None
        if clamp_depth is not None and use_depth_prior:
            depth_prior = clamp_depth

        findices, barycentric = rasterize_triangles(
            vertices.astype(mx.float32),
            tri.astype(mx.int32),
            width,
            height,
            depth_prior=depth_prior,
        )
        return findices, barycentric

    @staticmethod
    def interpolate(col, findices, barycentric, tri):
        """Interpolate per-vertex attributes at rasterized pixels.

        Args:
            col: (1, N, C) or (N, C) mx.array per-vertex data.
            findices: (H, W) int32.
            barycentric: (H, W, 3) float32.
            tri: (F, 3) int32.

        Returns:
            (1, H, W, C) mx.array.
        """
        attributes = col[0] if col.ndim == 3 else col
        result = mlx_interpolate(
            attributes.astype(mx.float32),
            findices,
            barycentric,
            tri.astype(mx.int32),
        )
        return mx.expand_dims(result, axis=0)


# ---------------------------------------------------------------------------
# Mesh renderer
# ---------------------------------------------------------------------------


class MeshRenderMLX:
    """MLX mesh renderer for normal / position / alpha rendering.

    Drop-in replacement for MeshRender's core render pipeline.
    All internal computation uses MLX arrays; public render methods
    return numpy arrays for downstream compatibility (PIL, cv2, etc.).
    """

    def __init__(
        self,
        camera_distance=1.45,
        camera_type="orth",
        default_resolution=1024,
        texture_size=1024,
        shader_type="face",
        scale_factor=1.15,
        bake_angle_thres=75,
        boundary_scale=2,
        bake_mode="back_sample",
        raster_mode="mlx",
    ):
        self.camera_distance = camera_distance
        self.shader_type = shader_type
        self._default_scale_factor = scale_factor
        self.bake_angle_thres = bake_angle_thres
        self.raster = MLXRasterizer()

        if isinstance(default_resolution, int):
            self.default_resolution = (default_resolution, default_resolution)
        else:
            self.default_resolution = tuple(default_resolution)

        if isinstance(texture_size, int):
            self.texture_size = (texture_size, texture_size)
        else:
            self.texture_size = tuple(texture_size)

        self.bake_unreliable_kernel_size = int(
            (boundary_scale / 512)
            * max(self.default_resolution[0], self.default_resolution[1])
        )

        if camera_type == "orth":
            self.set_orth_scale(1.2)
        elif camera_type == "perspective":
            self.camera_proj_mat = get_perspective_projection_matrix(
                49.13,
                self.default_resolution[1] / self.default_resolution[0],
                0.01,
                100.0,
            )
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")

        # Geometry (set by set_mesh)
        self.vtx_pos = None
        self.pos_idx = None
        self.vtx_uv = None
        self.uv_idx = None
        self.scale_factor = 1.0
        self.mesh_normalize_scale_factor = 1.0
        self.mesh_normalize_scale_center = np.array([[0, 0, 0]])

        # Texture-space data (set by extract_textiles)
        self.tex_position = None
        self.tex_normal = None
        self.tex_grid = None
        self.texture_indices = None

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def set_orth_scale(self, ortho_scale):
        self.ortho_scale = ortho_scale
        self.camera_proj_mat = get_orthographic_projection_matrix(
            left=-ortho_scale * 0.5,
            right=ortho_scale * 0.5,
            bottom=-ortho_scale * 0.5,
            top=ortho_scale * 0.5,
            near=0.1,
            far=100,
        )

    # ------------------------------------------------------------------
    # Mesh loading
    # ------------------------------------------------------------------

    def set_mesh(self, vtx_pos, pos_idx, vtx_uv=None, uv_idx=None,
                 scale_factor=None, auto_center=True):
        """Load mesh geometry from numpy arrays.

        Applies the same coordinate transform and normalization
        as the original MeshRender.set_mesh().
        """
        if scale_factor is None:
            scale_factor = self._default_scale_factor

        vp = np.array(vtx_pos, dtype=np.float32)
        pi = np.array(pos_idx, dtype=np.int32)

        # Coordinate transform: flip X/Y, swap Y/Z
        vp[:, [0, 1]] = -vp[:, [0, 1]]
        vp[:, [1, 2]] = vp[:, [2, 1]].copy()

        if auto_center:
            max_bb = vp.max(axis=0)
            min_bb = vp.min(axis=0)
            center = (max_bb + min_bb) / 2.0
            scale = np.linalg.norm(vp - center, axis=1).max() * 2.0
            vp = (vp - center) * (scale_factor / scale)
            self.scale_factor = scale_factor
            self.mesh_normalize_scale_factor = scale_factor / scale
            self.mesh_normalize_scale_center = center[None, :]
        else:
            self.scale_factor = 1.0
            self.mesh_normalize_scale_factor = 1.0
            self.mesh_normalize_scale_center = np.array([[0, 0, 0]])

        self.vtx_pos = mx.array(vp)
        self.pos_idx = mx.array(pi)

        if vtx_uv is not None and uv_idx is not None:
            uv = np.array(vtx_uv, dtype=np.float32)
            uv[:, 1] = 1.0 - uv[:, 1]
            self.vtx_uv = mx.array(uv)
            self.uv_idx = mx.array(np.array(uv_idx, dtype=np.int32))
            self.extract_textiles()
        else:
            self.vtx_uv = None
            self.uv_idx = None

    def load_mesh(self, mesh, scale_factor=1.15, auto_center=True):
        """Load mesh from file path or trimesh object."""
        import trimesh

        if isinstance(mesh, str):
            mesh = trimesh.load(mesh, process=False, force="mesh")

        vtx_pos = np.array(mesh.vertices, dtype=np.float32)
        pos_idx = np.array(mesh.faces, dtype=np.int32)

        vtx_uv, uv_idx = None, None
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            vtx_uv = np.array(mesh.visual.uv, dtype=np.float32)
            uv_idx = pos_idx  # same topology

        self.set_mesh(
            vtx_pos, pos_idx, vtx_uv=vtx_uv, uv_idx=uv_idx,
            scale_factor=scale_factor, auto_center=auto_center,
        )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _create_view_state(self, elev, azim, camera_distance=None,
                           center=None, resolution=None):
        """Build clip-space vertices and camera-space positions."""
        if camera_distance is None:
            camera_distance = self.camera_distance
        if resolution is None:
            resolution = self.default_resolution

        mv = get_mv_matrix(elev, azim, camera_distance, center)
        pos_camera = transform_pos(mv, self.vtx_pos, keepdim=True)  # (N, 4)
        pos_clip = transform_pos(self.camera_proj_mat, pos_camera)   # (1, N, 4)
        return pos_camera, pos_clip, resolution

    def _rasterize(self, pos_clip, resolution):
        """Run rasterization, return packed rast_out (1, H, W, 4)."""
        findices, bary = self.raster.rasterize(
            pos_clip, self.pos_idx, resolution
        )
        # Pack [bary(3) | findices(1)] -> (1, H, W, 4)
        rast_out = mx.concatenate(
            [bary, mx.expand_dims(findices.astype(mx.float32), axis=-1)],
            axis=-1,
        )
        return mx.expand_dims(rast_out, axis=0)

    def _compute_face_normals(self, triangles):
        """Face normals from (F, 3, 3) triangle vertices."""
        e1 = triangles[:, 1, :] - triangles[:, 0, :]
        e2 = triangles[:, 2, :] - triangles[:, 0, :]
        # Cross product
        n = mx.stack([
            e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1],
            e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
            e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0],
        ], axis=-1)
        norm = mx.sqrt((n * n).sum(axis=-1, keepdims=True) + 1e-12)
        return n / norm

    def _get_normals(self, pos_camera, pos_clip, resolution, use_abs_coor=False):
        """Compute per-pixel normals via face or vertex shading."""
        if use_abs_coor:
            mesh_tris = self.vtx_pos[self.pos_idx]  # (F, 3, 3)
        else:
            pc3 = pos_camera[:, :3] / pos_camera[:, 3:4]
            mesh_tris = pc3[self.pos_idx]

        face_normals = self._compute_face_normals(mesh_tris)
        rast_out = self._rasterize(pos_clip, resolution)

        if self.shader_type == "vertex":
            vertex_normals = self._vertex_normals_from_faces(face_normals)
            normal = self.raster.interpolate(
                mx.expand_dims(vertex_normals, 0),
                rast_out[0, ..., -1].astype(mx.int32),
                rast_out[0, ..., :-1],
                self.pos_idx,
            )
        else:
            # Face shader: direct face-normal lookup per pixel
            tri_ids = rast_out[..., 3].astype(mx.int32)  # (1, H, W)
            mask = (tri_ids > 0).astype(mx.float32)
            tri_ids_0 = mx.maximum(tri_ids - 1, 0)  # 0-indexed
            normal = face_normals[tri_ids_0[0]]  # (H, W, 3)
            normal = normal * mask[0, ..., None]
            normal = mx.expand_dims(normal, 0)  # (1, H, W, 3)

        return normal, rast_out

    def _vertex_normals_from_faces(self, face_normals):
        """Compute vertex normals as average of adjacent face normals."""
        num_verts = self.vtx_pos.shape[0]
        fi = np.array(self.pos_idx)
        fn = np.array(face_normals)
        vn = np.zeros((num_verts, 3), dtype=np.float32)
        counts = np.zeros((num_verts, 1), dtype=np.float32)
        for c in range(3):
            np.add.at(vn, fi[:, c], fn)
            np.add.at(counts, fi[:, c], 1.0)
        counts = np.maximum(counts, 1.0)
        vn = vn / counts
        norms = np.maximum(np.linalg.norm(vn, axis=-1, keepdims=True), 1e-12)
        return mx.array(vn / norms)

    # ------------------------------------------------------------------
    # Public render API
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(image_np, return_type="np"):
        """Convert numpy output to requested format."""
        if return_type == "pl":
            from PIL import Image as PILImage
            img = np.clip(image_np, 0, 1)
            return PILImage.fromarray((img * 255).astype(np.uint8))
        return image_np

    def render_normal(self, elev, azim, resolution=None, bg_color=(1, 1, 1),
                      use_abs_coor=False, camera_distance=None, center=None,
                      return_type="np"):
        """Render surface normals from a given viewpoint.

        Returns:
            (H, W, 3) numpy array (return_type="np") or PIL Image ("pl").
        """
        pos_camera, pos_clip, res = self._create_view_state(
            elev, azim, camera_distance, center, resolution,
        )
        normal, rast_out = self._get_normals(
            pos_camera, pos_clip, res, use_abs_coor,
        )
        visible = mx.clip(rast_out[..., -1:], 0, 1)
        bg = mx.array(list(bg_color), dtype=mx.float32).reshape(1, 1, 1, 3)
        result = normal * visible + bg * (1.0 - visible)
        result = (result + 1.0) * 0.5
        mx.synchronize()
        return self._format_output(np.array(result[0]), return_type)

    def render_position(self, elev, azim, resolution=None, bg_color=(1, 1, 1),
                        camera_distance=None, center=None, return_type="np"):
        """Render world-space positions from a given viewpoint.

        Returns:
            (H, W, 3) numpy array (return_type="np") or PIL Image ("pl").
        """
        _, pos_clip, res = self._create_view_state(
            elev, azim, camera_distance, center, resolution,
        )
        rast_out = self._rasterize(pos_clip, res)
        findices = rast_out[0, ..., -1].astype(mx.int32)
        bary = rast_out[0, ..., :-1]

        tex_pos = 0.5 - self.vtx_pos[:, :3] / self.scale_factor
        position = self.raster.interpolate(
            mx.expand_dims(tex_pos, 0), findices, bary, self.pos_idx,
        )

        visible = mx.clip(rast_out[..., -1:], 0, 1)
        bg = mx.array(list(bg_color), dtype=mx.float32).reshape(1, 1, 1, 3)
        result = position * visible + bg * (1.0 - visible)
        mx.synchronize()
        return self._format_output(np.array(result[0]), return_type)

    def render_alpha(self, elev, azim, resolution=None,
                     camera_distance=None, center=None, return_type="np"):
        """Render face-index map from a given viewpoint.

        Returns face indices (1-indexed, 0=background) matching the original
        MeshRender behavior. Shape: (1, H, W, 1) for return_type="np".
        """
        _, pos_clip, res = self._create_view_state(
            elev, azim, camera_distance, center, resolution,
        )
        rast_out = self._rasterize(pos_clip, res)
        # Return face indices (int), same as original: rast_out[..., -1:].long()
        face_ids = np.array(rast_out[..., -1:])[0].astype(np.int64)
        # Shape (1, H, W, 1) to match original pipeline expectations
        return face_ids[None, ...]

    # ------------------------------------------------------------------
    # Mesh & texture management
    # ------------------------------------------------------------------

    def set_default_render_resolution(self, default_resolution):
        if isinstance(default_resolution, int):
            default_resolution = (default_resolution, default_resolution)
        self.default_resolution = default_resolution

    def set_boundary_unreliable_scale(self, scale):
        self.bake_unreliable_kernel_size = int(
            (scale / 512)
            * max(self.default_resolution[0], self.default_resolution[1])
        )

    def get_face_areas(self, from_one_index=False):
        """Compute area of each triangle face.

        Returns:
            numpy array of face areas. If from_one_index, prepends a zero
            so areas[face_id] works with 1-indexed face IDs.
        """
        vp = np.array(self.vtx_pos)
        fi = np.array(self.pos_idx)
        v0 = vp[fi[:, 0]]
        v1 = vp[fi[:, 1]]
        v2 = vp[fi[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=-1) * 0.5
        if from_one_index:
            areas = np.insert(areas, 0, 0.0)
        return areas

    def set_texture(self, tex, force_set=False):
        """Store diffuse texture (numpy uint8 or float, or PIL Image)."""
        from PIL import Image as PILImage
        if isinstance(tex, PILImage.Image):
            tex = np.array(tex).astype(np.float32) / 255.0
        elif isinstance(tex, np.ndarray) and tex.dtype == np.uint8:
            tex = tex.astype(np.float32) / 255.0
        self.tex = tex

    def get_texture(self):
        """Return diffuse texture as numpy float32 array."""
        return self.tex if hasattr(self, "tex") and self.tex is not None else None

    def set_texture_mr(self, mr, force_set=False):
        """Store metallic-roughness texture."""
        from PIL import Image as PILImage
        if isinstance(mr, PILImage.Image):
            mr = np.array(mr).astype(np.float32) / 255.0
        elif isinstance(mr, np.ndarray) and mr.dtype == np.uint8:
            mr = mr.astype(np.float32) / 255.0
        self.tex_mr = mr

    def get_texture_mr(self):
        metallic, roughness = None, None
        if hasattr(self, "tex_mr") and self.tex_mr is not None:
            mr = self.tex_mr
            metallic = np.repeat(mr[:, :, 0:1], 3, axis=2)
            roughness = np.repeat(mr[:, :, 1:2], 3, axis=2)
        return metallic, roughness

    def get_texture_normal(self):
        if hasattr(self, "tex_normalMap") and self.tex_normalMap is not None:
            return self.tex_normalMap
        return None

    def get_mesh(self, normalize=True):
        """Return mesh geometry as numpy arrays.

        Applies inverse coordinate transform to restore original space.
        """
        vtx_pos = np.array(self.vtx_pos)
        pos_idx = np.array(self.pos_idx)

        if not normalize:
            vtx_pos = vtx_pos / self.mesh_normalize_scale_factor
            vtx_pos = vtx_pos + self.mesh_normalize_scale_center

        # Inverse of set_mesh transform: undo swap Y/Z, then undo flip X/Y
        vtx_pos[:, [1, 2]] = vtx_pos[:, [2, 1]].copy()
        vtx_pos[:, [0, 1]] = -vtx_pos[:, [0, 1]]

        vtx_uv, uv_idx = None, None
        if self.vtx_uv is not None:
            vtx_uv = np.array(self.vtx_uv)
            vtx_uv[:, 1] = 1.0 - vtx_uv[:, 1]
            uv_idx = np.array(self.uv_idx)

        return vtx_pos, pos_idx, vtx_uv, uv_idx

    def save_mesh(self, mesh_path, downsample=False):
        """Save textured mesh to OBJ file."""
        import cv2
        import trimesh

        vtx_pos, pos_idx, vtx_uv, uv_idx = self.get_mesh(normalize=False)
        texture = self.get_texture()

        if texture is not None and downsample:
            h, w = texture.shape[0] // 2, texture.shape[1] // 2
            texture = cv2.resize(texture, (w, h))

        mesh = trimesh.Trimesh(
            vertices=vtx_pos,
            faces=pos_idx,
            process=False,
        )
        if vtx_uv is not None and texture is not None:
            from PIL import Image as PILImage
            tex_img = PILImage.fromarray(
                (np.clip(texture, 0, 1) * 255).astype(np.uint8)
            )
            # Use PBRMaterial so GLB exports tag the baseColorTexture as
            # sRGB. SimpleMaterial doesn't carry color-space metadata and
            # GLB viewers assumed the texture was linear, applying an
            # unwanted gamma curve that darkened / desaturated the mesh
            # relative to the atlas PNG.
            mr_tex_img = None
            mr_atlas = getattr(self, "tex_mr", None)
            if mr_atlas is not None:
                # Repack for glTF: the diffusion MR atlas uses R=metallic,
                # G=roughness (PT convention, see MeshRender.get_texture_mr),
                # but glTF metallicRoughnessTexture expects B=metallic,
                # G=roughness, R ignored (occlusion in some pipelines).
                # Without this swap, glTF viewers read metallic=0 and the
                # mesh looks uniformly dielectric.
                mr_u8 = (np.clip(mr_atlas, 0, 1) * 255).astype(np.uint8)
                if mr_u8.ndim == 2:
                    mr_u8 = np.stack([mr_u8, mr_u8, mr_u8], axis=-1)
                metallic_ch = mr_u8[..., 0]
                roughness_ch = mr_u8[..., 1]
                mr_gltf = np.stack(
                    [np.zeros_like(metallic_ch), roughness_ch, metallic_ch],
                    axis=-1,
                )
                mr_tex_img = PILImage.fromarray(mr_gltf)

            kwargs = dict(
                name="paint_pbr",
                baseColorTexture=tex_img,
            )
            if mr_tex_img is not None:
                kwargs["metallicRoughnessTexture"] = mr_tex_img
                kwargs["metallicFactor"] = 1.0
                kwargs["roughnessFactor"] = 1.0
            else:
                kwargs["metallicFactor"] = 0.0
                kwargs["roughnessFactor"] = 1.0
            material = trimesh.visual.material.PBRMaterial(**kwargs)
            # doubleSided stops backface culling that was making the
            # silhouettes see-through.
            material.doubleSided = True
            visuals = trimesh.visual.TextureVisuals(
                uv=vtx_uv, material=material,
            )
            mesh.visual = visuals

        mesh.export(mesh_path)
        # Expose the in-memory trimesh so callers can export GLB directly.
        # Going OBJ -> trimesh.load -> GLB would drop the PBRMaterial
        # (OBJ/MTL can't carry metallicRoughnessTexture, doubleSided, or
        # sRGB hints) and silently erase our PBR work.
        self._last_exported_mesh = mesh

    # ------------------------------------------------------------------
    # Texture-space operations
    # ------------------------------------------------------------------

    def extract_textiles(self):
        """Rasterize mesh in UV space to build texture-space geometry maps.

        Populates self.tex_position, tex_normal, tex_grid, texture_indices.
        Required by back_project(method="back_sample").
        """
        if self.vtx_uv is None or self.uv_idx is None:
            return

        # Build clip-space coords from UVs: (u, v, 0, 1) * 2 - 1
        vnum = self.vtx_uv.shape[0]
        vtx_uv_clip = mx.concatenate([
            self.vtx_uv,
            mx.zeros((vnum, 1)),
            mx.ones((vnum, 1)),
        ], axis=1) * 2.0 - 1.0
        vtx_uv_clip = mx.expand_dims(vtx_uv_clip, 0)  # (1, V, 4)

        # Rasterize in UV space
        rast_out = self._rasterize(vtx_uv_clip, self.texture_size)
        fi = rast_out[0, ..., -1].astype(mx.int32)
        bary = rast_out[0, ..., :-1]

        # Interpolate world positions in UV space
        position = self.raster.interpolate(
            mx.expand_dims(self.vtx_pos, 0), fi, bary, self.pos_idx,
        )[0]  # (H, W, 3)

        # Compute face normals and per-pixel normals (face shader)
        face_normals = self._compute_face_normals(self.vtx_pos[self.pos_idx])
        tri_ids = rast_out[0, ..., 3].astype(mx.int32)
        mask = tri_ids > 0
        tri_ids_0 = mx.maximum(tri_ids - 1, 0)
        position_normal = face_normals[tri_ids_0]  # (H, W, 3)
        position_normal = position_normal * mask[..., None].astype(mx.float32)

        visible_mask = mx.clip(rast_out[0, ..., -1], 0, 1)  # (H, W)
        mx.synchronize()

        # Extract visible pixels to flat arrays (numpy for indexing)
        vis_np = np.array(visible_mask).reshape(-1)
        mask_flat = vis_np > 0

        pos_np = np.array(position).reshape(-1, 3)
        norm_np = np.array(position_normal).reshape(-1, 3)

        th, tw = self.texture_size
        row, col = np.meshgrid(np.arange(th), np.arange(tw), indexing="ij")
        grid_np = np.stack([row, col], axis=-1).reshape(-1, 2)

        pos_visible = pos_np[mask_flat]
        norm_visible = norm_np[mask_flat]
        grid_visible = grid_np[mask_flat]

        # Add homogeneous w=1
        pos_visible = np.concatenate(
            [pos_visible, np.ones((pos_visible.shape[0], 1), dtype=np.float32)],
            axis=1,
        )

        # Build reverse index: texture pixel -> flat visible index
        texture_indices = np.full(th * tw, -1, dtype=np.int64)
        flat_idx = grid_visible[:, 0] * tw + grid_visible[:, 1]
        texture_indices[flat_idx] = np.arange(len(grid_visible))

        self.tex_position = pos_visible     # (K, 4) float32
        self.tex_normal = norm_visible      # (K, 3) float32
        self.tex_grid = grid_visible        # (K, 2) int
        self.texture_indices = texture_indices.reshape(th, tw)  # (H, W)

        # Per-texel face id (1-indexed, 0 = empty), for per-face bake merging
        self.tex_face_id = np.array(tri_ids).astype(np.int32)  # (H, W)

    def uv_feature_map(self, vert_feat):
        """Map per-vertex features to UV texture space.

        Args:
            vert_feat: (N, C) mx.array of per-vertex features.

        Returns:
            (H, W, C) mx.array feature map.
        """
        vtx_uv_clip = mx.concatenate([
            self.vtx_uv * 2 - 1,
            mx.zeros((self.vtx_uv.shape[0], 1)),
            mx.ones((self.vtx_uv.shape[0], 1)),
        ], axis=1)
        vtx_uv_clip = mx.expand_dims(vtx_uv_clip, 0)

        rast_out = self._rasterize(vtx_uv_clip, self.texture_size)
        fi = rast_out[0, ..., -1].astype(mx.int32)
        bary = rast_out[0, ..., :-1]

        feat_map = self.raster.interpolate(
            mx.expand_dims(vert_feat, 0), fi, bary, self.uv_idx,
        )
        return feat_map[0]  # (H, W, C)

    # ------------------------------------------------------------------
    # Texture baking
    # ------------------------------------------------------------------

    @staticmethod
    def _render_sketch_from_depth(depth_image_np):
        """Edge detection on depth map (CPU, cv2)."""
        import cv2
        depth_u8 = (depth_image_np * 255).astype(np.uint8)
        edges = cv2.Canny(depth_u8, 30, 80)
        return (edges.astype(np.float32) / 255.0)[..., None]  # (H, W, 1)

    @staticmethod
    def _erode_boundary(mask_np, sketch_np, kernel_size):
        """Erode visible mask near silhouette boundaries (CPU, cv2)."""
        import cv2
        if kernel_size <= 0:
            return mask_np
        ks = kernel_size * 2 + 1
        kernel = np.ones((ks, ks), dtype=np.float32)

        # Erode visible mask: pixels near background become unreliable
        inv = 1.0 - mask_np[..., 0]
        eroded = cv2.filter2D(inv, -1, kernel)
        mask_np = mask_np * (eroded <= 0).astype(np.float32)[..., None]

        # Also exclude pixels near depth edges
        sketch_dilated = cv2.filter2D(sketch_np[..., 0], -1, kernel)
        mask_np = mask_np * (sketch_dilated < 0.5).astype(np.float32)[..., None]
        return mask_np

    def back_project(self, image, elev, azim, camera_distance=None,
                     center=None):
        """Back-project a rendered image onto UV texture space.

        Uses the "back_sample" method: projects texture-space positions
        into image space and bilinearly samples colors.

        Args:
            image: (H, W, C) numpy array or PIL Image, values in [0, 1].
            elev: Camera elevation in degrees.
            azim: Camera azimuth in degrees.

        Returns:
            texture: (H, W, C) numpy array in UV space.
            cos_map: (H, W, 1) numpy array cosine weights.
            boundary_map: (H, W, 1) numpy array boundary mask.
        """
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            image = np.array(image).astype(np.float32) / 255.0
        elif not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.ndim == 2:
            image = image[..., None]
        image = image.astype(np.float32)

        resolution = image.shape[:2]  # (H, W)
        channel = image.shape[2]
        th, tw = self.texture_size

        if camera_distance is None:
            camera_distance = self.camera_distance

        # --- Camera setup ---
        proj = self.camera_proj_mat
        mv = get_mv_matrix(elev, azim, camera_distance, center)
        pos_camera = transform_pos(mv, self.vtx_pos, keepdim=True)  # (N, 4)
        pos_clip = transform_pos(self.camera_proj_mat, pos_camera)   # (1, N, 4)

        pc3 = pos_camera[:, :3] / pos_camera[:, 3:4]

        # --- Face normals in camera space ---
        mesh_tris = pc3[self.pos_idx]  # (F, 3, 3)
        face_normals = self._compute_face_normals(mesh_tris)

        # --- Rasterize from camera view ---
        rast_out = self._rasterize(pos_clip, resolution)
        fi = rast_out[0, ..., -1].astype(mx.int32)
        bary = rast_out[0, ..., :-1]

        visible = mx.clip(rast_out[0, ..., -1:], 0, 1)  # (H, W, 1)

        # Per-pixel normals (face shader)
        tri_ids = fi
        tri_mask = (tri_ids > 0).astype(mx.float32)
        tri_ids_0 = mx.maximum(tri_ids - 1, 0)
        normal = face_normals[tri_ids_0] * tri_mask[..., None]  # (H, W, 3)

        # Depth
        depth_attr = pc3[:, 2:3].reshape(1, -1, 1)
        depth = self.raster.interpolate(depth_attr, fi, bary, self.pos_idx)
        depth = depth[0]  # (H, W, 1)
        mx.synchronize()

        visible_np = np.array(visible)
        normal_np = np.array(normal)
        depth_np = np.array(depth)

        # Depth normalization for sketch
        vis_flat = visible_np.reshape(-1) > 0
        if vis_flat.sum() == 0:
            return (np.zeros((*self.texture_size, channel), dtype=np.float32),
                    np.zeros((*self.texture_size, 1), dtype=np.float32),
                    np.zeros((*self.texture_size, 1), dtype=np.float32))
        d_vis = depth_np.reshape(-1)[vis_flat]
        d_max, d_min = d_vis.max(), d_vis.min()
        depth_norm = (depth_np - d_min) / max(d_max - d_min, 1e-8)
        depth_image = depth_norm * visible_np

        sketch_np = self._render_sketch_from_depth(depth_image[..., 0])

        # Cosine weighting (view angle vs surface normal)
        lookat = np.array([0, 0, -1], dtype=np.float32)
        normal_flat = normal_np.reshape(-1, 3)
        cos_flat = (normal_flat * lookat).sum(axis=-1)
        cos_np = cos_flat.reshape(resolution[0], resolution[1], 1)
        cos_thres = np.cos(np.radians(self.bake_angle_thres))
        cos_np[cos_np < cos_thres] = 0.0

        # Boundary erosion
        visible_np = self._erode_boundary(
            visible_np, sketch_np, self.bake_unreliable_kernel_size,
        )
        cos_np[visible_np[..., 0] == 0] = 0.0

        # --- Back-sample: project tex_position to image space ---
        if self.tex_position is None:
            raise RuntimeError("extract_textiles() must be called first")

        img_proj = np.array([
            [proj[0, 0], 0, 0, 0],
            [0, proj[1, 1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        mv_f = mv.astype(np.float32)
        tex_pos = self.tex_position  # (K, 4)
        v_proj = tex_pos @ mv_f.T @ img_proj  # (K, 4)

        # Check which tex pixels project inside the image
        inner = (
            (v_proj[:, 0] >= -1.0) & (v_proj[:, 0] <= 1.0)
            & (v_proj[:, 1] >= -1.0) & (v_proj[:, 1] <= 1.0)
        )
        inner_idx = np.where(inner)[0]

        img_x = np.clip(
            ((np.clip(v_proj[:, 0], -1, 1) * 0.5 + 0.5) * resolution[0]).astype(np.int64),
            0, resolution[0] - 1,
        )
        img_y = np.clip(
            ((np.clip(v_proj[:, 1], -1, 1) * 0.5 + 0.5) * resolution[1]).astype(np.int64),
            0, resolution[1] - 1,
        )

        indices = img_y * resolution[0] + img_x
        depth_flat = depth_np.reshape(-1)
        vis_mask_flat = visible_np.reshape(-1)
        cos_flat = cos_np.reshape(-1)

        sampled_z = depth_flat[indices]
        sampled_m = vis_mask_flat[indices]
        sampled_w = cos_flat[indices]
        v_z = v_proj[:, 2]

        depth_thres = 3e-3
        valid_mask = (np.abs(v_z - sampled_z) < depth_thres) & (sampled_m * sampled_w > 0)
        valid_idx = np.where(valid_mask)[0]
        valid_idx = np.intersect1d(valid_idx, inner_idx)

        # Bilinear sampling from image
        wx = ((v_proj[:, 0] * 0.5 + 0.5) * resolution[0] - img_x)[valid_idx].reshape(-1, 1)
        wy = ((v_proj[:, 1] * 0.5 + 0.5) * resolution[1] - img_y)[valid_idx].reshape(-1, 1)
        ix = img_x[valid_idx]
        iy = img_y[valid_idx]
        ix_r = np.clip(ix + 1, 0, resolution[0] - 1)
        iy_r = np.clip(iy + 1, 0, resolution[1] - 1)

        rgb = image.reshape(-1, channel)
        i00 = iy * resolution[0] + ix
        i10 = iy * resolution[0] + ix_r
        i01 = iy_r * resolution[0] + ix
        i11 = iy_r * resolution[0] + ix_r
        sampled_rgb = (
            (rgb[i00] * (1 - wx) + rgb[i10] * wx) * (1 - wy)
            + (rgb[i01] * (1 - wx) + rgb[i11] * wx) * wy
        )

        sampled_b = sketch_np.reshape(-1)[indices[valid_idx]]
        sampled_w_valid = sampled_w[valid_idx]

        # Write to texture space
        texture = np.zeros((th * tw, channel), dtype=np.float32)
        cos_map = np.zeros(th * tw, dtype=np.float32)
        boundary_map = np.zeros(th * tw, dtype=np.float32)

        tex_grid = self.tex_grid[valid_idx]
        tex_flat_idx = tex_grid[:, 0] * tw + tex_grid[:, 1]
        texture[tex_flat_idx] = sampled_rgb
        cos_map[tex_flat_idx] = sampled_w_valid
        boundary_map[tex_flat_idx] = sampled_b

        return (
            texture.reshape(th, tw, channel),
            cos_map.reshape(th, tw, 1),
            boundary_map.reshape(th, tw, 1),
        )

    def fast_bake_texture(self, textures, cos_maps, mode="weighted"):
        """Merge multiple view textures into a single UV atlas.

        Three modes (PT parity is default):
          - ``"weighted"`` (default, PyTorch parity): cosine-weighted
            average of all views. Smooth across face boundaries; matches
            MeshRender.fast_bake_texture exactly.
          - ``"face_wta"``: pick one view per UV face. Sharper but creates
            visible discontinuities at mesh-face seams where adjacent
            faces pick different winning views.
          - ``"wta"``: per-texel argmax. Fastest, stripiest.

        Args:
            textures: list of (H, W, C) numpy arrays.
            cos_maps: list of (H, W, 1) numpy arrays (already weighted).
            mode: "wta" or "weighted".

        Returns:
            texture_merge: (H, W, C) numpy array.
            trust_map: (H, W, 1) boolean numpy array.
        """
        if not textures:
            raise ValueError("fast_bake_texture: no textures provided")

        if mode == "face_wta":
            # Per-face WTA: aggregate cos per (face, view), pick best view per
            # face, then bake all of that face's texels from that view.
            if getattr(self, "tex_face_id", None) is None:
                # Fall back to per-texel WTA if face ids aren't available
                mode = "wta"
            else:
                stack_t = np.stack(textures, axis=0)              # (V, H, W, C)
                stack_c = np.stack([c[..., 0] for c in cos_maps], axis=0)  # (V, H, W)
                face_ids = self.tex_face_id                        # (H, W) 1-indexed
                V = stack_c.shape[0]
                n_faces = int(face_ids.max())
                # Sum cos per (view, face)
                flat_face = face_ids.reshape(-1)
                valid = flat_face > 0
                face_idx = (flat_face[valid] - 1).astype(np.int64)
                stack_c_flat = stack_c.reshape(V, -1)[:, valid]    # (V, K)
                cos_per_face = np.zeros((V, n_faces), dtype=np.float32)
                for v in range(V):
                    np.add.at(cos_per_face[v], face_idx, stack_c_flat[v])
                best_view_per_face = cos_per_face.argmax(axis=0)   # (n_faces,)
                # Per-texel best view from face lookup
                best_per_texel = np.zeros_like(face_ids, dtype=np.int64)
                best_per_texel[face_ids > 0] = best_view_per_face[face_ids[face_ids > 0] - 1]
                # Painted = any view contributed at this texel
                best_c = stack_c.max(axis=0)
                H, W = face_ids.shape
                ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                merged = stack_t[best_per_texel, ii, jj]
                merged[best_c == 0] = 0
                return merged, (best_c > 0)[..., None]

        if mode == "wta":
            stack_t = np.stack(textures, axis=0)
            stack_c = np.stack([c[..., 0] for c in cos_maps], axis=0)
            best = stack_c.argmax(axis=0)
            best_c = stack_c.max(axis=0)
            H, W = best.shape
            ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            merged = stack_t[best, ii, jj]
            merged[best_c == 0] = 0
            return merged, (best_c > 0)[..., None]

        # PyTorch-parity weighted average path
        channel = textures[0].shape[-1]
        th, tw = self.texture_size
        texture_merge = np.zeros((th, tw, channel), dtype=np.float32)
        trust_map = np.zeros((th, tw, 1), dtype=np.float32)
        for texture, cos_map in zip(textures, cos_maps):
            view_sum = (cos_map > 0).sum()
            if view_sum == 0:
                continue
            painted_sum = ((cos_map > 0) & (trust_map > 0)).sum()
            if painted_sum / view_sum > 0.99:
                continue
            texture_merge += texture * cos_map
            trust_map += cos_map
        texture_merge = texture_merge / np.maximum(trust_map, 1e-8)
        return texture_merge, trust_map > 1e-8

    def bake_texture(self, colors, elevs, azims, camera_distance=None,
                     center=None, exp=6, weights=None):
        """Bake multiple views into a single UV texture.

        Args:
            colors: list of images (numpy or PIL).
            elevs: list of elevation angles.
            azims: list of azimuth angles.
            exp: exponent for cosine weighting.
            weights: optional per-view weights.

        Returns:
            texture_merge: (H, W, C) numpy array.
            trust_map: (H, W, 1) boolean numpy array.
        """
        from PIL import Image as PILImage

        if weights is None:
            weights = [1.0] * len(colors)

        textures = []
        cos_maps = []
        for color, elev, azim, weight in zip(colors, elevs, azims, weights):
            if isinstance(color, PILImage.Image):
                color = np.array(color).astype(np.float32) / 255.0
            texture, cos_map, _ = self.back_project(
                color, elev, azim, camera_distance, center,
            )
            cos_map = weight * (cos_map ** exp)
            textures.append(texture)
            cos_maps.append(cos_map)

        return self.fast_bake_texture(textures, cos_maps)

    def uv_inpaint(self, texture, mask, method="NS", vertex_inpaint=True):
        """Inpaint missing regions in UV texture.

        Two-pass pipeline matching PyTorch's bake flow:
          1. Mesh-aware vertex-color propagation + face-bary rasterization
             (fills most of the UV islands using 3D adjacency).
          2. Gutter fill for the remaining empty texels.

        For step 2 the default ``"edt"`` mode copies each gutter texel from
        its nearest painted texel via ``scipy.ndimage.distance_transform_edt``.
        It's deterministic and never blends colors across distant UV islands —
        unlike ``"NS"`` (cv2.INPAINT_NS) which diffuses and tends to mix
        unrelated islands together when they happen to be 2D-close in atlas
        space.

        Args:
            texture: (H, W, C) numpy array in [0, 1].
            mask: (H, W) or (H, W, 1) numpy uint8, 255=keep, 0=inpaint.
            method: "edt" (default, EDT nearest-fill) or "NS" (cv2 NS).
            vertex_inpaint: Run the mesh-aware propagation pass first.

        Returns:
            (H, W, C) numpy uint8 array.
        """
        if isinstance(texture, mx.array):
            texture = np.array(texture)

        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        tex_f = np.clip(texture, 0, 1).astype(np.float32)

        if vertex_inpaint and self.vtx_uv is not None and self.uv_idx is not None:
            from .mesh_inpaint_py import mesh_vertex_inpaint
            tex_f, mask = mesh_vertex_inpaint(
                tex_f, mask,
                np.asarray(self.vtx_pos), np.asarray(self.vtx_uv),
                np.asarray(self.pos_idx), np.asarray(self.uv_idx),
            )

        if method == "edt":
            from scipy.ndimage import distance_transform_edt
            painted = mask > 0
            if painted.any():
                idx = distance_transform_edt(
                    ~painted, return_distances=False, return_indices=True,
                )
                tex_f = tex_f[idx[0], idx[1]]
            return (np.clip(tex_f, 0, 1) * 255).astype(np.uint8)

        # cv2.INPAINT_NS for near-field fill (PT parity), then an explicit
        # EDT edge-padding pass for UV gutter. NS's 3-px radius is smaller
        # than typical UV-island spacing in a 2048^2 atlas, so 3D viewers
        # doing bilinear texture sampling across island boundaries read
        # the NS-smoothed (near-black) gutter and display thin wireframe
        # lines on the mesh. EDT pads every unpainted texel with the
        # nearest painted color — this is the standard post-bake step in
        # production UV pipelines and never affects texels that are
        # visible in 3D (painted).
        import cv2
        from scipy.ndimage import distance_transform_edt
        texture_u8 = (np.clip(tex_f, 0, 1) * 255).astype(np.uint8)
        inpainted = cv2.inpaint(texture_u8, 255 - mask, 3, cv2.INPAINT_NS)
        painted = mask > 0
        if painted.any() and (~painted).any():
            idx = distance_transform_edt(
                ~painted, return_distances=False, return_indices=True,
            )
            inpainted = inpainted[idx[0], idx[1]]
        return inpainted
