"""MLX-compatible ViewProcessor for Hunyuan3D texture pipeline.

Drop-in replacement for ViewProcessor that works with MeshRenderMLX.
Uses numpy instead of torch for tensor operations.
"""

import numpy as np


class ViewProcessorMLX:
    """Orchestrates multiview rendering and texture baking using MeshRenderMLX."""

    def __init__(self, config, render):
        self.config = config
        self.render = render

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        return [
            self.render.render_normal(elev, azim, use_abs_coor=use_abs_coor,
                                     return_type="pl")
            for elev, azim in zip(camera_elevs, camera_azims)
        ]

    def render_position_multiview(self, camera_elevs, camera_azims):
        return [
            self.render.render_position(elev, azim, return_type="pl")
            for elev, azim in zip(camera_elevs, camera_azims)
        ]

    def bake_view_selection(self, candidate_camera_elevs, candidate_camera_azims,
                            candidate_view_weights, max_selected_view_num):
        """Select optimal camera views for texture baking via greedy set cover."""
        original_resolution = self.render.default_resolution
        self.render.set_default_render_resolution(1024)

        selected_elevs = []
        selected_azims = []
        selected_weights = []

        face_areas = self.render.get_face_areas(from_one_index=True)
        total_area = face_areas.sum()
        face_area_ratios = face_areas / total_area

        n_candidates = len(candidate_camera_elevs)
        self.render.set_boundary_unreliable_scale(2)

        # Render alpha (face indices) for all candidates
        viewed_tri_idxs = []
        viewed_masks = []
        for elev, azim in zip(candidate_camera_elevs, candidate_camera_azims):
            tri_idx = self.render.render_alpha(elev, azim, return_type="np")
            viewed_tri_idxs.append(set(np.unique(tri_idx.flatten()).tolist()))
            viewed_masks.append(tri_idx[0, :, :, 0] > 0)

        is_selected = [False] * n_candidates
        total_viewed = set()

        # Always select first 6 (cardinal directions)
        for idx in range(min(6, n_candidates)):
            selected_elevs.append(candidate_camera_elevs[idx])
            selected_azims.append(candidate_camera_azims[idx])
            selected_weights.append(candidate_view_weights[idx])
            is_selected[idx] = True
            total_viewed.update(viewed_tri_idxs[idx])

        # Greedy: add views that cover the most new area
        for _ in range(max_selected_view_num - len(selected_weights)):
            max_inc = 0.0
            max_idx = -1

            for idx in range(n_candidates):
                if is_selected[idx]:
                    continue
                new_tris = viewed_tri_idxs[idx] - total_viewed
                if not new_tris:
                    continue
                new_area = face_area_ratios[list(new_tris)].sum()
                if new_area > max_inc:
                    max_inc = new_area
                    max_idx = idx

            if max_inc > 0.01 and max_idx >= 0:
                is_selected[max_idx] = True
                selected_elevs.append(candidate_camera_elevs[max_idx])
                selected_azims.append(candidate_camera_azims[max_idx])
                selected_weights.append(candidate_view_weights[max_idx])
                total_viewed.update(viewed_tri_idxs[max_idx])
            else:
                break

        self.render.set_default_render_resolution(original_resolution)
        return selected_elevs, selected_azims, selected_weights

    def bake_from_multiview(self, views, camera_elevs, camera_azims,
                            view_weights):
        """Back-project multiple views and merge into a single UV texture."""
        textures = []
        cos_maps = []

        for view, elev, azim, weight in zip(views, camera_elevs, camera_azims,
                                            view_weights):
            texture, cos_map, _ = self.render.back_project(view, elev, azim)
            cos_map = weight * (cos_map ** self.config.bake_exp)
            textures.append(texture)
            cos_maps.append(cos_map)
            # Incrementally merge (matches original behavior)
            merged, trust = self.render.fast_bake_texture(textures, cos_maps)

        return merged, trust

    def texture_inpaint(self, texture, mask, default=None):
        """Inpaint missing texture regions.

        Args:
            texture: numpy float32 array or will be converted.
            mask: numpy uint8 mask (255=keep, 0=inpaint).
            default: optional fill value for masked regions.

        Returns:
            numpy float32 texture.
        """
        if default is not None:
            bool_mask = mask.astype(bool) if mask.dtype != bool else mask
            texture[~bool_mask] = default
            return texture

        texture_np = self.render.uv_inpaint(texture, mask)
        return texture_np.astype(np.float32) / 255.0
