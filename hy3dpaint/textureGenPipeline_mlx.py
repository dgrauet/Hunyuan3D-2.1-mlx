"""Hybrid texture generation pipeline: MLX renderer + PyTorch ML models.

Uses MeshRenderMLX for all rendering and texture baking (Metal GPU),
while keeping multiview diffusion and super-resolution in PyTorch.
The interface between them is PIL Images.
"""

import copy
import os
import warnings

import numpy as np
import trimesh
from PIL import Image
from typing import List

from DifferentiableRenderer.mesh_render_mlx import MeshRenderMLX
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
from utils.pipeline_utils_mlx import ViewProcessorMLX
from utils.simplify_mesh_utils import remesh_mesh
from utils.uvwrap_utils import mesh_uv_wrap

warnings.filterwarnings("ignore")


class Hunyuan3DPaintConfigMLX:
    def __init__(self, max_num_view, resolution):
        self.device = "cuda"  # for PyTorch ML models only

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "mlx"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        # View selection candidates
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipelineMLX:
    """Texture generation pipeline with MLX rendering backend.

    The rendering, rasterization, and texture baking run on Apple Silicon
    via MLX Metal kernels. The ML models (multiview diffusion, super-res)
    remain in PyTorch and communicate via PIL Images.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else Hunyuan3DPaintConfigMLX(
            max_num_view=6, resolution=1024,
        )
        self.models = {}

        self.render = MeshRenderMLX(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
        )
        self.view_processor = ViewProcessorMLX(self.config, self.render)
        self.load_models()

    def load_models(self):
        """Load PyTorch ML models (diffusion + super-res)."""
        try:
            import torch
            torch.cuda.empty_cache()
            from utils.image_super_utils import imageSuperNet
            from utils.multiview_utils import multiviewDiffusionNet
            self.models["super_model"] = imageSuperNet(self.config)
            self.models["multiview_model"] = multiviewDiffusionNet(self.config)
            print("ML models loaded (PyTorch).")
        except (ImportError, RuntimeError) as e:
            print(f"ML models not loaded ({e}). "
                  "Pipeline can still run render/bake with pre-generated images.")

    def __call__(self, mesh_path=None, image_path=None,
                 output_mesh_path=None, use_remesh=True, save_glb=True):
        """Generate texture for 3D mesh using multiview diffusion.

        Args:
            mesh_path: Path to input mesh.
            image_path: Path to reference image or PIL Image.
            output_mesh_path: Output path for textured mesh.
            use_remesh: Whether to remesh before texturing.
            save_glb: Whether to also save as GLB.

        Returns:
            Path to the output mesh file.
        """
        # --- Prepare image prompt ---
        if isinstance(image_path, str):
            image_prompt = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image_prompt = image_path
        else:
            image_prompt = image_path

        if not isinstance(image_prompt, list):
            image_prompt = [image_prompt]

        # --- Process mesh ---
        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, "textured_mesh.obj")

        mesh = trimesh.load(processed_mesh_path)
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        # --- View selection ---
        selected_elevs, selected_azims, selected_weights = \
            self.view_processor.bake_view_selection(
                self.config.candidate_camera_elevs,
                self.config.candidate_camera_azims,
                self.config.candidate_view_weights,
                self.config.max_selected_view_num,
            )

        # --- Render reference maps (MLX, Metal GPU) ---
        normal_maps = self.view_processor.render_normal_multiview(
            selected_elevs, selected_azims, use_abs_coor=True,
        )
        position_maps = self.view_processor.render_position_multiview(
            selected_elevs, selected_azims,
        )

        # --- Style prep ---
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image.convert("RGB"))

        # --- Multiview diffusion (PyTorch, CUDA GPU) ---
        if "multiview_model" not in self.models:
            raise RuntimeError(
                "Multiview diffusion model not loaded. "
                "Install PyTorch with CUDA support."
            )

        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt="high quality",
            custom_view_size=self.config.resolution,
            resize_input=True,
        )

        # --- Super-resolution (PyTorch, CUDA GPU) ---
        enhance_images = {
            "albedo": copy.deepcopy(multiviews_pbr["albedo"]),
            "mr": copy.deepcopy(multiviews_pbr["mr"]),
        }

        if "super_model" in self.models:
            for i in range(len(enhance_images["albedo"])):
                enhance_images["albedo"][i] = self.models["super_model"](
                    enhance_images["albedo"][i]
                )
                enhance_images["mr"][i] = self.models["super_model"](
                    enhance_images["mr"][i]
                )

        # --- Texture baking (MLX, Metal GPU) ---
        render_size = self.config.render_size
        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (render_size, render_size),
            )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize(
                (render_size, render_size),
            )

        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"],
            selected_elevs, selected_azims, selected_weights,
        )
        mask_np = (mask[..., 0] * 255).astype(np.uint8)

        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"],
            selected_elevs, selected_azims, selected_weights,
        )
        mask_mr_np = (mask_mr[..., 0] * 255).astype(np.uint8)

        # --- Inpainting (OpenCV, CPU) ---
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture)

        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(
                texture_mr, mask_mr_np,
            )
            self.render.set_texture_mr(texture_mr)

        # --- Save ---
        self.render.save_mesh(output_mesh_path, downsample=True)

        if save_glb:
            glb_path = output_mesh_path.replace(".obj", ".glb")
            convert_obj_to_glb(output_mesh_path, glb_path)

        return output_mesh_path
