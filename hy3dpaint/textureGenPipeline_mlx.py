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
from utils.pipeline_utils_mlx import ViewProcessorMLX


def _convert_obj_to_glb(obj_path: str, glb_path: str) -> bool:
    """Convert OBJ + MTL + textures to GLB.

    Tries Blender (mesh_utils.convert_obj_to_glb) first for highest fidelity,
    falls back to trimesh (pure Python, no CUDA/Blender dependency) so the
    pipeline runs on Apple Silicon without `bpy`.
    """
    try:
        from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
        return convert_obj_to_glb(obj_path, glb_path)
    except ImportError:
        pass

    import trimesh
    mesh = trimesh.load(obj_path, process=False)
    mesh.export(glb_path)
    return os.path.exists(glb_path)
try:
    from utils.simplify_mesh_utils import remesh_mesh
except ImportError:
    remesh_mesh = None  # pymeshlab unavailable; require use_remesh=False
try:
    from utils.uvwrap_utils import mesh_uv_wrap
except ImportError:
    mesh_uv_wrap = None  # xatlas unavailable; require pre-wrapped mesh

warnings.filterwarnings("ignore")


class Hunyuan3DPaintConfigMLX:
    def __init__(self, max_num_view, resolution):
        self.device = "cuda"  # for PyTorch ML models only

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        # Fully-MLX diffusion (Apple Silicon, no CUDA).
        # When True, loads HunyuanPaintModelMLX from mlx_weights_source and
        # skips the PyTorch multiview model / super-resolution.
        #
        # mlx_weights_source may be either a HuggingFace repo ID (auto-
        # downloaded via huggingface_hub) or an absolute local path. The
        # env var HUNYUAN3D_MLX_WEIGHTS_DIR overrides this value.
        self.use_mlx_diffusion = True
        self.mlx_weights_source = "dgrauet/hunyuan3d-2.1-mlx"
        self.mlx_num_inference_steps = 15
        self.mlx_guidance_scale = 3.0
        self.mlx_seed = 42

        # Fully-MLX super-resolution (RealESRGAN x4, ~17M params).
        # Upscales each 512^2 generated view to 2048^2 before baking so
        # the UV atlas keeps detail. Disable for faster iteration.
        self.use_mlx_super_res = True
        self.mlx_super_res_filename = "realesrgan_x4plus.safetensors"

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
        """Load diffusion model.

        MLX path: HunyuanPaintModelMLX on Apple Silicon.
        PyTorch path: diffusion + super-res on CUDA (legacy).
        """
        if self.config.use_mlx_diffusion:
            from hunyuanpaintpbr_mlx.load_model import HunyuanPaintModelMLX
            self.models["mlx_model"] = HunyuanPaintModelMLX.from_pretrained(
                self.config.mlx_weights_source,
            )
            print("MLX diffusion model loaded.")

            if getattr(self.config, "use_mlx_super_res", False):
                from utils.image_super_utils_mlx import imageSuperNetMLX
                sr_path = self._resolve_super_res_weights()
                self.models["mlx_super"] = imageSuperNetMLX(sr_path)
                print(f"MLX super-resolution loaded ({sr_path}).")
            return

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

    def _resolve_super_res_weights(self) -> str:
        """Resolve RealESRGAN MLX weights path.

        Reuses the HF repo (or local dir) used for the diffusion weights so a
        single ``HUNYUAN3D_MLX_WEIGHTS_DIR`` override covers everything.
        """
        src = self.config.mlx_weights_source
        filename = self.config.mlx_super_res_filename
        env_override = os.environ.get("HUNYUAN3D_MLX_WEIGHTS_DIR")
        if env_override:
            return os.path.join(env_override, filename)
        if os.path.isdir(src):
            return os.path.join(src, filename)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=src, filename=filename)

    def _run_multiview_mlx(self, image_style, normal_maps, position_maps,
                            camera_azims):
        """Run MLX multiview diffusion.

        Args:
            image_style: list of PIL reference images (we use only the first).
            normal_maps: list of PIL normal maps (one per view).
            position_maps: list of PIL position maps.
            camera_azims: list of azimuth angles.

        Returns:
            dict with 'albedo' and 'mr', each a list of PIL images.
        """
        from hunyuanpaintpbr_mlx.inference import generate_multiview_pbr

        ref_np = np.array(image_style[0].convert("RGB"))

        def _pil_to_np_f32(img):
            """Convert a PIL image to (H, W, 3) float32 in [0, 1]."""
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            return arr

        n_maps = [_pil_to_np_f32(m) for m in normal_maps]
        p_maps = [_pil_to_np_f32(m) for m in position_maps]

        result = generate_multiview_pbr(
            model=self.models["mlx_model"],
            normal_maps=n_maps,
            position_maps=p_maps,
            reference_image=ref_np,
            camera_azims=camera_azims,
            num_inference_steps=self.config.mlx_num_inference_steps,
            guidance_scale=self.config.mlx_guidance_scale,
            view_size=self.config.resolution,
            seed=self.config.mlx_seed,
        )

        return {
            "albedo": [Image.fromarray(a) for a in result["albedo"]],
            "mr": [Image.fromarray(m) for m in result["mr"]],
        }

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
        # trimesh.load can return a Scene for .glb/.gltf; flatten to a single
        # Trimesh so we can inspect UVs (a Scene wrapper hides them).
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        # Only UV-wrap if the mesh has no UVs (xatlas creates fragmented
        # islands; a hand-authored UV layout produces much cleaner textures).
        uv = getattr(getattr(mesh, "visual", None), "uv", None)
        needs_wrap = uv is None or len(uv) == 0
        if needs_wrap:
            if mesh_uv_wrap is None:
                raise RuntimeError(
                    "Mesh has no UVs and `xatlas` is not installed. "
                    "Install xatlas or provide a pre-wrapped mesh."
                )
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

        # --- Multiview diffusion ---
        if self.config.use_mlx_diffusion:
            multiviews_pbr = self._run_multiview_mlx(
                image_style, normal_maps, position_maps, selected_azims,
            )
            enhance_images = {
                "albedo": list(multiviews_pbr["albedo"]),
                "mr": list(multiviews_pbr["mr"]),
            }

            # RealESRGAN x4 super-resolution per view (MLX)
            if "mlx_super" in self.models:
                sr = self.models["mlx_super"]
                for i in range(len(enhance_images["albedo"])):
                    enhance_images["albedo"][i] = sr(enhance_images["albedo"][i])
                    enhance_images["mr"][i] = sr(enhance_images["mr"][i])
        else:
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

            # Super-resolution (CUDA-only)
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
            _convert_obj_to_glb(output_mesh_path, glb_path)

        return output_mesh_path
