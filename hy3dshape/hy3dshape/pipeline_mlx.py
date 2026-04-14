"""Shape generation pipeline in MLX.

MLX port of Hunyuan3DDiTFlowMatchingPipeline from pipelines.py.
Orchestrates: DINOv2 encoding -> DiT denoising -> ShapeVAE decoding -> marching cubes.
"""

import gc
import json
import os
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .schedulers_mlx import FlowMatchEulerDiscreteScheduler
from .models.conditioner_mlx import ImageEncoder
from .models.denoisers.hunyuandit_mlx import HunYuanDiTPlain
from .models.autoencoders.model_mlx import ShapeVAEDecoder


def _materialize(*args):
    """Trigger MLX lazy evaluation to materialize computation graphs.
    This is the standard MLX pattern -- mx.eval forces pending computations."""
    mx.eval(*args)  # noqa: S307 - mx.eval is MLX lazy-eval trigger, not Python eval


class ShapePipeline:
    """End-to-end shape generation pipeline.

    Encodes an image with DINOv2, denoises shape latents with DiT
    using flow matching, then decodes to a mesh via ShapeVAE.

    Args:
        image_encoder: DINOv2 image encoder.
        dit: HunYuanDiTPlain denoiser.
        vae: ShapeVAEDecoder.
        scheduler: FlowMatchEulerDiscreteScheduler.
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        dit: HunYuanDiTPlain,
        vae: ShapeVAEDecoder,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        self.image_encoder = image_encoder
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler

    def preprocess_image(self, image: Union[str, Image.Image, mx.array]) -> mx.array:
        """Load and preprocess an image for DINOv2.

        Handles: file path, PIL Image, or pre-loaded mx.array.
        Resizes to the encoder's expected image_size and converts to channels-last.

        Args:
            image: Input image (path, PIL Image, or mx.array).

        Returns:
            (1, H, W, 3) normalized image tensor in channels-last format.
        """
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            # Convert to RGB, remove alpha
            if image.mode == "RGBA":
                # Composite on white background
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Resize to encoder's expected size
            target_size = self.image_encoder.image_size
            image = image.resize((target_size, target_size), Image.BILINEAR)

            # Convert to numpy then mx array: (H, W, 3) float32 in [-1, 1]
            arr = np.array(image, dtype=np.float32) / 127.5 - 1.0
            return mx.array(arr[None])  # (1, H, W, 3)

        # Already mx.array
        return image

    def __call__(
        self,
        image: Union[str, Image.Image, mx.array],
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        octree_resolution: int = 256,
        box_v: float = 1.01,
        mc_level: float = 0.0,
        num_chunks: int = 10000,
        seed: Optional[int] = None,
    ):
        """Run the full shape generation pipeline.

        Args:
            image: Input image (path, PIL Image, or tensor).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            octree_resolution: Grid resolution for marching cubes.
            box_v: Bounding box extent.
            mc_level: Marching cubes iso-level.
            num_chunks: Batch size for SDF volume queries.
            seed: Random seed.

        Returns:
            trimesh.Trimesh mesh object.
        """
        if seed is not None:
            mx.random.seed(seed)

        # 1. Preprocess image
        image_tensor = self.preprocess_image(image)

        # 2. Encode with DINOv2
        cond = self.image_encoder(image_tensor, value_range=(-1, 1))
        _materialize(cond)

        # Unconditional embedding for CFG
        uncond = self.image_encoder.unconditional_embedding(1)

        # 3. Prepare latents
        num_latents, latent_dim = self.vae.latent_shape
        latents = mx.random.normal((1, num_latents, latent_dim))
        _materialize(latents)

        # 4. Set up scheduler with sigmas from 0 to 1 (matching original pipeline)
        sigmas = np.linspace(0, 1, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        do_cfg = guidance_scale >= 0

        # 5. Denoising loop
        for i in range(num_inference_steps):
            t = self.scheduler.timesteps[i]

            if do_cfg:
                # Duplicate latents for CFG
                latent_input = mx.concatenate([latents, latents], axis=0)
                context = mx.concatenate([cond, uncond], axis=0)
                t_batch = mx.broadcast_to(t, (2,))
            else:
                latent_input = latents
                context = cond
                t_batch = mx.broadcast_to(t, (1,))

            # Normalize timesteps to [0, 1] range (matching original flow matching pipeline)
            t_batch = t_batch / self.scheduler.num_train_timesteps

            # Model prediction
            noise_pred = self.dit(
                latent_input, t_batch, {"main": context}
            )
            _materialize(noise_pred)

            if do_cfg:
                pred_cond, pred_uncond = noise_pred[:1], noise_pred[1:]
                noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # Euler step
            latents = self.scheduler.step(noise_pred, t, latents)
            _materialize(latents)

        # 6. Free DiT to save memory
        del self.dit
        self.dit = None
        gc.collect()

        # 7. Decode with VAE to mesh
        mesh = self.vae.decode_to_mesh(
            latents,
            bounds=box_v,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            mc_level=mc_level,
        )

        return mesh

    @classmethod
    def from_pretrained(
        cls,
        weights_source: str = "dgrauet/hunyuan3d-2.1-mlx",
    ) -> "ShapePipeline":
        """Load all components from converted weights.

        Args:
            weights_source: Either a HuggingFace repo ID (auto-downloaded via
                ``huggingface_hub.snapshot_download`` and cached locally) or an
                absolute path to a directory with ``config.json``,
                ``image_encoder.safetensors``, ``dit.safetensors``,
                ``vae.safetensors``.

        Environment:
            HUNYUAN3D_MLX_WEIGHTS_DIR — overrides ``weights_source`` if set.

        Returns:
            Loaded ShapePipeline.
        """
        env_override = os.environ.get("HUNYUAN3D_MLX_WEIGHTS_DIR")
        if env_override:
            weights_source = env_override

        if os.path.isdir(weights_source):
            model_dir = Path(weights_source)
        else:
            from huggingface_hub import snapshot_download
            model_dir = Path(snapshot_download(
                repo_id=weights_source,
                allow_patterns=[
                    "config.json",
                    "split_model.json",
                    "image_encoder.safetensors",
                    "dit.safetensors",
                    "vae.safetensors",
                ],
            ))

        # Load config
        config_path = model_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Load image encoder
        image_encoder = ImageEncoder.from_pretrained(
            str(model_dir / "image_encoder.safetensors"),
            config.get("image_encoder", {}),
        )

        # Load DiT (pass quantization params if present)
        dit_config = dict(config.get("dit", {}))
        split_path = model_dir / "split_model.json"
        if split_path.exists():
            with open(split_path) as f:
                split_info = json.load(f)
            quant_info = split_info.get("quantization", {})
            if "dit" in quant_info.get("quantized_components", []):
                dit_config["bits"] = quant_info.get("bits", 8)
                dit_config["group_size"] = quant_info.get("group_size", 64)

        dit = HunYuanDiTPlain.from_pretrained(
            str(model_dir / "dit.safetensors"),
            dit_config,
        )

        # Load VAE
        vae = ShapeVAEDecoder.from_pretrained(
            str(model_dir / "vae.safetensors"),
            config.get("vae", {}),
        )

        # Create scheduler
        scheduler_config = config.get("scheduler", {})
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
            shift=scheduler_config.get("shift", 1.0),
        )

        return cls(
            image_encoder=image_encoder,
            dit=dit,
            vae=vae,
            scheduler=scheduler,
        )
