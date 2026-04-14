"""HunyuanPaintPBR inference pipeline in MLX.

Full native MLX pipeline: VAE encode/decode, DINO feature extraction,
UNet denoising with classifier-free guidance, multiview PBR generation.
"""

from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .dino_mlx import DINOv2MLX, preprocess_for_dino
from .scheduler_mlx import UniPCMultistepSchedulerMLX
from .vae_mlx import AutoencoderKLMLX
from .unet.unet_mlx import UNet2p5DConditionModelMLX


class HunyuanPaintPipelineMLX:
    """Full MLX inference pipeline for multiview PBR texture generation.

    Generates 6-view albedo + metallic-roughness textures from a reference
    image conditioned on normal and position maps.
    """

    def __init__(
        self,
        unet: UNet2p5DConditionModelMLX,
        vae: AutoencoderKLMLX,
        dino: Optional[DINOv2MLX] = None,
        scheduler: Optional[UniPCMultistepSchedulerMLX] = None,
        view_size: int = 512,
        n_views: int = 6,
        pbr_settings: tuple = ("albedo", "mr"),
        scaling_factor: float = 0.18215,
    ):
        self.unet = unet
        self.vae = vae
        self.dino = dino
        self.scheduler = scheduler or UniPCMultistepSchedulerMLX()
        self.view_size = view_size
        self.n_views = n_views
        self.pbr_settings = pbr_settings
        self.n_pbr = len(pbr_settings)
        self.scaling_factor = scaling_factor

    def encode_image(self, image: mx.array) -> mx.array:
        """Encode image to latent space.

        Args:
            image: (B, H, W, 3) in [0, 1].

        Returns:
            (B, H/8, W/8, 4) latent.
        """
        x = image * 2.0 - 1.0  # [0,1] → [-1,1]
        z = self.vae.encode(x)
        return z * self.scaling_factor

    def decode_latent(self, z: mx.array) -> mx.array:
        """Decode latent to image.

        Args:
            z: (B, H/8, W/8, 4) latent.

        Returns:
            (B, H, W, 3) in [0, 1].
        """
        z = z / self.scaling_factor
        x = self.vae.decode(z)
        return mx.clip((x + 1.0) / 2.0, 0, 1)

    def prepare_conditions(
        self,
        normal_maps: List[Image.Image],
        position_maps: List[Image.Image],
    ) -> dict:
        """Encode normal and position maps to latent conditioning.

        Args:
            normal_maps: list of N PIL images (normal renders).
            position_maps: list of N PIL images (position renders).

        Returns:
            dict with 'embeds_normal' and 'embeds_position' latents.
        """
        def pil_list_to_latent(images):
            arrays = []
            for img in images:
                arr = np.array(img.resize(
                    (self.view_size, self.view_size)
                )).astype(np.float32) / 255.0
                arrays.append(arr)
            batch = mx.array(np.stack(arrays))  # (N, H, W, 3)
            return self.encode_image(batch)     # (N, H/8, W/8, 4)

        cond = {}
        if normal_maps:
            latents = pil_list_to_latent(normal_maps)
            cond["embeds_normal"] = latents[None, ...]  # (1, N, H/8, W/8, 4)
        if position_maps:
            latents = pil_list_to_latent(position_maps)
            cond["embeds_position"] = latents[None, ...]
            # Also keep raw position maps for voxel indices
            pos_arrays = []
            for img in position_maps:
                arr = np.array(img.resize(
                    (self.view_size, self.view_size)
                )).astype(np.float32) / 255.0
                pos_arrays.append(arr)
            cond["position_maps"] = mx.array(np.stack(pos_arrays))[None, ...]

        return cond

    def prepare_dino_features(
        self, reference_image: Image.Image
    ) -> Optional[mx.array]:
        """Extract DINO features from reference image.

        Returns:
            (1, N_tokens, C_dino) features or None if no DINO model.
        """
        if self.dino is None:
            return None

        pixel_values = preprocess_for_dino(np.array(reference_image))
        features = self.dino(pixel_values)  # (1, L+1, 1536)
        return features

    def __call__(
        self,
        reference_image: Image.Image,
        normal_maps: List[Image.Image],
        position_maps: List[Image.Image],
        num_inference_steps: int = 15,
        guidance_scale: float = 3.0,
        seed: int = 0,
    ) -> Dict[str, List[Image.Image]]:
        """Generate multiview PBR textures.

        Args:
            reference_image: Input image to texture from.
            normal_maps: Normal renders for each view (N images).
            position_maps: Position renders for each view (N images).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed.

        Returns:
            dict with 'albedo' and 'mr' keys, each a list of PIL images.
        """
        mx.random.seed(seed)
        n_views = len(normal_maps)
        latent_size = self.view_size // 8  # 512 → 64

        # --- Prepare conditions ---
        conditions = self.prepare_conditions(normal_maps, position_maps)
        dino_features = self.prepare_dino_features(reference_image)

        # Project DINO features
        dino_proj = None
        if dino_features is not None:
            dino_proj = self.unet.image_proj_model_dino(dino_features)

        # Encode reference image
        ref_arr = np.array(reference_image.resize(
            (self.view_size, self.view_size)
        )).astype(np.float32) / 255.0
        ref_latent = self.encode_image(mx.array(ref_arr[None, ...]))
        ref_latents = ref_latent[:, None, ...]  # (1, 1, H/8, W/8, 4)

        # --- Initialize noise ---
        total_samples = self.n_pbr * n_views
        latents = mx.random.normal(
            (total_samples, latent_size, latent_size, 4)
        )

        # --- Denoising loop ---
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            t_mx = mx.array([int(t)])

            # Reshape latents: (Np*Ng, H, W, 4) → (1, Np, Ng, H, W, 4)
            sample = latents.reshape(
                1, self.n_pbr, n_views, latent_size, latent_size, 4
            )

            # UNet forward
            noise_pred = self.unet.forward(
                sample,
                t_mx,
                embeds_normal=conditions.get("embeds_normal"),
                embeds_position=conditions.get("embeds_position"),
                ref_latents=ref_latents,
                dino_hidden_states=dino_proj,
                position_maps=conditions.get("position_maps"),
                n_views=n_views,
            )
            # noise_pred: (1, Np*Ng, H, W, 4) → (Np*Ng, H, W, 4)
            noise_pred = noise_pred[0]

            # Scheduler step
            latents = self.scheduler.step(noise_pred, int(t), latents)

        # --- Decode ---
        images = self.decode_latent(latents)  # (Np*Ng, H, W, 3)
        mx.synchronize()
        images_np = np.array(images)

        # Split into PBR materials
        result = {}
        for i, token in enumerate(self.pbr_settings):
            views = []
            for j in range(n_views):
                idx = i * n_views + j
                img_np = np.clip(images_np[idx] * 255, 0, 255).astype(np.uint8)
                views.append(Image.fromarray(img_np))
            result[token] = views

        return result

    @staticmethod
    def from_pretrained(weights_dir: str, **kwargs) -> "HunyuanPaintPipelineMLX":
        """Load pipeline from converted MLX weights.

        Args:
            weights_dir: Directory containing mlx weight files
                         (unet_mlx.safetensors, vae_mlx.safetensors, etc.)

        Returns:
            Initialized pipeline.
        """
        import os
        from safetensors.numpy import load_file

        # Initialize models with default architecture
        vae = AutoencoderKLMLX()
        unet = UNet2p5DConditionModelMLX(
            in_channels=4,
            condition_channels=8,
            out_channels=4,
            block_out_channels=kwargs.get(
                "block_out_channels", (320, 640, 1280, 1280)
            ),
            cross_attention_dim=kwargs.get("cross_attention_dim", 1024),
            attention_head_dim=kwargs.get("attention_head_dim", 8),
            pbr_settings=kwargs.get("pbr_settings", ("albedo", "mr")),
        )

        # Load weights if available
        vae_path = os.path.join(weights_dir, "vae_mlx.safetensors")
        if os.path.exists(vae_path):
            weights = load_file(vae_path)
            vae.load_weights(list(weights.items()))

        unet_path = os.path.join(weights_dir, "unet_mlx.safetensors")
        if os.path.exists(unet_path):
            weights = load_file(unet_path)
            unet.load_weights(list(weights.items()))

        # DINO is optional
        dino = None
        dino_path = os.path.join(weights_dir, "dino_mlx.safetensors")
        if os.path.exists(dino_path):
            dino = DINOv2MLX()
            weights = load_file(dino_path)
            dino.load_weights(list(weights.items()))

        return HunyuanPaintPipelineMLX(
            unet=unet,
            vae=vae,
            dino=dino,
            view_size=kwargs.get("view_size", 512),
        )
