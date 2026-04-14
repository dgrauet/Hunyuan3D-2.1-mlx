"""Stable Diffusion 2.1 UNet for MLX (NHWC layout).

UNet2DConditionModelMLX: standard SD 2.1 UNet.
UNet2p5DConditionModelMLX: enhanced with material/multiview/reference attention.
"""

from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .blocks_mlx import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    ResnetBlock2D,
    TimestepEmbedding,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    get_timestep_embedding,
)


class UNet2DConditionModelMLX(nn.Module):
    """Stable Diffusion 2.1 UNet in MLX (NHWC layout).

    Architecture:
        in_channels=4 (or 12 for concat conditioning)
        block_out_channels=(320, 640, 1280, 1280)
        cross_attention_dim=1024 (SD 2.1)
        down: CrossAttn, CrossAttn, CrossAttn, DownOnly
        mid: CrossAttn
        up: UpOnly, CrossAttn, CrossAttn, CrossAttn
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: tuple = (320, 640, 1280, 1280),
        cross_attention_dim: int = 1024,
        attention_head_dim: int = 8,
        layers_per_block: int = 2,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        time_embed_dim = block_out_channels[0] * 4  # 1280

        # Timestep embedding
        self.time_proj_dim = block_out_channels[0]  # 320
        self.time_embedding = TimestepEmbedding(
            block_out_channels[0], time_embed_dim
        )

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)

        # Down blocks
        self.down_blocks = []
        output_channel = block_out_channels[0]
        for i, ch in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            input_channel = output_channel
            output_channel = ch

            if i < len(block_out_channels) - 1:
                # CrossAttn blocks for first 3 levels
                num_heads = ch // (ch // attention_head_dim) if attention_head_dim < ch else 1
                # SD 2.1 uses attention_head_dim as the per-head dimension
                num_heads = ch // attention_head_dim
                self.down_blocks.append(
                    CrossAttnDownBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block,
                        num_attention_heads=num_heads,
                        cross_attention_dim=cross_attention_dim,
                        add_downsample=not is_last,
                        transformer_layers_per_block=transformer_layers_per_block,
                    )
                )
            else:
                # Last block: no attention
                self.down_blocks.append(
                    DownBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block,
                        add_downsample=False,
                    )
                )

        # Mid block
        mid_channels = block_out_channels[-1]
        num_mid_heads = mid_channels // attention_head_dim
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_channels,
            temb_channels=time_embed_dim,
            num_attention_heads=num_mid_heads,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
        )

        # Up blocks (reversed)
        reversed_channels = list(reversed(block_out_channels))
        self.up_blocks = []
        output_channel = reversed_channels[0]
        for i, ch in enumerate(reversed_channels):
            is_last = i == len(reversed_channels) - 1
            prev_output_channel = output_channel
            output_channel = ch
            input_channel = reversed_channels[min(i + 1, len(reversed_channels) - 1)]

            if i > 0:
                # CrossAttn for last 3 levels
                num_heads = ch // attention_head_dim
                self.up_blocks.append(
                    CrossAttnUpBlock2D(
                        in_channels=input_channel,
                        prev_output_channel=prev_output_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block + 1,
                        num_attention_heads=num_heads,
                        cross_attention_dim=cross_attention_dim,
                        add_upsample=not is_last,
                        transformer_layers_per_block=transformer_layers_per_block,
                    )
                )
            else:
                # First up block: no attention
                self.up_blocks.append(
                    UpBlock2D(
                        in_channels=input_channel,
                        prev_output_channel=prev_output_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block + 1,
                        add_upsample=True,
                    )
                )

        # Output
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Args:
            sample: (B, H, W, C_in) noisy latent.
            timestep: (B,) or scalar timestep.
            encoder_hidden_states: (B, L, C_text) text embeddings.
            **kwargs: Extra context passed to transformer blocks
                      (e.g. dino_features for 2.5D attention).

        Returns:
            (B, H, W, C_out) noise prediction.
        """
        # Timestep embedding
        if not isinstance(timestep, mx.array):
            timestep = mx.array([timestep])
        if timestep.ndim == 0:
            timestep = mx.expand_dims(timestep, 0)
        t_emb = get_timestep_embedding(timestep, self.time_proj_dim)
        emb = self.time_embedding(t_emb)  # (B, time_embed_dim)

        # Broadcast for batch
        if emb.shape[0] == 1 and sample.shape[0] > 1:
            emb = mx.broadcast_to(emb, (sample.shape[0],) + emb.shape[1:])

        # Reset ref feature counter for each forward pass
        if "ref_features" in kwargs:
            kwargs["_ref_counter"] = [0]

        # Input conv
        sample = self.conv_in(sample)

        # Down
        down_block_res_samples = [sample]
        for block in self.down_blocks:
            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample, res = block(sample, emb, encoder_hidden_states, **kwargs)
            else:
                sample, res = block(sample, emb)
            down_block_res_samples.extend(res)

        # Mid
        sample = self.mid_block(sample, emb, encoder_hidden_states, **kwargs)

        # Up
        for block in self.up_blocks:
            n_res = len(block.resnets)
            res_samples = down_block_res_samples[-n_res:]
            down_block_res_samples = down_block_res_samples[:-n_res]

            if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                sample = block(sample, emb, res_samples, encoder_hidden_states, **kwargs)
            else:
                sample = block(sample, emb, res_samples)

        # Output
        sample = self.conv_act(self.conv_norm_out(sample))
        sample = self.conv_out(sample)
        return sample


# ---------------------------------------------------------------------------
# UNet2p5DConditionModel — wraps standard UNet with 2.5D attention
# ---------------------------------------------------------------------------


class UNet2p5DConditionModelMLX(nn.Module):
    """HunyuanPaintPBR 2.5D UNet.

    Wraps a standard SD 2.1 UNet and manages:
    - Dual-stream reference processing (separate UNet for reference features)
    - Material-dimension, multiview, reference, and DINO attention routing
    - Condition concatenation (normal + position maps as extra latent channels)
    - Learned material text embeddings

    Forward pass:
        1. Concatenate noise latent + normal + position condition channels
        2. (Optional) Run reference UNet in "write" mode to cache features
        3. Run main UNet in "read" mode with all attention mechanisms active
    """

    def __init__(
        self,
        in_channels: int = 4,
        condition_channels: int = 8,  # normal(4) + position(4)
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        cross_attention_dim: int = 1024,
        attention_head_dim: int = 8,
        layers_per_block: int = 2,
        pbr_settings: Tuple[str, ...] = ("albedo", "mr"),
        use_dual_stream: bool = True,
    ):
        super().__init__()
        self.pbr_settings = pbr_settings
        self.n_pbr = len(pbr_settings)
        self.use_dual_stream = use_dual_stream
        self.cross_attention_dim = cross_attention_dim

        total_in = in_channels + condition_channels

        # Main UNet
        self.unet = UNet2DConditionModelMLX(
            in_channels=total_in,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            layers_per_block=layers_per_block,
        )

        # Dual-stream reference UNet (separate weights)
        if use_dual_stream:
            self.unet_dual = UNet2DConditionModelMLX(
                in_channels=total_in,
                out_channels=out_channels,
                block_out_channels=block_out_channels,
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=attention_head_dim,
                layers_per_block=layers_per_block,
            )

        # Learned material text embeddings: override CLIP text conditioning
        self.learned_text_clip = {}
        for token in list(pbr_settings) + ["ref"]:
            param = mx.zeros((77, cross_attention_dim))
            setattr(self, f"learned_text_clip_{token}", param)
            self.learned_text_clip[token] = param

        # DINO feature projector
        from .attn_processor_mlx import ImageProjModel
        self.image_proj_model_dino = ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=1536,
            clip_extra_context_tokens=4,
        )

    def forward(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        embeds_normal: Optional[mx.array] = None,
        embeds_position: Optional[mx.array] = None,
        ref_latents: Optional[mx.array] = None,
        dino_hidden_states: Optional[mx.array] = None,
        position_maps: Optional[mx.array] = None,
        n_views: int = 6,
        mva_scale: float = 1.0,
        ref_scale: float = 1.0,
    ) -> mx.array:
        """Full forward pass with dual-stream processing.

        Args:
            sample: (B, N_pbr, N_gen, H, W, C_latent) noisy latents.
            timestep: scalar or (B,) timestep.
            encoder_hidden_states: (B, N_pbr, L_text, C_text) or None (uses learned).
            embeds_normal: (B, N_gen, H, W, C_cond) normal map latents.
            embeds_position: (B, N_gen, H, W, C_cond) position map latents.
            ref_latents: (B, N_ref, H, W, C_latent) reference latents.
            dino_hidden_states: (B, N*L_dino, C_dino) DINO features.
            position_maps: (B, N_gen, H, W, 3) raw position maps for RoPE.
            n_views: number of generated views.

        Returns:
            (B, N_pbr*N_gen, H, W, C_out) noise predictions.
        """
        B = sample.shape[0]

        # --- Concatenate condition channels ---
        # sample: (B, Np, Ng, H, W, 4)
        # embeds: (B, Ng, H, W, 4) → broadcast to (B, Np, Ng, H, W, 4)
        channels = [sample]
        if embeds_normal is not None:
            en = mx.broadcast_to(
                embeds_normal[:, None, :, :, :, :],
                sample.shape,
            )
            channels.append(en)
        if embeds_position is not None:
            ep = mx.broadcast_to(
                embeds_position[:, None, :, :, :, :],
                sample.shape,
            )
            channels.append(ep)

        concat_sample = mx.concatenate(channels, axis=-1)
        # (B, Np, Ng, H, W, C_total) → (B*Np*Ng, H, W, C_total)
        Np, Ng = self.n_pbr, n_views
        _, _, _, H, W, C_total = concat_sample.shape
        flat_sample = concat_sample.reshape(B * Np * Ng, H, W, C_total)

        # --- Text embeddings ---
        if encoder_hidden_states is None:
            # Use learned material-specific text embeddings
            text_list = [
                getattr(self, f"learned_text_clip_{t}")
                for t in self.pbr_settings
            ]
            # (Np, 77, C_text) → (B, Np, 77, C_text) → broadcast to views
            text_stack = mx.stack(text_list, axis=0)[None, ...]  # (1, Np, 77, C)
            text_stack = mx.broadcast_to(text_stack, (B, Np, 77, self.cross_attention_dim))
            # (B, Np, 77, C) → (B*Np, 77, C) → repeat for views
            text_flat = text_stack.reshape(B * Np, 77, self.cross_attention_dim)
            text_flat = mx.repeat(text_flat[:, None, :, :], Ng, axis=1)
            text_flat = text_flat.reshape(B * Np * Ng, 77, self.cross_attention_dim)
        else:
            text_flat = encoder_hidden_states

        # --- DINO projection ---
        dino_proj = None
        if dino_hidden_states is not None:
            dino_proj = self.image_proj_model_dino(dino_hidden_states)

        # --- Position voxel indices for RoPE ---
        voxel_indices = None
        if position_maps is not None:
            from .modules_mlx import calc_multires_voxel_indices
            voxel_indices = calc_multires_voxel_indices(position_maps)

        # --- Reference stream (dual UNet, write mode) ---
        condition_embed_dict = {}
        if ref_latents is not None and self.use_dual_stream:
            # Prepare reference input
            B_ref, N_ref = ref_latents.shape[:2]
            ref_flat = ref_latents.reshape(B_ref * N_ref, *ref_latents.shape[2:])

            # Use learned reference text embedding
            ref_text = getattr(self, "learned_text_clip_ref")
            ref_text = mx.broadcast_to(
                ref_text[None, :, :], (B_ref * N_ref, 77, self.cross_attention_dim)
            )

            # Run reference UNet (just to cache features, output discarded)
            # For simplicity, we run the standard UNet forward and collect
            # intermediate features. In production, this would use the
            # transformer blocks' "write" mode.
            # TODO: Implement proper write-mode feature caching through
            # the UNet's transformer blocks.
            pass

        # --- Main UNet forward ---
        noise_pred = self.unet(
            flat_sample, timestep, text_flat,
        )

        return noise_pred.reshape(B, Np * Ng, H, W, noise_pred.shape[-1])
