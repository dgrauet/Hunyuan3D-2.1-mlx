"""Stable Diffusion 2.1 UNet for MLX (NHWC layout).

UNet2DConditionModelMLX: standard SD 2.1 UNet with optional 2.5D attention
modules attached externally via load_model._enhance_unet.
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
        attention_head_dim: tuple | int = (5, 10, 20, 20),
        layers_per_block: int = 2,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        time_embed_dim = block_out_channels[0] * 4  # 1280

        # Normalize attention_head_dim to a list
        if isinstance(attention_head_dim, int):
            attention_head_dim = [attention_head_dim] * len(block_out_channels)
        self._attention_head_dims = list(attention_head_dim)

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
                # Diffusers convention: attention_head_dim is actually num_heads.
                # The true per-head dim is ch // num_heads.
                num_heads = self._attention_head_dims[i]
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
        num_mid_heads = self._attention_head_dims[-1]
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
                # CrossAttn — use reversed num_heads to match down blocks.
                # Diffusers convention: attention_head_dim == num_heads.
                rev_idx = len(block_out_channels) - 1 - i
                num_heads = self._attention_head_dims[max(rev_idx, 0)]
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
        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0], pytorch_compatible=True)
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
