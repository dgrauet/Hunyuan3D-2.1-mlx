"""UNet building blocks for Stable Diffusion 2.1 in MLX.

All tensors use NHWC layout: (batch, height, width, channels).
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

def get_timestep_embedding(timesteps: mx.array, embedding_dim: int) -> mx.array:
    """Sinusoidal timestep embedding (B,) → (B, embedding_dim).

    Matches diffusers convention with downscale_freq_shift=0 (SD 2.1).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / half_dim  # was / (half_dim - 1) — off-by-one bug
    emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class TimestepEmbedding(nn.Module):
    """Projects sinusoidal timestep embedding to model dimension."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(self.act(self.linear_1(sample)))


# ---------------------------------------------------------------------------
# ResnetBlock2D
# ---------------------------------------------------------------------------

class ResnetBlock2D(nn.Module):
    """Residual block with optional timestep conditioning.

    Input/output: (B, H, W, C) NHWC format.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()

        self.time_emb_proj = (
            nn.Linear(temb_channels, out_channels)
            if temb_channels > 0
            else None
        )

        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def __call__(
        self, x: mx.array, temb: Optional[mx.array] = None
    ) -> mx.array:
        residual = x

        h = self.nonlinearity(self.norm1(x))
        h = self.conv1(h)

        if temb is not None and self.time_emb_proj is not None:
            temb_proj = self.nonlinearity(temb)
            temb_proj = self.time_emb_proj(temb_proj)
            # (B, C) → (B, 1, 1, C) for broadcasting
            h = h + temb_proj[:, None, None, :]

        h = self.nonlinearity(self.norm2(h))
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return h + residual


# ---------------------------------------------------------------------------
# Attention Block (for UNet transformer blocks)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention with optional cross-attention.

    Input: (B, L, C) where L = H*W for spatial attention.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        cross_attention_dim = cross_attention_dim or query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            hidden_states: (B, L, C)
            encoder_hidden_states: (B, L_ctx, C_ctx) for cross-attention.
        """
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        B, L, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape to multi-head: (B, L, H*D) → (B, H, L, D)
        def reshape_heads(x):
            return x.reshape(B, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)

        # Use the fused flash-attention kernel — explicit softmax(QK^T)V
        # builds a (L, L_ctx) scores tensor per head per layer and blows
        # past unified-memory limits when L is large (multiview attention
        # at 6 views * 4096 tokens = 24k).
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape back: (B, H, L, D) -> (B, L, H*D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# BasicTransformerBlock
# ---------------------------------------------------------------------------

class BasicTransformerBlock(nn.Module):
    """Standard transformer block with self-attention + cross-attention + FFN."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )

        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim or dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        temb: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """(B, L, C) → (B, L, C)

        If 2.5D modules are attached (by _enhance_unet), they participate:
        - MDA: material-dimension self-attention (split albedo/MR)
        - Multiview: cross-view self-attention
        - Reference: cross-attention to cached reference features
        - DINO: cross-attention to vision features

        kwargs:
            dino_features: (B, N_dino, C) DINO projected features
            n_views: int, number of views per material
            n_pbr: int, number of materials (2)
            ref_features: (B, L_ref, C) cached reference features (optional)
        """
        n_views = kwargs.get("n_views", 0)
        n_pbr = kwargs.get("n_pbr", 2)
        has_25d = hasattr(self, "attn_multiview")

        # --- Capture mode: store post-norm1 (pre-attn1) features for reference ---
        _capture_dict = kwargs.get("_capture_dict")

        # --- Step 1: Self-attention (with MDA if available) ---
        norm_hs = self.norm1(hidden_states)

        # --- Capture: write mode (BEFORE attn1 output is applied) ---
        # Matches PyTorch: condition_embed_dict[layer_name] =
        #   rearrange(norm_hidden_states, "(b n) l c -> b (n l) c", n=num_in_batch)
        if _capture_dict is not None:
            block_id = getattr(self, "_block_id", "unknown")
            B_total, L, C = norm_hs.shape
            # For ref extraction, n_views=1 => (B, L, C) unchanged.
            ref_n = kwargs.get("_ref_n_views", 1)
            if ref_n < 1:
                ref_n = 1
            reshaped = norm_hs.reshape(B_total // ref_n, ref_n * L, C)
            _capture_dict[block_id] = reshaped

        if has_25d and hasattr(self.attn1, "processor") and n_views > 0:
            # MDA: process each material separately with its own projections
            B_total, L, C = hidden_states.shape
            B = B_total // (n_pbr * n_views)

            # Standard attention for albedo (first material)
            albedo_idx = list(range(0, B * n_views))
            mr_idx = list(range(B * n_views, B * n_pbr * n_views))

            hs_albedo = norm_hs[mx.array(albedo_idx)]
            hs_mr = norm_hs[mx.array(mr_idx)]

            # Albedo uses standard attn1 projections
            attn_albedo = self.attn1(hs_albedo)

            # MR uses processor's projections
            proc = self.attn1.processor
            q = proc.to_q_mr(hs_mr)
            k = proc.to_k_mr(hs_mr)
            v = proc.to_v_mr(hs_mr)
            heads = self.attn1.heads
            dim_head = C // heads

            def _reshape_heads(x):
                return x.reshape(x.shape[0], -1, heads, dim_head).transpose(0, 2, 1, 3)

            scores = (_reshape_heads(q) @ _reshape_heads(k).transpose(0, 1, 3, 2)) * (dim_head ** -0.5)
            attn_mr = (mx.softmax(scores, axis=-1) @ _reshape_heads(v))
            attn_mr = attn_mr.transpose(0, 2, 1, 3).reshape(hs_mr.shape[0], L, C)
            attn_mr = proc.to_out_mr(attn_mr)

            # Recombine
            attn_out = mx.concatenate([attn_albedo, attn_mr], axis=0)
        else:
            attn_out = self.attn1(norm_hs)

        hidden_states = attn_out + hidden_states

        # --- Step 2: Reference attention (if features provided) ---
        # Matches PyTorch modules.py lines 586-609:
        #   query = rearrange(norm_hidden_states,
        #                     "(b n_pbr n) l c -> b n_pbr (n l) c")[:, 0]  # albedo only
        #   out   = attn_refview(query, condition_embed)
        #   scatter back to albedo samples; MR samples get zero (processed
        #   inside attn_refview via processor for material-specific V/out, but
        #   the MLX port uses albedo-only residual here)
        ref_features = kwargs.get("ref_features")
        if has_25d and hasattr(self, "attn_refview") and ref_features is not None:
            block_id = getattr(self, "_block_id", None)
            if block_id is not None and block_id in ref_features:
                ref_ctx = ref_features[block_id]  # (B_ref, n_ref*L, C)
                B_total, L, C = hidden_states.shape
                nv = max(int(n_views) if n_views else 1, 1)
                np_ = max(int(n_pbr) if n_pbr else 1, 1)
                B_batch = B_total // (np_ * nv)

                # Reshape: (B*n_pbr*n_views, L, C) -> (B, n_pbr, n_views, L, C)
                # PT modules.py:586-609 takes albedo only as the query but
                # then RefAttnProcessor2_0 internally produces a per-material
                # output by applying material-specific V/out projections
                # (to_v_mr / to_out_mr from the processor). Albedo uses the
                # plain attn_refview.to_v / to_out.
                nh = norm_hs.reshape(B_batch, np_, nv, L, C)
                query_albedo = nh[:, 0].reshape(B_batch, nv * L, C)  # (B, nv*L, C)

                if ref_ctx.shape[0] < query_albedo.shape[0]:
                    ref_ctx = mx.broadcast_to(
                        ref_ctx,
                        (query_albedo.shape[0],) + ref_ctx.shape[1:],
                    )

                attn = self.attn_refview
                heads = attn.heads
                head_dim = attn.dim_head

                def _split_heads(t, B):
                    return t.reshape(B, -1, heads, head_dim).transpose(0, 2, 1, 3)

                # Q, K shared across materials (computed from albedo query +
                # ref ctx). V depends on material — albedo uses to_v, MR uses
                # processor.to_v_mr. Attention scores are the same for all
                # materials (same Q, same K), so we compute scores once and
                # reuse.
                q = _split_heads(attn.to_q(query_albedo), query_albedo.shape[0])
                k = _split_heads(attn.to_k(ref_ctx), ref_ctx.shape[0])
                scale = head_dim ** -0.5

                ref_out_per_mat = []
                for mat_i in range(np_):
                    if mat_i == 0:
                        v_proj = attn.to_v
                        out_proj = attn.to_out
                    else:
                        # PBR materials beyond albedo use the per-material
                        # processor weights (to_v_mr, to_out_mr).
                        proc = getattr(attn, "processor", None)
                        if proc is None or not hasattr(proc, "to_v_mr"):
                            ref_out_per_mat.append(
                                mx.zeros((B_batch, nv * L, C), dtype=q.dtype)
                            )
                            continue
                        v_proj = proc.to_v_mr
                        out_proj = proc.to_out_mr

                    v = _split_heads(v_proj(ref_ctx), ref_ctx.shape[0])
                    attn_out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale,
                    )
                    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
                        B_batch, nv * L, C
                    )
                    ref_out_per_mat.append(out_proj(attn_out))

                # (B, np, nv*L, C) -> (B, np, nv, L, C) -> (B*np*nv, L, C)
                ref_out_full = mx.stack(ref_out_per_mat, axis=1)
                ref_out_full = ref_out_full.reshape(B_batch, np_, nv, L, C)
                ref_out_flat = ref_out_full.reshape(B_total, L, C)
                hidden_states = hidden_states + ref_out_flat

        # --- Step 3: Multiview attention (with optional 3D RoPE) ---
        if has_25d and n_views > 1:
            B_total, L, C = hidden_states.shape
            B_mat = B_total // n_views  # B * n_pbr
            norm_hs = self.norm1(hidden_states)

            # Reshape to concat views: (B*n_pbr, n_views*L, C)
            mv_input = norm_hs.reshape(B_mat, n_views * L, C)

            # Optional 3D RoPE (matches PT's PoseRoPEAttnProcessor):
            # position_voxel_indices is keyed by the MULTIVIEW seq length
            # (n_views * L), not the per-view L (PT modules.py:618 uses
            # multivew_hidden_states.shape[1]). Without this lookup the
            # RoPE never fired and the model — trained with RoPE — output
            # garbage when multiview attention was active.
            mv_seq_len = n_views * L
            position_voxel_indices = kwargs.get("position_voxel_indices")
            pos_idx_dict = None
            if position_voxel_indices is not None and mv_seq_len in position_voxel_indices:
                pos_idx_dict = position_voxel_indices[mv_seq_len]

            if pos_idx_dict is not None:
                from .attn_processor_mlx import RotaryEmbedding
                attn = self.attn_multiview
                heads = attn.heads
                head_dim = attn.dim_head

                def _split_heads(t):
                    B, NL, C = t.shape
                    return t.reshape(B, NL, heads, head_dim).transpose(0, 2, 1, 3)

                q = _split_heads(attn.to_q(mv_input))
                k = _split_heads(attn.to_k(mv_input))
                v = _split_heads(attn.to_v(mv_input))

                vi = pos_idx_dict["voxel_indices"]
                vr = pos_idx_dict["voxel_resolution"]
                cos, sin = RotaryEmbedding.get_3d_rotary_pos_embed(vi, head_dim, vr)
                q = RotaryEmbedding.apply_rotary_emb(q, (cos, sin))
                k = RotaryEmbedding.apply_rotary_emb(k, (cos, sin))

                # Use mx.fast.scaled_dot_product_attention for memory-
                # efficient flash-attention style fused kernel (the multiview
                # seq_len = n_views * L can be 6*4096 = 24k tokens; the
                # explicit softmax(QK^T)V allocates a 24k*24k matrix per
                # head per layer and OOMs on M-series unified memory).
                mv_out = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=head_dim ** -0.5,
                )
                mv_out = mv_out.transpose(0, 2, 1, 3).reshape(B_mat, n_views * L, C)
                mv_out = attn.to_out(mv_out)
            else:
                mv_out = self.attn_multiview(mv_input)

            mv_out = mv_out.reshape(B_total, L, C)
            hidden_states = mv_out + hidden_states

        # --- Step 4: Text cross-attention ---
        hidden_states = (
            self.attn2(self.norm2(hidden_states), encoder_hidden_states)
            + hidden_states
        )

        # --- Step 5: DINO cross-attention ---
        dino_features = kwargs.get("dino_features")
        if has_25d and hasattr(self, "attn_dino") and dino_features is not None:
            hidden_states = (
                self.attn_dino(self.norm2(hidden_states), dino_features)
                + hidden_states
            )

        # --- Step 6: FFN ---
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class FeedForward(nn.Module):
    """GEGLU feed-forward network."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        # GEGLU: projects to 2*inner_dim, splits, applies gating
        self.proj_in = nn.Linear(dim, inner_dim * 2)
        self.proj_out = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj_in(x)
        h, gate = mx.split(h, 2, axis=-1)
        h = h * nn.gelu(gate)
        return self.proj_out(h)


# ---------------------------------------------------------------------------
# Transformer2DModel (spatial transformer)
# ---------------------------------------------------------------------------

class Transformer2DModel(nn.Module):
    """Spatial transformer: reshapes spatial dims to sequence, applies
    transformer blocks, reshapes back.

    Input/output: (B, H, W, C) NHWC.
    """

    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 320,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(norm_num_groups, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = [
            BasicTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ]

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        temb: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """(B, H, W, C) → (B, H, W, C)"""
        residual = hidden_states
        B, H, W, C = hidden_states.shape

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(B, H * W, C)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, temb, **kwargs
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(B, H, W, C)
        return hidden_states + residual


# ---------------------------------------------------------------------------
# Down/Up sampling
# ---------------------------------------------------------------------------

class Downsample2D(nn.Module):
    """Spatial downsample by 2x using strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample2D(nn.Module):
    """Spatial upsample by 2x using nearest + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(self.upsample(x))


# ---------------------------------------------------------------------------
# UNet Down/Mid/Up Blocks
# ---------------------------------------------------------------------------

class DownBlock2D(nn.Module):
    """UNet down block: ResnetBlocks + optional Downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 1280,
        num_layers: int = 2,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(in_ch, out_channels, temb_channels)
            )

        self.downsamplers = (
            [Downsample2D(out_channels)] if add_downsample else None
        )

    def __call__(
        self, hidden_states: mx.array, temb: mx.array
    ) -> tuple:
        output_states = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    """UNet down block with cross-attention transformers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 1280,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        add_downsample: bool = True,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        attention_head_dim = out_channels // num_attention_heads

        self.resnets = []
        self.attentions = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(in_ch, out_channels, temb_channels)
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.downsamplers = (
            [Downsample2D(out_channels)] if add_downsample else None
        )
        self.has_cross_attention = True

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        **kwargs,
    ) -> tuple:
        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states, **kwargs)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class UNetMidBlock2DCrossAttn(nn.Module):
    """UNet middle block: ResNet + CrossAttn + ResNet."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int = 1280,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        attention_head_dim = in_channels // num_attention_heads

        self.resnets = [
            ResnetBlock2D(in_channels, in_channels, temb_channels),
            ResnetBlock2D(in_channels, in_channels, temb_channels),
        ]
        self.attentions = [
            Transformer2DModel(
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                in_channels=in_channels,
                num_layers=transformer_layers_per_block,
                cross_attention_dim=cross_attention_dim,
            )
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states, **kwargs)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class UpBlock2D(nn.Module):
    """UNet up block: ResnetBlocks + optional Upsample."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int = 1280,
        num_layers: int = 3,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            # Skip connection channels: last layer gets in_channels, others get out_channels
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in = (
                (prev_output_channel if i == 0 else out_channels) + res_skip_channels
            )
            self.resnets.append(
                ResnetBlock2D(resnet_in, out_channels, temb_channels)
            )

        self.upsamplers = (
            [Upsample2D(out_channels)] if add_upsample else None
        )

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        res_hidden_states: list,
    ) -> mx.array:
        for resnet in self.resnets:
            res = res_hidden_states.pop()
            hidden_states = mx.concatenate(
                [hidden_states, res], axis=-1
            )
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    """UNet up block with cross-attention."""

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int = 1280,
        num_layers: int = 3,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        add_upsample: bool = True,
        transformer_layers_per_block: int = 1,
    ):
        super().__init__()
        attention_head_dim = out_channels // num_attention_heads

        self.resnets = []
        self.attentions = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in = (
                (prev_output_channel if i == 0 else out_channels) + res_skip_channels
            )
            self.resnets.append(
                ResnetBlock2D(resnet_in, out_channels, temb_channels)
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block,
                    cross_attention_dim=cross_attention_dim,
                )
            )

        self.upsamplers = (
            [Upsample2D(out_channels)] if add_upsample else None
        )
        self.has_cross_attention = True

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        res_hidden_states: list,
        encoder_hidden_states: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        for resnet, attn in zip(self.resnets, self.attentions):
            res = res_hidden_states.pop()
            hidden_states = mx.concatenate(
                [hidden_states, res], axis=-1
            )
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states, **kwargs)

        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states
