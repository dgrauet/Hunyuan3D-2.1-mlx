"""HunYuanDiTPlain shape transformer in MLX.

MLX port of the HunYuanDiTPlain denoiser from hunyuandit.py.
21 blocks with U-Net skip connections, last 6 blocks use MoE with shared expert.

Weight key remapping (checkpoint -> module):
  dit.t_embedder.mlp.0.*            -> t_embedder.mlp_0.*
  dit.t_embedder.mlp.2.*            -> t_embedder.mlp_2.*
  dit.blocks.N.moe.gate.weight      -> blocks.N.moe.gate.gate.weight  (raw param -> nn.Linear)
  dit.blocks.N.moe.experts.E.net.0.proj.*  -> blocks.N.moe.experts.E.fc1.*
  dit.blocks.N.moe.experts.E.net.2.*      -> blocks.N.moe.experts.E.fc2.*
  dit.blocks.N.moe.shared_experts.*       -> blocks.N.moe.shared_expert.*
"""

import math
import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_ops.moe import MoELayer


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class Timesteps(nn.Module):
    """Sinusoidal timestep embedding (no learnable params).

    Args:
        num_channels: Output embedding dimension.
        max_period: Maximum period for the sinusoidal frequencies.
    """

    def __init__(self, num_channels: int, max_period: int = 10000):
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period

    def __call__(self, timesteps: mx.array) -> mx.array:
        """
        Args:
            timesteps: (B,) 1-D array of timestep values.
        Returns:
            (B, num_channels) sinusoidal embedding.
        """
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * mx.arange(half_dim).astype(mx.float32)
        exponent = exponent / half_dim
        emb = mx.exp(exponent)
        emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
        emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Checkpoint keys use ``t_embedder.mlp.0.weight/bias`` and
    ``t_embedder.mlp.2.weight/bias`` (Sequential indices). The module uses
    ``mlp_0``/``mlp_2`` and keys are remapped in ``from_pretrained``.

    Args:
        hidden_size: Input size for sinusoidal embedding.
        frequency_embedding_size: MLP hidden size.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.time_embed = Timesteps(hidden_size)
        self.mlp_0 = nn.Linear(hidden_size, frequency_embedding_size)
        self.mlp_2 = nn.Linear(frequency_embedding_size, hidden_size)

    def __call__(self, t: mx.array) -> mx.array:
        """
        Args:
            t: (B,) timestep values.
        Returns:
            (B, 1, hidden_size) timestep embedding.
        """
        t_freq = self.time_embed(t).astype(self.mlp_0.weight.dtype)
        t = self.mlp_2(nn.gelu(self.mlp_0(t_freq)))
        return t[:, None, :]


class MLP(nn.Module):
    """Standard feed-forward network: Linear -> GELU -> Linear.

    Args:
        width: Input/output dimension.
    """

    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head self-attention with optional QK RMSNorm.

    Note: Checkpoint has no bias on to_q/to_k/to_v but does have bias on
    out_proj. The ``qkv_bias`` parameter only affects to_q/to_k/to_v.

    Args:
        dim: Hidden dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projections.
        qk_norm: Whether to apply RMSNorm to Q and K.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Match PyTorch's interleaved QKV head assignment:
        # cat Q,K,V -> view(1, N, heads, 3*hd) -> split per head
        qkv = mx.concatenate([q, k, v], axis=-1)  # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, self.num_heads, 3 * self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=-1)  # each (B, N, heads, hd)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, -1)
        return self.out_proj(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention (query attends to context).

    Note: Checkpoint has no bias on to_q/to_k/to_v but does have bias on
    out_proj.

    Args:
        qdim: Query dimension.
        kdim: Key/value dimension (context).
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias.
        qk_norm: Whether to apply RMSNorm to Q and K.
    """

    def __init__(
        self,
        qdim: int,
        kdim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        self.head_dim = qdim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.to_k = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.to_v = nn.Linear(kdim, qdim, bias=qkv_bias)
        self.out_proj = nn.Linear(qdim, qdim)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if qk_norm else None

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        B, S1, _ = x.shape
        _, S2, _ = y.shape

        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        # Match PyTorch's interleaved KV head assignment:
        # cat K,V -> view(B, S2, heads, 2*hd) -> split per head
        kv = mx.concatenate([k, v], axis=-1)  # (B, S2, 2*qdim)
        kv = kv.reshape(B, S2, self.num_heads, 2 * self.head_dim)
        k, v = mx.split(kv, 2, axis=-1)  # each (B, S2, heads, hd)

        # Q uses standard reshape (no interleaving)
        q = q.reshape(B, S1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S2, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S2, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, S1, -1)
        return self.out_proj(out)


class AttentionPool(nn.Module):
    """Attention pooling over features to produce a single vector.

    Uses a learnable query (positional_embedding) to cross-attend to context.

    Args:
        spatial_dim: Number of spatial positions (text_len).
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        output_dim: Output dimension (default: embed_dim).
    """

    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        output_dim = output_dim or embed_dim
        self.positional_embedding = mx.zeros((spatial_dim + 1, embed_dim))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def __call__(self, x: mx.array, attention_mask=None) -> mx.array:
        """
        Args:
            x: (B, L, D) input features.
        Returns:
            (B, D) pooled output.
        """
        B, L, D = x.shape
        # Prepend mean token
        mean_token = x.mean(axis=1, keepdims=True)  # (B, 1, D)
        x_with_mean = mx.concatenate([mean_token, x], axis=1)  # (B, L+1, D)

        # Add positional embedding
        x_with_mean = x_with_mean + self.positional_embedding[None, :, :].astype(x.dtype)

        # Cross-attention: query is first token, key/value is all
        q = self.q_proj(x_with_mean[:, :1, :])  # (B, 1, D)
        k = self.k_proj(x_with_mean)  # (B, L+1, D)
        v = self.v_proj(x_with_mean)  # (B, L+1, D)

        q = q.reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L + 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L + 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, -1)  # (B, D)
        return self.c_proj(out)


class FinalLayer(nn.Module):
    """Final layer: LayerNorm -> remove prepended token -> Linear.

    Args:
        hidden_size: Hidden dimension.
        out_channels: Output channels.
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm_final(x)
        x = x[:, 1:]  # Remove prepended conditioning token
        return self.linear(x)


# --------------------------------------------------------------------------- #
# DiT Block
# --------------------------------------------------------------------------- #


class HunYuanDiTBlock(nn.Module):
    """Single HunYuanDiT block with optional skip connection and MoE.

    Args:
        hidden_size: Hidden dimension.
        num_heads: Number of attention heads.
        context_dim: Context (text/image) embedding dimension.
        qk_norm: Whether to use QK normalization.
        qkv_bias: Whether QKV projections have bias.
        skip_connection: Whether this block has skip connections (U-Net second half).
        timestep_modulate: Whether to use timestep-based modulation.
        use_moe: Whether to use MoE instead of MLP.
        num_experts: Number of experts for MoE.
        moe_top_k: Top-k routing for MoE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_dim: int = 1024,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        skip_connection: bool = False,
        timestep_modulate: bool = False,
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
    ):
        super().__init__()
        # Self-attention (no bias on Q/K/V, bias on out_proj)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)

        # Cross-attention (no bias on Q/K/V, bias on out_proj)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, context_dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm)

        # FFN / MoE
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.use_moe = use_moe
        if use_moe:
            def make_expert():
                return MLP(hidden_size)

            shared = MLP(hidden_size)
            self.moe = MoELayer(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=moe_top_k,
                expert_fn=make_expert,
                shared_expert=shared,
            )
        else:
            self.mlp = MLP(hidden_size)

        # Timestep modulation
        self.timestep_modulate = timestep_modulate
        if timestep_modulate:
            self.default_modulation_0 = nn.SiLU()
            self.default_modulation_1 = nn.Linear(hidden_size, hidden_size)

        # Skip connection (U-Net second half)
        if skip_connection:
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        else:
            self.skip_linear = None

    def __call__(
        self,
        x: mx.array,
        c: mx.array = None,
        text_states: mx.array = None,
        skip_value: mx.array = None,
    ) -> mx.array:
        # Skip connection
        if self.skip_linear is not None and skip_value is not None:
            cat = mx.concatenate([skip_value, x], axis=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # Timestep modulation
        if self.timestep_modulate and c is not None:
            shift = self.default_modulation_1(self.default_modulation_0(c))
            x = x + shift

        # Self-attention
        x = x + self.attn1(self.norm1(x))

        # Cross-attention
        x = x + self.attn2(self.norm2(x), text_states)

        # FFN or MoE
        normed = self.norm3(x)
        if self.use_moe:
            x = x + self.moe(normed)
        else:
            x = x + self.mlp(normed)

        return x


# --------------------------------------------------------------------------- #
# Full DiT Model
# --------------------------------------------------------------------------- #


class HunYuanDiTPlain(nn.Module):
    """HunYuanDiT shape transformer with U-Net skip connections.

    Args:
        input_size: Number of input tokens (latent sequence length).
        in_channels: Input/output channels per token.
        hidden_size: Hidden dimension.
        context_dim: Context (DINOv2) embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        qk_norm: Whether to use QK normalization.
        text_len: Length of conditioning text/image tokens.
        use_pos_emb: Whether to use positional embedding.
        use_attention_pooling: Whether to use attention pooling for conditioning.
        qkv_bias: Whether QKV projections have bias.
        num_moe_layers: Number of final blocks that use MoE.
        num_experts: Number of experts in MoE blocks.
        moe_top_k: Top-k routing for MoE.
    """

    def __init__(
        self,
        input_size: int = 1024,
        in_channels: int = 4,
        hidden_size: int = 1024,
        context_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        qk_norm: bool = True,
        text_len: int = 257,
        use_pos_emb: bool = False,
        use_attention_pooling: bool = False,
        qkv_bias: bool = False,
        num_moe_layers: int = 6,
        num_experts: int = 8,
        moe_top_k: int = 2,
        # Ignored kwargs for compatibility
        mlp_ratio: float = 4.0,
        norm_type: str = "layer",
        qk_norm_type: str = "rms",
        with_decoupled_ca: bool = False,
        additional_cond_hidden_state: int = 768,
        decoupled_ca_dim: int = 16,
        decoupled_ca_weight: float = 1.0,
        guidance_cond_proj_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.use_pos_emb = use_pos_emb
        self.use_attention_pooling = use_attention_pooling

        # Input projection
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)

        # Positional embedding (disabled in v2.1)
        if use_pos_emb:
            pos = np.arange(input_size, dtype=np.float32)
            pos_embed = _get_1d_sincos_pos_embed(hidden_size, pos)
            self.pos_embed = mx.array(pos_embed[None])  # (1, input_size, hidden_size)
        else:
            self.pos_embed = None

        # Attention pooling for conditioning
        if use_attention_pooling:
            self.pooler = AttentionPool(text_len, context_dim, num_heads=8, output_dim=1024)
            self.extra_embedder_0 = nn.Linear(1024, hidden_size * 4)
            self.extra_embedder_1 = nn.SiLU()
            self.extra_embedder_2 = nn.Linear(hidden_size * 4, hidden_size)

        # Transformer blocks
        self.blocks = [
            HunYuanDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                context_dim=context_dim,
                qk_norm=qk_norm,
                qkv_bias=qkv_bias,
                skip_connection=(layer > depth // 2),
                timestep_modulate=False,
                use_moe=(depth - layer <= num_moe_layers),
                num_experts=num_experts,
                moe_top_k=moe_top_k,
            )
            for layer in range(depth)
        ]

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def __call__(self, x: mx.array, t: mx.array, contexts: dict, **kwargs) -> mx.array:
        """Forward pass.

        Args:
            x: (B, seq_len, in_channels) latent tokens.
            t: (B,) timestep values (0 to 1 range).
            contexts: Dict with 'main' key containing DINOv2 features (B, L, D).

        Returns:
            (B, seq_len, in_channels) predicted velocity.
        """
        cond = contexts["main"]

        # Timestep embedding
        t_emb = self.t_embedder(t)  # (B, 1, hidden_size)

        # Input projection
        x = self.x_embedder(x)

        # Optional positional embedding
        if self.use_pos_emb and self.pos_embed is not None:
            x = x + self.pos_embed.astype(x.dtype)

        # Attention pool conditioning + extra embedder
        if self.use_attention_pooling:
            extra_vec = self.pooler(cond, None)  # (B, D)
            extra = self.extra_embedder_2(
                self.extra_embedder_1(self.extra_embedder_0(extra_vec))
            )  # (B, hidden_size)
            c = t_emb + extra[:, None, :]  # (B, 1, hidden_size)
        else:
            c = t_emb

        # Prepend conditioning token
        x = mx.concatenate([c, x], axis=1)  # (B, 1+seq_len, hidden_size)

        # U-Net loop
        skip_value_list = []
        for layer_idx, block in enumerate(self.blocks):
            skip_value = None if layer_idx <= self.depth // 2 else skip_value_list.pop()
            x = block(x, c, cond, skip_value=skip_value)
            if layer_idx < self.depth // 2:
                skip_value_list.append(x)

        # Final layer (removes prepended token internally)
        x = self.final_layer(x)
        return x

    @classmethod
    def from_pretrained(cls, weights_path: str, config: dict) -> "HunYuanDiTPlain":
        """Load from safetensors weights.

        Handles key remapping from checkpoint format to MLX module format:
        - Strips ``dit.`` prefix
        - ``t_embedder.mlp.0`` -> ``t_embedder.mlp_0``
        - ``t_embedder.mlp.2`` -> ``t_embedder.mlp_2``
        - ``moe.gate.weight`` -> ``moe.gate.gate.weight`` (raw param to nn.Linear)
        - ``moe.experts.E.net.0.proj`` -> ``moe.experts.E.fc1``
        - ``moe.experts.E.net.2`` -> ``moe.experts.E.fc2``
        - ``moe.shared_experts`` -> ``moe.shared_expert``

        Args:
            weights_path: Path to dit.safetensors.
            config: Dict with model config.

        Returns:
            Loaded HunYuanDiTPlain model.
        """
        model = cls(**config)
        raw_weights = mx.load(weights_path)

        remapped = {}
        for key, value in raw_weights.items():
            k = key
            # Strip top-level prefix
            if k.startswith("dit."):
                k = k[len("dit."):]

            # t_embedder MLP: Sequential index -> attribute name
            k = k.replace("t_embedder.mlp.0.", "t_embedder.mlp_0.")
            k = k.replace("t_embedder.mlp.2.", "t_embedder.mlp_2.")

            # MoE gate: checkpoint stores raw param, module uses nn.Linear
            # checkpoint: blocks.N.moe.gate.weight -> module: blocks.N.moe.gate.gate.weight
            k = re.sub(
                r"(blocks\.\d+\.moe)\.gate\.weight$",
                r"\1.gate.gate.weight",
                k,
            )

            # MoE experts: diffusers FeedForward naming -> simple MLP naming
            # net.0.proj.weight/bias -> fc1.weight/bias
            k = re.sub(
                r"(blocks\.\d+\.moe\.experts\.\d+)\.net\.0\.proj\.",
                r"\1.fc1.",
                k,
            )
            # net.2.weight/bias -> fc2.weight/bias
            k = re.sub(
                r"(blocks\.\d+\.moe\.experts\.\d+)\.net\.2\.",
                r"\1.fc2.",
                k,
            )

            # MoE shared experts: shared_experts -> shared_expert (plural -> singular)
            # Also apply the same net.0.proj / net.2 remapping
            k = re.sub(
                r"(blocks\.\d+\.moe)\.shared_experts\.net\.0\.proj\.",
                r"\1.shared_expert.fc1.",
                k,
            )
            k = re.sub(
                r"(blocks\.\d+\.moe)\.shared_experts\.net\.2\.",
                r"\1.shared_expert.fc2.",
                k,
            )

            remapped[k] = value

        model.load_weights(list(remapped.items()))
        # Force materialization of lazy parameters
        mx.eval(model.parameters())  # noqa: S307 - mx.eval triggers MLX lazy computation
        return model


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #


def _get_1d_sincos_pos_embed(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal positional embedding."""
    half = embed_dim // 2
    omega = np.arange(half, dtype=np.float64) / half
    omega = 1.0 / (10000.0**omega)
    out = np.outer(positions.ravel(), omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb.astype(np.float32)
