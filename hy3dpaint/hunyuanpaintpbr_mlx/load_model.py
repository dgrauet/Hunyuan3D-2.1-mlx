"""Load HunyuanPaintPBR models from converted MLX weights.

Handles the full weight loading including 2.5D attention modules
that get dynamically attached to transformer blocks.
"""

from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .dino_mlx import DINOv2MLX
from .scheduler_mlx import UniPCMultistepSchedulerMLX
from .vae_mlx import AutoencoderKLMLX
from .unet.blocks_mlx import Attention
from .unet.unet_mlx import UNet2DConditionModelMLX
from .unet.attn_processor_mlx import ImageProjModel


# ---------------------------------------------------------------------------
# 2.5D module containers (match checkpoint key paths)
# ---------------------------------------------------------------------------


class MDAProcessor(nn.Module):
    """Material-dimension attention: extra Q/K/V/out for non-primary materials.

    Key path: attn1.processor.to_{q,k,v,out}_mr.*
    """
    def __init__(self, dim: int):
        super().__init__()
        self.to_q_mr = nn.Linear(dim, dim, bias=False)
        self.to_k_mr = nn.Linear(dim, dim, bias=False)
        self.to_v_mr = nn.Linear(dim, dim, bias=False)
        self.to_out_mr = nn.Linear(dim, dim)


class RefProcessor(nn.Module):
    """Reference attention: material-specific V and out projections.

    Key path: attn_refview.processor.to_{v,out}_mr.*
    """
    def __init__(self, dim: int):
        super().__init__()
        self.to_v_mr = nn.Linear(dim, dim, bias=False)
        self.to_out_mr = nn.Linear(dim, dim)


def _enhance_transformer_block(block, dim: int, cross_attention_dim: int):
    """Add 2.5D attention modules to a BasicTransformerBlock.

    Attaches modules with key paths matching the checkpoint:
    - block.attn1.processor -> MDAProcessor
    - block.attn_dino -> Attention (cross-attn to DINO features)
    - block.attn_multiview -> Attention (self-attn with RoPE)
    - block.attn_refview -> Attention + processor
    """
    head_dim = dim // 8 if dim >= 64 else dim
    heads = dim // head_dim

    # MDA processor on attn1
    block.attn1.processor = MDAProcessor(dim)

    # DINO cross-attention
    block.attn_dino = Attention(
        query_dim=dim,
        cross_attention_dim=cross_attention_dim,
        heads=heads,
        dim_head=head_dim,
    )

    # Multiview self-attention
    block.attn_multiview = Attention(
        query_dim=dim,
        heads=heads,
        dim_head=head_dim,
    )

    # Reference cross-attention
    block.attn_refview = Attention(
        query_dim=dim,
        heads=heads,
        dim_head=head_dim,
    )
    block.attn_refview.processor = RefProcessor(dim)


def _enhance_unet(unet: UNet2DConditionModelMLX, cross_attention_dim: int = 1024):
    """Add 2.5D modules to all transformer blocks in a UNet."""
    # Down blocks
    for block in unet.down_blocks:
        if hasattr(block, "attentions"):
            for attn_block in block.attentions:
                dim = attn_block.in_channels
                for tb in attn_block.transformer_blocks:
                    _enhance_transformer_block(tb, dim, cross_attention_dim)

    # Mid block
    if hasattr(unet.mid_block, "attentions"):
        for attn_block in unet.mid_block.attentions:
            dim = attn_block.in_channels
            for tb in attn_block.transformer_blocks:
                _enhance_transformer_block(tb, dim, cross_attention_dim)

    # Up blocks
    for block in unet.up_blocks:
        if hasattr(block, "attentions"):
            for attn_block in block.attentions:
                dim = attn_block.in_channels
                for tb in attn_block.transformer_blocks:
                    _enhance_transformer_block(tb, dim, cross_attention_dim)


# ---------------------------------------------------------------------------
# Reference feature extraction
# ---------------------------------------------------------------------------


def extract_reference_features(
    unet: "UNet2DConditionModelMLX",
    ref_latent: mx.array,
    ref_text: mx.array,
    condition_latents: mx.array = None,
) -> dict:
    """Run a forward pass through the UNet to capture transformer block outputs.

    Used for reference attention: the reference image is processed once,
    and the intermediate features at each transformer block are cached.
    These are then used as key/value context by attn_refview in the main pass.

    Args:
        unet: The UNet (main or dual) to extract features from.
        ref_latent: (1, H, W, 4) reference image latent.
        ref_text: (1, 77, C) text embeddings for reference.
        condition_latents: (1, H, W, 8) optional normal+position latent concat.

    Returns:
        Dict mapping block index (str) to (1, L, C) feature tensors.
    """
    # Prepare input
    if condition_latents is not None:
        unet_input = mx.concatenate([ref_latent, condition_latents], axis=-1)
    else:
        # Pad to 12 channels with zeros
        pad = mx.zeros((*ref_latent.shape[:-1], 8))
        unet_input = mx.concatenate([ref_latent, pad], axis=-1)

    # Collect features from each transformer block during forward
    features = {}
    block_idx = [0]

    def _collect_features(block, hidden_states, encoder_hidden_states, temb, **kw):
        """Wrapper that captures output and stores it."""
        result = _original_calls[id(block)](hidden_states, encoder_hidden_states, temb, **kw)
        features[str(block_idx[0])] = result
        block_idx[0] += 1
        return result

    # Monkey-patch transformer block __call__ temporarily
    _original_calls = {}
    all_blocks = []

    for down_block in unet.down_blocks:
        if hasattr(down_block, "attentions"):
            for attn in down_block.attentions:
                for tb in attn.transformer_blocks:
                    _original_calls[id(tb)] = tb.__call__
                    all_blocks.append(tb)

    if hasattr(unet.mid_block, "attentions"):
        for attn in unet.mid_block.attentions:
            for tb in attn.transformer_blocks:
                _original_calls[id(tb)] = tb.__call__
                all_blocks.append(tb)

    for up_block in unet.up_blocks:
        if hasattr(up_block, "attentions"):
            for attn in up_block.attentions:
                for tb in attn.transformer_blocks:
                    _original_calls[id(tb)] = tb.__call__
                    all_blocks.append(tb)

    # Patch
    for tb in all_blocks:
        orig = _original_calls[id(tb)]
        tb.__call__ = lambda hs, enc=None, t=None, _orig=orig, **kw: (
            features.update({str(block_idx[0]): _orig(hs, enc, t, **kw)}),
            block_idx.__setitem__(0, block_idx[0] + 1),
            features[str(block_idx[0] - 1)],
        )[-1]

    # Forward pass at timestep 0 (frozen reference)
    _ = unet(unet_input, mx.array([0]), ref_text)
    mx.synchronize()

    # Restore
    for tb in all_blocks:
        tb.__call__ = _original_calls[id(tb)]

    return features


# ---------------------------------------------------------------------------
# Full model loading
# ---------------------------------------------------------------------------


class HunyuanPaintModelMLX:
    """Container for all HunyuanPaintPBR model components.

    Loads weights from mlx-forge converted safetensors.
    """

    def __init__(
        self,
        unet: UNet2DConditionModelMLX,
        vae: AutoencoderKLMLX,
        dino: DINOv2MLX,
        image_proj: ImageProjModel,
        learned_text_clip: dict,
        scheduler: UniPCMultistepSchedulerMLX,
        cross_attention_dim: int = 1024,
    ):
        self.unet = unet
        self.vae = vae
        self.dino = dino
        self.image_proj = image_proj
        self.learned_text_clip = learned_text_clip
        self.scheduler = scheduler
        self.cross_attention_dim = cross_attention_dim

    @staticmethod
    def from_pretrained(weights_dir: str) -> "HunyuanPaintModelMLX":
        """Load all components from converted weights directory.

        Args:
            weights_dir: Path to directory with paint_unet.safetensors,
                         paint_vae.safetensors, paint_dino.safetensors.
        """
        weights_dir = Path(weights_dir)

        # --- VAE ---
        print("Loading VAE...")
        vae = AutoencoderKLMLX()
        vae_w = dict(mx.load(str(weights_dir / "paint_vae.safetensors")))
        vae.load_weights(list(vae_w.items()))
        del vae_w

        # --- DINOv2 ---
        print("Loading DINOv2...")
        dino = DINOv2MLX()
        dino_w = dict(mx.load(str(weights_dir / "paint_dino.safetensors")))
        dino.load_weights(list(dino_w.items()))
        del dino_w

        # --- UNet ---
        print("Loading UNet...")
        unet = UNet2DConditionModelMLX(
            in_channels=12, out_channels=4,
            block_out_channels=(320, 640, 1280, 1280),
            cross_attention_dim=1024,
            attention_head_dim=8,
        )
        # Add 2.5D modules to match checkpoint structure
        _enhance_unet(unet, cross_attention_dim=1024)

        unet_w = dict(mx.load(str(weights_dir / "paint_unet.safetensors")))
        # Strip 'unet.' prefix
        stripped = {}
        for k, v in unet_w.items():
            if k.startswith("unet."):
                stripped[k[5:]] = v

        # Load matching weights (ignore unet_dual for now)
        model_keys = set(k for k, _ in tree_flatten(unet.parameters()))

        # Handle .to_out_mr.0. → .to_out_mr. (ModuleList artifact)
        normalized = {}
        for k, v in stripped.items():
            nk = k.replace(".to_out_mr.0.", ".to_out_mr.")
            nk = nk.replace(".to_out_albedo.0.", ".to_out_albedo.")
            normalized[nk] = v

        matched = [(k, v) for k, v in normalized.items() if k in model_keys]
        unet.load_weights(matched)

        n_total = len(stripped)
        n_loaded = len(matched)
        n_extra = n_total - n_loaded
        print(f"  Loaded {n_loaded}/{n_total} UNet weights ({n_extra} unmatched)")

        # --- Learned text embeddings ---
        learned = {}
        for token in ("albedo", "mr", "ref"):
            key = f"learned_text_clip_{token}"
            if key in stripped:
                learned[token] = stripped[key]
            else:
                learned[token] = mx.zeros((77, 1024))

        # --- ImageProjModel ---
        image_proj = ImageProjModel(
            cross_attention_dim=1024,
            clip_embeddings_dim=1536,
            clip_extra_context_tokens=4,
        )
        proj_weights = [
            (k.replace("image_proj_model_dino.", ""), v)
            for k, v in stripped.items()
            if k.startswith("image_proj_model_dino.")
        ]
        if proj_weights:
            image_proj.load_weights(proj_weights)

        del unet_w, stripped

        scheduler = UniPCMultistepSchedulerMLX()

        model = HunyuanPaintModelMLX(
            unet=unet,
            vae=vae,
            dino=dino,
            image_proj=image_proj,
            learned_text_clip=learned,
            scheduler=scheduler,
        )
        print("All components loaded.")
        return model
