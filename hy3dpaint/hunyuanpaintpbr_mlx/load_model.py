"""Load HunyuanPaintPBR models from converted MLX weights.

Handles the full weight loading including 2.5D attention modules
that get dynamically attached to transformer blocks.
"""

import os
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
    # Get head config from the block's existing attn1 (set by UNet with correct head_dims)
    heads = block.attn1.heads
    head_dim = block.attn1.dim_head

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
    """Add 2.5D modules to all transformer blocks in a UNet.

    Also assigns a unique ``_block_id`` (str) to each transformer block so
    that the capture-based reference feature extraction can key features
    without monkey-patching ``__call__``.
    """
    counter = 0

    # Down blocks
    for bi, block in enumerate(unet.down_blocks):
        if hasattr(block, "attentions"):
            for ai, attn_block in enumerate(block.attentions):
                dim = attn_block.in_channels
                for ti, tb in enumerate(attn_block.transformer_blocks):
                    _enhance_transformer_block(tb, dim, cross_attention_dim)
                    tb._block_id = str(counter)
                    counter += 1

    # Mid block
    if hasattr(unet.mid_block, "attentions"):
        for ai, attn_block in enumerate(unet.mid_block.attentions):
            dim = attn_block.in_channels
            for ti, tb in enumerate(attn_block.transformer_blocks):
                _enhance_transformer_block(tb, dim, cross_attention_dim)
                tb._block_id = str(counter)
                counter += 1

    # Up blocks
    for bi, block in enumerate(unet.up_blocks):
        if hasattr(block, "attentions"):
            for ai, attn_block in enumerate(block.attentions):
                dim = attn_block.in_channels
                for ti, tb in enumerate(attn_block.transformer_blocks):
                    _enhance_transformer_block(tb, dim, cross_attention_dim)
                    tb._block_id = str(counter)
                    counter += 1


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

    Instead of monkey-patching ``__call__`` (which hangs on Apple Silicon),
    this passes a ``_capture_dict`` through kwargs.  Each
    ``BasicTransformerBlock`` that has a ``_block_id`` attribute will store
    its post-self-attention hidden states into this dict during the forward
    pass.

    Args:
        unet: The UNet (main or dual) to extract features from.
              Transformer blocks must have ``_block_id`` set (done by
              ``_enhance_unet``).
        ref_latent: (1, H, W, 4) reference image latent.
        ref_text: (1, 77, C) text embeddings for reference.
        condition_latents: (1, H, W, 8) optional normal+position latent concat.

    Returns:
        Dict mapping block id (str) to (1, L, C) feature tensors.
    """
    # Prepare input
    if condition_latents is not None:
        unet_input = mx.concatenate([ref_latent, condition_latents], axis=-1)
    else:
        # Pad to 12 channels with zeros
        pad = mx.zeros((*ref_latent.shape[:-1], 8))
        unet_input = mx.concatenate([ref_latent, pad], axis=-1)

    # Dict that BasicTransformerBlock will populate during the forward pass
    capture_dict: dict = {}

    # Forward pass at timestep 0 (frozen reference).
    # The _capture_dict kwarg is propagated through UNet -> blocks -> transformer
    # blocks via **kwargs.
    _ = unet(
        unet_input,
        mx.array([0]),
        ref_text,
        _capture_dict=capture_dict,
        _ref_n_views=1,
        n_views=0,   # disable multiview / MDA paths during capture
        n_pbr=1,
    )
    mx.synchronize()

    return capture_dict


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
    def from_pretrained(
        weights_source: str = "dgrauet/hunyuan3d-2.1-mlx-mlx",
    ) -> "HunyuanPaintModelMLX":
        """Load all components from converted weights.

        Args:
            weights_source: Either a HuggingFace repo ID (e.g.
                ``dgrauet/hunyuan3d-2.1-mlx``) which will be downloaded
                via ``huggingface_hub.snapshot_download`` and cached
                locally, or an absolute path to a local directory
                containing ``paint_{unet,vae,dino}.safetensors``.

        Environment:
            HUNYUAN3D_MLX_WEIGHTS_DIR — overrides ``weights_source`` if set.
        """
        env_override = os.environ.get("HUNYUAN3D_MLX_WEIGHTS_DIR")
        if env_override:
            weights_source = env_override

        if os.path.isdir(weights_source):
            weights_dir = Path(weights_source)
        else:
            # Treat as HF repo ID. Only fetch the paint_* files we need.
            from huggingface_hub import snapshot_download
            weights_dir = Path(snapshot_download(
                repo_id=weights_source,
                allow_patterns=[
                    "paint_unet.safetensors",
                    "paint_vae.safetensors",
                    "paint_dino.safetensors",
                    "config.json",
                    "split_model.json",
                ],
            ))

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
            attention_head_dim=(5, 10, 20, 20),
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
