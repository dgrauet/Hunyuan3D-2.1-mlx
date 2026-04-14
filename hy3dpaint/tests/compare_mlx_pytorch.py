"""Compare PyTorch and MLX UNet/VAE outputs numerically.

Loads both models, runs the same input through each, and compares
intermediate activations to find where divergence occurs.

Usage:
    cd /Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint
    python tests/compare_mlx_pytorch.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import gc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# PyTorch source weights for ground-truth comparison. Override via env var
# or pass --pt-weights on CLI if you keep them elsewhere.
PT_WEIGHTS_ROOT = os.environ.get(
    "HUNYUAN3D_PT_WEIGHTS_ROOT",
    os.path.expanduser(
        "~/Work/mlx-forge/downloads/hunyuan3d-2.1/hunyuan3d-paintpbr-v2-1"
    ),
)
PT_UNET_DIR = os.path.join(PT_WEIGHTS_ROOT, "unet")
PT_VAE_DIR = os.path.join(PT_WEIGHTS_ROOT, "vae")

# MLX weights: resolve HF repo ID (cached under ~/.cache/huggingface/hub).
# Override with HUNYUAN3D_MLX_WEIGHTS_DIR to use a local checkout.
_env_mlx = os.environ.get("HUNYUAN3D_MLX_WEIGHTS_DIR")
if _env_mlx:
    MLX_WEIGHTS_DIR = _env_mlx
else:
    from huggingface_hub import snapshot_download
    MLX_WEIGHTS_DIR = snapshot_download(
        repo_id="dgrauet/hunyuan3d-2.1-mlx",
        allow_patterns=[
            "paint_unet.safetensors",
            "paint_vae.safetensors",
            "paint_dino.safetensors",
        ],
    )

# Latent size (small so both models fit comfortably)
LATENT_H, LATENT_W = 8, 8
IN_CHANNELS = 12          # 4 latent + 4 normal + 4 position
TEXT_LEN = 77
CROSS_DIM = 1024
SEED = 42
TOL = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def compare_tensors(name: str, pt_np: np.ndarray, mlx_np: np.ndarray,
                    tol: float = TOL, detail: bool = True) -> dict:
    """Compare two numpy arrays, print a MATCH/MISMATCH verdict, return stats."""
    result = {"name": name, "pt_shape": pt_np.shape, "mlx_shape": mlx_np.shape}
    if pt_np.shape != mlx_np.shape:
        print(f"  [{name}] SHAPE MISMATCH: PT {pt_np.shape} vs MLX {mlx_np.shape}")
        result["status"] = "SHAPE_MISMATCH"
        return result

    diff = np.abs(pt_np - mlx_np)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    denom = np.maximum(np.abs(pt_np), 1e-8)
    rel = diff / denom
    max_rel = float(rel.max())
    pt_norm = float(np.linalg.norm(pt_np.ravel()))
    mlx_norm = float(np.linalg.norm(mlx_np.ravel()))

    status = "MATCH" if max_diff < tol else "MISMATCH"
    print(f"  [{name}] shape={pt_np.shape}  max_abs={max_diff:.6e}  "
          f"max_rel={max_rel:.6e}  mean_abs={mean_diff:.6e}  "
          f"pt_norm={pt_norm:.4f}  mlx_norm={mlx_norm:.4f}  [{status}]")
    if detail and max_diff > tol:
        flat_idx = int(np.argmax(diff.ravel()))
        coords = np.unravel_index(flat_idx, diff.shape)
        print(f"         worst at {coords}: PT={pt_np[coords]:.6f} "
              f"MLX={mlx_np[coords]:.6f}")
        print(f"         first 6 PT : {pt_np.ravel()[:6]}")
        print(f"         first 6 MLX: {mlx_np.ravel()[:6]}")

    result.update({
        "status": status,
        "max_abs": max_diff,
        "max_rel": max_rel,
        "mean_abs": mean_diff,
        "pt_norm": pt_norm,
        "mlx_norm": mlx_norm,
    })
    return result


def pt_to_nhwc(t):
    """Convert PyTorch NCHW tensor to NHWC numpy."""
    import torch  # noqa: F401
    arr = t.detach().cpu().float().numpy()
    if arr.ndim == 4:
        return arr.transpose(0, 2, 3, 1)
    return arr


def nhwc_to_nchw(arr):
    return arr.transpose(0, 3, 1, 2)


def mx_sync(x):
    import mlx.core as mx
    mx.eval(x)


# ---------------------------------------------------------------------------
# PyTorch UNet loader: force in_channels=12 despite config saying 4
# ---------------------------------------------------------------------------

def load_pt_unet():
    """Build a fresh UNet2DConditionModel (in_channels=12) and load matching
    weights from the bundled checkpoint.

    The checkpoint stores the full 2.5D model with keys prefixed ``unet.``
    (plus an extra ``unet_dual.`` copy). We strip the prefix, drop keys that
    don't correspond to standard diffusers UNet submodules, and load only the
    matching subset. Everything extra (``attn_multiview``, ``attn_refview``,
    ``.transformer.``, ``_mr``, etc.) is simply ignored.
    """
    import torch
    from diffusers import UNet2DConditionModel

    # Read config and override in_channels=12
    import json
    with open(os.path.join(PT_UNET_DIR, "config.json")) as f:
        cfg = json.load(f)
    cfg["in_channels"] = IN_CHANNELS
    # Remove diffusers-internal keys that from_config dislikes
    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith("_")}

    unet_pt = UNet2DConditionModel.from_config(cfg_clean)
    unet_pt = unet_pt.to(torch.float32)
    unet_pt.eval()
    assert unet_pt.conv_in.weight.shape[1] == IN_CHANNELS

    # Load checkpoint
    sd = torch.load(
        os.path.join(PT_UNET_DIR, "diffusion_pytorch_model.bin"),
        map_location="cpu",
        weights_only=False,
    )

    model_keys = set(unet_pt.state_dict().keys())
    stripped: dict = {}
    # In the 2.5D checkpoint, the vanilla BasicTransformerBlock lives one
    # level deeper: ``transformer_blocks.0.transformer.<subkey>`` instead of
    # ``transformer_blocks.0.<subkey>``. We rewrite those keys so they map
    # onto the standard diffusers UNet state dict.
    for k, v in sd.items():
        if not k.startswith("unet."):
            continue
        nk = k[len("unet."):]
        # Collapse the 2.5D wrapper: "...transformer_blocks.0.transformer.X"
        # -> "...transformer_blocks.0.X"
        nk2 = nk.replace(".transformer_blocks.0.transformer.",
                         ".transformer_blocks.0.")
        # Skip 2.5D-only extras: attn_multiview / attn_refview / attn_dino /
        # attn1.processor / *_mr weights — these don't exist in a plain
        # UNet2DConditionModel state dict.
        if any(tag in nk2 for tag in (
            ".attn_multiview.", ".attn_refview.", ".attn_dino.",
            ".processor.",
        )):
            continue
        if nk2 in model_keys:
            stripped[nk2] = v

    missing, unexpected = unet_pt.load_state_dict(stripped, strict=False)
    print(f"  PT UNet: loaded {len(stripped)} weights "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"    first missing: {missing[:3]}")
    return unet_pt


def load_mlx_unet_for_comparison():
    """Create MLX UNet at in_channels=12 and load ONLY stock-UNet weights.

    We do NOT call _enhance_unet here so that the MLX model matches the
    plain diffusers UNet2DConditionModel topology (no 2.5D modules).
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from hunyuanpaintpbr_mlx.unet.unet_mlx import UNet2DConditionModelMLX

    unet_mlx = UNet2DConditionModelMLX(
        in_channels=IN_CHANNELS,
        out_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        cross_attention_dim=CROSS_DIM,
        attention_head_dim=(5, 10, 20, 20),  # matches PT config
    )

    unet_w = dict(mx.load(os.path.join(MLX_WEIGHTS_DIR, "paint_unet.safetensors")))
    stripped = {}
    for k, v in unet_w.items():
        if k.startswith("unet."):
            stripped[k[5:]] = v

    model_keys = set(k for k, _ in tree_flatten(unet_mlx.parameters()))
    matched = [(k, v) for k, v in stripped.items() if k in model_keys]
    unet_mlx.load_weights(matched)

    # cast to fp32 for fair comparison
    float32_params = [(k, v.astype(mx.float32))
                      for k, v in tree_flatten(unet_mlx.parameters())]
    unet_mlx.load_weights(float32_params)

    n_total = len(stripped)
    n_loaded = len(matched)
    print(f"  Loaded {n_loaded}/{n_total} MLX UNet weights "
          f"(ignored {n_total - n_loaded} 2.5D/extra tensors)")
    return unet_mlx


# ---------------------------------------------------------------------------
# Part 1: Weight spot-check
# ---------------------------------------------------------------------------

def compare_weights(results: list):
    separator("PART 1: WEIGHT SPOT CHECK")

    import mlx.core as mx

    unet_pt = load_pt_unet()
    pt_state = unet_pt.state_dict()

    print(f"  PT UNet in_channels (weight): {unet_pt.conv_in.weight.shape[1]}")
    print(f"  PT attention_head_dim (config): {unet_pt.config.attention_head_dim}")
    print(f"  PT use_linear_projection: {unet_pt.config.use_linear_projection}")

    mlx_raw = dict(mx.load(os.path.join(MLX_WEIGHTS_DIR, "paint_unet.safetensors")))
    mlx_weights = {}
    for k, v in mlx_raw.items():
        if k.startswith("unet."):
            mlx_weights[k[5:]] = v
    del mlx_raw

    keys = [
        "conv_in.weight",
        "conv_in.bias",
        "time_embedding.linear_1.weight",
        "time_embedding.linear_1.bias",
        "time_embedding.linear_2.weight",
        "conv_out.weight",
        "conv_out.bias",
        "conv_norm_out.weight",
        "down_blocks.0.resnets.0.norm1.weight",
        "down_blocks.0.resnets.0.conv1.weight",
        "down_blocks.0.resnets.0.time_emb_proj.weight",
        "down_blocks.0.attentions.0.norm.weight",
        "down_blocks.0.attentions.0.proj_in.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.weight",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    ]

    for key in keys:
        if key not in pt_state:
            print(f"  [{key}] NOT IN PT state_dict")
            continue
        if key not in mlx_weights:
            print(f"  [{key}] NOT IN MLX weights")
            continue

        pt_w = pt_state[key].detach().cpu().float().numpy()
        mlx_w = np.array(mlx_weights[key].astype(mx.float32))

        if pt_w.ndim == 4:
            # PT (O, I, kH, kW) -> MLX (O, kH, kW, I)
            pt_w_n = pt_w.transpose(0, 2, 3, 1)
            results.append(compare_tensors(f"W:{key}", pt_w_n, mlx_w))
        elif pt_w.ndim == 2:
            if pt_w.shape == mlx_w.shape:
                results.append(compare_tensors(f"W:{key}", pt_w, mlx_w))
            elif pt_w.T.shape == mlx_w.shape:
                results.append(compare_tensors(f"W:{key}(T)", pt_w.T, mlx_w))
            else:
                print(f"  [W:{key}] SHAPE MISMATCH: PT {pt_w.shape} MLX {mlx_w.shape}")
        else:
            results.append(compare_tensors(f"W:{key}", pt_w, mlx_w))

    del unet_pt, pt_state, mlx_weights
    gc.collect()


# ---------------------------------------------------------------------------
# Part 2: Attention head dim analysis
# ---------------------------------------------------------------------------

def check_attention_head_dims():
    separator("PART 2: ATTENTION HEAD DIM ANALYSIS")

    unet_pt = load_pt_unet()

    print(f"  PT config.attention_head_dim: {unet_pt.config.attention_head_dim}")
    for i, block in enumerate(unet_pt.down_blocks):
        if hasattr(block, "attentions") and len(block.attentions) > 0:
            tb = block.attentions[0].transformer_blocks[0]
            print(f"  down_blocks[{i}]: channels={unet_pt.config.block_out_channels[i]}, "
                  f"heads={tb.attn1.heads}, to_q={tuple(tb.attn1.to_q.weight.shape)}")

    if hasattr(unet_pt.mid_block, "attentions"):
        tb = unet_pt.mid_block.attentions[0].transformer_blocks[0]
        print(f"  mid_block: heads={tb.attn1.heads}, "
              f"to_q={tuple(tb.attn1.to_q.weight.shape)}")

    del unet_pt
    gc.collect()


# ---------------------------------------------------------------------------
# Part 3: Timestep embedding detail
# ---------------------------------------------------------------------------

def compare_timestep_embedding_detail(results: list):
    separator("PART 3: TIMESTEP EMBEDDING")
    import torch
    import mlx.core as mx
    from diffusers.models.embeddings import get_timestep_embedding as pt_get
    from hunyuanpaintpbr_mlx.unet.blocks_mlx import get_timestep_embedding as mlx_get

    t = 500.0
    dim = 320

    emb_pt = pt_get(torch.tensor([t]), dim,
                    flip_sin_to_cos=True, downscale_freq_shift=0).numpy()[0]
    emb_mlx = np.array(mlx_get(mx.array([t]), dim))[0]

    results.append(compare_tensors("timestep_sinusoidal", emb_pt, emb_mlx))


# ---------------------------------------------------------------------------
# Part 4: UNet forward pass comparison
# ---------------------------------------------------------------------------

def compare_unet_forward(results: list) -> dict:
    separator("PART 4: UNET FORWARD PASS")

    import torch
    import mlx.core as mx

    # --- Deterministic inputs ---
    rng = np.random.RandomState(SEED)
    sample_nhwc = rng.randn(1, LATENT_H, LATENT_W, IN_CHANNELS).astype(np.float32) * 0.1
    text_np = np.zeros((1, TEXT_LEN, CROSS_DIM), dtype=np.float32)
    timestep_val = 500

    sample_pt = torch.tensor(nhwc_to_nchw(sample_nhwc))
    text_pt = torch.tensor(text_np)
    timestep_pt = torch.tensor([timestep_val])

    # --- PyTorch ---
    print("Loading PyTorch UNet...")
    unet_pt = load_pt_unet()

    pt_intermediates: dict = {}

    def mk_hook(name, is_block=False):
        def _h(module, inp, output):
            val = output[0] if (is_block and isinstance(output, tuple)) else output
            pt_intermediates[name] = val.detach()
        return _h

    handles = []
    handles.append(unet_pt.conv_in.register_forward_hook(mk_hook("conv_in")))
    handles.append(unet_pt.down_blocks[0].register_forward_hook(
        mk_hook("down_block_0", is_block=True)))
    handles.append(unet_pt.mid_block.register_forward_hook(mk_hook("mid_block")))
    handles.append(unet_pt.up_blocks[0].register_forward_hook(mk_hook("up_block_0")))
    handles.append(unet_pt.conv_out.register_forward_hook(mk_hook("conv_out")))

    print("Running PyTorch forward pass...")
    with torch.no_grad():
        out_pt = unet_pt(sample_pt, timestep_pt, encoder_hidden_states=text_pt).sample
    for h in handles:
        h.remove()

    pt_intermediates["final"] = out_pt.detach()
    pt_results = {k: pt_to_nhwc(v) for k, v in pt_intermediates.items()}

    print(f"  PT final (NHWC): {pt_results['final'].shape}  "
          f"min={pt_results['final'].min():.4f} max={pt_results['final'].max():.4f}")

    del unet_pt, out_pt
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- MLX ---
    print("\nLoading MLX UNet...")
    unet_mlx = load_mlx_unet_for_comparison()

    from hunyuanpaintpbr_mlx.unet.blocks_mlx import get_timestep_embedding

    sample_mlx = mx.array(sample_nhwc)
    text_mlx = mx.array(text_np)
    timestep_mlx = mx.array([timestep_val])

    mlx_results: dict = {}

    # Timestep embedding
    t_emb = get_timestep_embedding(timestep_mlx, unet_mlx.time_proj_dim)
    emb = unet_mlx.time_embedding(t_emb)
    mx_sync(emb)

    # Step 1: conv_in
    x = unet_mlx.conv_in(sample_mlx)
    mx_sync(x)
    mlx_results["conv_in"] = np.array(x)

    # Step 2: down blocks (capture first)
    down_res = [x]
    block = unet_mlx.down_blocks[0]
    if getattr(block, "has_cross_attention", False):
        x, res = block(x, emb, text_mlx)
    else:
        x, res = block(x, emb)
    down_res.extend(res)
    mx_sync(x)
    mlx_results["down_block_0"] = np.array(x)

    for block in unet_mlx.down_blocks[1:]:
        if getattr(block, "has_cross_attention", False):
            x, res = block(x, emb, text_mlx)
        else:
            x, res = block(x, emb)
        down_res.extend(res)

    # Step 3: mid
    x = unet_mlx.mid_block(x, emb, text_mlx)
    mx_sync(x)
    mlx_results["mid_block"] = np.array(x)

    # Step 4: up blocks (capture first)
    first_up = True
    for block in unet_mlx.up_blocks:
        n_res = len(block.resnets)
        res_samples = down_res[-n_res:]
        down_res = down_res[:-n_res]
        if getattr(block, "has_cross_attention", False):
            x = block(x, emb, res_samples, text_mlx)
        else:
            x = block(x, emb, res_samples)
        if first_up:
            mx_sync(x)
            mlx_results["up_block_0"] = np.array(x)
            first_up = False

    # Step 5: out
    h = unet_mlx.conv_act(unet_mlx.conv_norm_out(x))
    out = unet_mlx.conv_out(h)
    mx_sync(out)
    mlx_results["conv_out"] = np.array(out)
    mlx_results["final"] = np.array(out)

    print(f"  MLX final: {mlx_results['final'].shape}  "
          f"min={mlx_results['final'].min():.4f} max={mlx_results['final'].max():.4f}")

    # --- Compare layer by layer ---
    separator("LAYER-BY-LAYER COMPARISON")
    order = ["conv_in", "down_block_0", "mid_block", "up_block_0", "conv_out", "final"]
    layer_results = {}
    divergence = None
    for name in order:
        pt_arr = pt_results.get(name)
        mlx_arr = mlx_results.get(name)
        if pt_arr is None or mlx_arr is None:
            print(f"  [{name}] missing  (PT={pt_arr is not None}, MLX={mlx_arr is not None})")
            continue
        r = compare_tensors(name, pt_arr, mlx_arr)
        results.append(r)
        layer_results[name] = r
        if divergence is None and r.get("status") == "MISMATCH":
            divergence = name

    del unet_mlx
    gc.collect()

    return {"layers": layer_results, "divergence": divergence,
            "pt": pt_results, "mlx": mlx_results}


# ---------------------------------------------------------------------------
# Part 5: Dig into the divergence point
# ---------------------------------------------------------------------------

def drill_down_first_layer(results: list):
    """If conv_in already diverges, compare conv_in weights directly.

    Also manually applies the exact PyTorch conv_in via numpy math, and
    compares both MLX and the reference numpy conv to PyTorch output.
    This isolates weight-layout vs numerical-op bugs.
    """
    separator("PART 5: DRILL-DOWN (CONV_IN ISOLATION)")

    import torch
    import torch.nn.functional as F
    import mlx.core as mx

    rng = np.random.RandomState(SEED)
    sample_nhwc = rng.randn(1, LATENT_H, LATENT_W, IN_CHANNELS).astype(np.float32) * 0.1
    sample_nchw = nhwc_to_nchw(sample_nhwc)

    unet_pt = load_pt_unet()
    w_pt = unet_pt.conv_in.weight.detach().cpu().float().numpy()  # (O, I, kH, kW)
    b_pt = unet_pt.conv_in.bias.detach().cpu().float().numpy()

    # PT reference
    with torch.no_grad():
        ref = F.conv2d(torch.tensor(sample_nchw),
                       torch.tensor(w_pt), torch.tensor(b_pt), padding=1)
    ref_nhwc = pt_to_nhwc(ref)

    # MLX with weights loaded from safetensors
    unet_mlx = load_mlx_unet_for_comparison()
    x_mlx = unet_mlx.conv_in(mx.array(sample_nhwc))
    mx_sync(x_mlx)
    mlx_out = np.array(x_mlx)

    # Direct MLX conv with transposed PT weights (sanity check)
    import mlx.nn as nn
    conv = nn.Conv2d(IN_CHANNELS, 320, 3, padding=1)
    conv.weight = mx.array(w_pt.transpose(0, 2, 3, 1))  # (O, kH, kW, I)
    conv.bias = mx.array(b_pt)
    direct = conv(mx.array(sample_nhwc))
    mx_sync(direct)
    direct_np = np.array(direct)

    # MLX conv_in weight stored
    mlx_w = np.array(unet_mlx.conv_in.weight)
    # Compare MLX stored weight with PT transposed weight
    print("  Weight tensor comparison (MLX stored vs PT transposed to OHWI):")
    compare_tensors("conv_in.weight", w_pt.transpose(0, 2, 3, 1), mlx_w)

    print("\n  Activation comparisons:")
    results.append(compare_tensors("conv_in PT-vs-MLX(loaded)", ref_nhwc, mlx_out))
    results.append(compare_tensors("conv_in PT-vs-MLX(direct weight set)", ref_nhwc, direct_np))

    del unet_pt, unet_mlx
    gc.collect()


# ---------------------------------------------------------------------------
# Part 6: Drill-down into down_block_0
# ---------------------------------------------------------------------------

def drill_down_block0(results: list):
    """Capture PT outputs at resnet-0, attention-0, resnet-1, attention-1,
    downsample inside down_blocks[0] and compare to MLX step-by-step.

    Also tries a 'corrected timestep' MLX run that uses the PyTorch temb
    instead of the MLX timestep embedding, to confirm/deny the timestep
    embedding bug as the root cause of downstream divergence.
    """
    separator("PART 6: DRILL-DOWN INTO down_blocks[0]")

    import torch
    import mlx.core as mx
    from diffusers.models.embeddings import get_timestep_embedding as pt_get

    rng = np.random.RandomState(SEED)
    sample_nhwc = rng.randn(1, LATENT_H, LATENT_W, IN_CHANNELS).astype(np.float32) * 0.1
    text_np = np.zeros((1, TEXT_LEN, CROSS_DIM), dtype=np.float32)
    timestep_val = 500
    sample_pt = torch.tensor(nhwc_to_nchw(sample_nhwc))

    # --- PT forward, hooking each sub-module of down_blocks[0] ---
    unet_pt = load_pt_unet()
    block = unet_pt.down_blocks[0]
    captured: dict = {}

    def hook(key):
        def _h(module, inp, output):
            if isinstance(output, tuple):
                output = output[0]
            captured[key] = output.detach()
        return _h

    handles = [
        block.resnets[0].register_forward_hook(hook("resnet0")),
        block.attentions[0].register_forward_hook(hook("attn0")),
        block.resnets[1].register_forward_hook(hook("resnet1")),
        block.attentions[1].register_forward_hook(hook("attn1")),
    ]
    if block.downsamplers:
        handles.append(
            block.downsamplers[0].register_forward_hook(hook("downsample"))
        )

    with torch.no_grad():
        unet_pt(sample_pt, torch.tensor([timestep_val]),
                encoder_hidden_states=torch.tensor(text_np))
    for h in handles:
        h.remove()

    pt_sub = {k: pt_to_nhwc(v) for k, v in captured.items()}

    # Also grab the PT timestep embedding for MLX "corrected" run
    pt_temb_sinus = pt_get(
        torch.tensor([timestep_val]).float(), 320,
        flip_sin_to_cos=True, downscale_freq_shift=0,
    )
    pt_emb = unet_pt.time_embedding(pt_temb_sinus).detach().cpu().float().numpy()

    del unet_pt
    gc.collect()

    # --- MLX forward: standard (buggy timestep) ---
    unet_mlx = load_mlx_unet_for_comparison()
    from hunyuanpaintpbr_mlx.unet.blocks_mlx import get_timestep_embedding as mlx_get

    sample_mlx = mx.array(sample_nhwc)
    text_mlx = mx.array(text_np)

    t_emb = mlx_get(mx.array([timestep_val]), unet_mlx.time_proj_dim)
    emb_buggy = unet_mlx.time_embedding(t_emb)

    x0 = unet_mlx.conv_in(sample_mlx)
    b = unet_mlx.down_blocks[0]
    # Inspect sub-modules individually
    r0 = b.resnets[0](x0, emb_buggy); mx_sync(r0)
    a0 = b.attentions[0](r0, text_mlx); mx_sync(a0)
    r1 = b.resnets[1](a0, emb_buggy); mx_sync(r1)
    a1 = b.attentions[1](r1, text_mlx); mx_sync(a1)

    mlx_sub = {
        "resnet0": np.array(r0),
        "attn0": np.array(a0),
        "resnet1": np.array(r1),
        "attn1": np.array(a1),
    }

    print("  [MLX using its own (buggy?) timestep embedding]")
    for k in ("resnet0", "attn0", "resnet1", "attn1"):
        if k in pt_sub and k in mlx_sub:
            results.append(compare_tensors(f"block0.{k}", pt_sub[k], mlx_sub[k]))

    # --- MLX with pytorch_compatible GroupNorm patched everywhere ---
    print("\n  [MLX with GroupNorm.pytorch_compatible=True applied to all GNs]")
    import mlx.nn as mnn

    def patch_gn(module):
        for _, m in module.named_modules():
            if isinstance(m, mnn.GroupNorm):
                m.pytorch_compatible = True
    patch_gn(unet_mlx)

    x0p = unet_mlx.conv_in(sample_mlx)
    r0p = b.resnets[0](x0p, emb_buggy); mx_sync(r0p)
    a0p = b.attentions[0](r0p, text_mlx); mx_sync(a0p)
    r1p = b.resnets[1](a0p, emb_buggy); mx_sync(r1p)
    a1p = b.attentions[1](r1p, text_mlx); mx_sync(a1p)
    for k, arr in [("resnet0", r0p), ("attn0", a0p),
                   ("resnet1", r1p), ("attn1", a1p)]:
        results.append(compare_tensors(
            f"block0.{k} (GN fix)", pt_sub[k], np.array(arr)))

    # Undo patch for the next experiment
    def unpatch_gn(module):
        for _, m in module.named_modules():
            if isinstance(m, mnn.GroupNorm):
                m.pytorch_compatible = False
    unpatch_gn(unet_mlx)

    # --- MLX forward: use PyTorch's temb instead ---
    print("\n  [MLX using PyTorch's timestep embedding]")
    emb_fixed = mx.array(pt_emb)
    r0f = b.resnets[0](x0, emb_fixed); mx_sync(r0f)
    a0f = b.attentions[0](r0f, text_mlx); mx_sync(a0f)
    r1f = b.resnets[1](a0f, emb_fixed); mx_sync(r1f)
    a1f = b.attentions[1](r1f, text_mlx); mx_sync(a1f)

    for k, arr in [("resnet0", r0f), ("attn0", a0f),
                   ("resnet1", r1f), ("attn1", a1f)]:
        if k in pt_sub:
            results.append(compare_tensors(
                f"block0.{k} (PT temb)", pt_sub[k], np.array(arr)))

    del unet_mlx
    gc.collect()


# ---------------------------------------------------------------------------
# Write markdown report
# ---------------------------------------------------------------------------

def write_report(results: list, forward_summary: dict, path: str):
    lines = [
        "# MLX vs PyTorch Numerical Comparison Report",
        "",
        f"- latent size: {LATENT_H}x{LATENT_W}",
        f"- in_channels: {IN_CHANNELS}",
        f"- tolerance: {TOL}",
        "",
        "## Layer-by-layer divergence",
        "",
    ]
    layers = forward_summary.get("layers", {}) if forward_summary else {}
    if layers:
        lines += [
            "| layer | status | max_abs | max_rel | mean_abs | pt_norm | mlx_norm |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, r in layers.items():
            lines.append(
                f"| {name} | {r.get('status','?')} | "
                f"{r.get('max_abs',float('nan')):.3e} | "
                f"{r.get('max_rel',float('nan')):.3e} | "
                f"{r.get('mean_abs',float('nan')):.3e} | "
                f"{r.get('pt_norm',float('nan')):.3f} | "
                f"{r.get('mlx_norm',float('nan')):.3f} |"
            )
        div = forward_summary.get("divergence")
        lines += ["", f"**First divergent layer:** {div or 'none — all layers MATCH'}", ""]

    lines += ["## All comparisons", "",
              "| comparison | status | max_abs | max_rel |",
              "| --- | --- | --- | --- |"]
    for r in results:
        if "status" not in r:
            continue
        lines.append(
            f"| {r['name']} | {r['status']} | "
            f"{r.get('max_abs', float('nan')):.3e} | "
            f"{r.get('max_rel', float('nan')):.3e} |"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  MLX vs PyTorch UNet Numerical Comparison")
    print(f"  Latent size: {LATENT_H}x{LATENT_W}  in_channels={IN_CHANNELS}")
    print("=" * 70)

    results: list = []
    forward_summary: dict = {}

    sections = [
        ("Weight spot check", lambda: compare_weights(results)),
        ("Attention head dims", check_attention_head_dims),
        ("Timestep embedding", lambda: compare_timestep_embedding_detail(results)),
        ("UNet forward pass",
         lambda: forward_summary.update(compare_unet_forward(results) or {})),
        ("Conv_in drill down", lambda: drill_down_first_layer(results)),
        ("down_block_0 drill down", lambda: drill_down_block0(results)),
    ]

    for name, func in sections:
        try:
            func()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    separator("SUMMARY")
    print(f"  First divergent layer: "
          f"{forward_summary.get('divergence') or 'none'}")

    report_path = os.path.join(os.path.dirname(__file__), "comparison_report_auto.md")
    try:
        write_report(results, forward_summary, report_path)
    except Exception as e:
        print(f"  Failed to write report: {e}")
