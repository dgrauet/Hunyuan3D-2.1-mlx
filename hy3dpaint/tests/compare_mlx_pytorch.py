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

PT_UNET_DIR = "/Users/dgrauet/Work/mlx-forge/downloads/hunyuan3d-2.1/hunyuan3d-paintpbr-v2-1/unet/"
PT_VAE_DIR = "/Users/dgrauet/Work/mlx-forge/downloads/hunyuan3d-2.1/hunyuan3d-paintpbr-v2-1/vae/"
MLX_WEIGHTS_DIR = "/Users/dgrauet/Work/mlx-forge/models/hunyuan3d-2.1-mlx"

# Latent size (small to fit both models in memory alongside both models)
LATENT_H, LATENT_W = 8, 8
SEED = 42


def separator(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def compare_tensors(name: str, pt_np: np.ndarray, mlx_np: np.ndarray, detail: bool = True):
    """Compare two numpy arrays and print diagnostics."""
    if pt_np.shape != mlx_np.shape:
        print(f"  [{name}] SHAPE MISMATCH: PT {pt_np.shape} vs MLX {mlx_np.shape}")
        return

    diff = np.abs(pt_np - mlx_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    pt_norm = np.linalg.norm(pt_np.ravel())
    mlx_norm = np.linalg.norm(mlx_np.ravel())

    status = "OK" if max_diff < 0.01 else ("WARN" if max_diff < 0.1 else "FAIL")
    print(f"  [{name}] max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  "
          f"pt_norm={pt_norm:.4f}  mlx_norm={mlx_norm:.4f}  [{status}]")
    if detail and max_diff > 0.001:
        # Show where the biggest differences are
        flat_idx = np.argmax(diff.ravel())
        coords = np.unravel_index(flat_idx, diff.shape)
        print(f"         worst at {coords}: PT={pt_np[coords]:.6f} MLX={mlx_np[coords]:.6f}")
        # Show first few values
        pt_flat = pt_np.ravel()[:8]
        mlx_flat = mlx_np.ravel()[:8]
        print(f"         first 8 PT:  {pt_flat}")
        print(f"         first 8 MLX: {mlx_flat}")


def pt_to_nhwc(t):
    """Convert PyTorch NCHW tensor to NHWC numpy."""
    return t.detach().cpu().float().numpy().transpose(0, 2, 3, 1)


def nhwc_to_nchw(arr):
    """Convert NHWC numpy to NCHW numpy."""
    return arr.transpose(0, 3, 1, 2)


# ============================================================================
# Part 1: Weight Comparison
# ============================================================================

def compare_weights():
    separator("WEIGHT COMPARISON")

    import torch
    from diffusers import UNet2DConditionModel
    import mlx.core as mx

    print("Loading PyTorch UNet...")
    unet_pt = UNet2DConditionModel.from_pretrained(PT_UNET_DIR, torch_dtype=torch.float32)

    # The PyTorch model has in_channels=4, but the actual checkpoint was modified
    # to have in_channels=12 for concat conditioning. Check what we got.
    print(f"  PT UNet in_channels (config): {unet_pt.config.in_channels}")
    print(f"  PT conv_in weight shape: {unet_pt.conv_in.weight.shape}")
    pt_in_ch = unet_pt.conv_in.weight.shape[1]
    print(f"  PT conv_in actual in_channels: {pt_in_ch}")

    # Check attention head dims from config
    print(f"  PT attention_head_dim: {unet_pt.config.attention_head_dim}")
    print(f"  PT use_linear_projection: {unet_pt.config.use_linear_projection}")

    print("\nLoading MLX UNet weights (raw safetensors)...")
    mlx_raw = dict(mx.load(os.path.join(MLX_WEIGHTS_DIR, "paint_unet.safetensors")))

    # Strip 'unet.' prefix
    mlx_weights = {}
    for k, v in mlx_raw.items():
        if k.startswith("unet."):
            mlx_weights[k[5:]] = v
    del mlx_raw

    # Compare specific weight tensors
    pt_state = unet_pt.state_dict()

    weight_keys_to_compare = [
        "conv_in.weight",
        "conv_in.bias",
        "time_embedding.linear_1.weight",
        "time_embedding.linear_1.bias",
        "conv_out.weight",
        "conv_out.bias",
        "conv_norm_out.weight",
    ]

    # Also check first down block resnet
    weight_keys_to_compare += [
        "down_blocks.0.resnets.0.norm1.weight",
        "down_blocks.0.resnets.0.conv1.weight",
    ]

    # Check first attention block
    weight_keys_to_compare += [
        "down_blocks.0.attentions.0.norm.weight",
        "down_blocks.0.attentions.0.proj_in.weight",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
    ]

    for key in weight_keys_to_compare:
        pt_key = key
        mlx_key = key

        if pt_key not in pt_state:
            print(f"  [{key}] NOT IN PT state_dict")
            continue
        if mlx_key not in mlx_weights:
            print(f"  [{key}] NOT IN MLX weights")
            continue

        pt_w = pt_state[pt_key].detach().cpu().float().numpy()
        mlx_w = np.array(mlx_weights[mlx_key].astype(mx.float32))

        # Conv weights: PT is (O, I, kH, kW), MLX is (O, kH, kW, I)
        if pt_w.ndim == 4:
            # Transpose PT to MLX layout for comparison
            pt_w_nhwc = pt_w.transpose(0, 2, 3, 1)
            compare_tensors(f"weight:{key} (PT->OHWI)", pt_w_nhwc, mlx_w)
        elif pt_w.ndim == 2 and "norm" not in key:
            # Linear: PT is (out, in), MLX nn.Linear stores (out, in) by default
            # BUT mlx-forge conversion transposes linear weights
            # Check both orientations to find which one matches
            if pt_w.shape == mlx_w.shape:
                compare_tensors(f"weight:{key} (same shape)", pt_w, mlx_w)
            elif pt_w.shape == mlx_w.T.shape:
                compare_tensors(f"weight:{key} (PT vs MLX.T)", pt_w, mlx_w.T)
            else:
                print(f"  [weight:{key}] SHAPE MISMATCH: PT {pt_w.shape} vs MLX {mlx_w.shape}")
        else:
            compare_tensors(f"weight:{key}", pt_w, mlx_w)

    # Check for use_linear_projection mismatch
    if hasattr(unet_pt.down_blocks[0].attentions[0], 'proj_in'):
        proj_in = unet_pt.down_blocks[0].attentions[0].proj_in
        print(f"\n  PT proj_in type: {type(proj_in).__name__}")
        print(f"  PT proj_in weight shape: {proj_in.weight.shape}")

    del unet_pt, pt_state, mlx_weights
    gc.collect()


# ============================================================================
# Part 2: Attention Head Dim Analysis
# ============================================================================

def check_attention_head_dims():
    separator("ATTENTION HEAD DIM ANALYSIS")

    import torch
    from diffusers import UNet2DConditionModel

    unet_pt = UNet2DConditionModel.from_pretrained(PT_UNET_DIR, torch_dtype=torch.float32)

    print("PyTorch UNet attention config:")
    print(f"  attention_head_dim: {unet_pt.config.attention_head_dim}")
    print(f"  block_out_channels: {unet_pt.config.block_out_channels}")
    print(f"  use_linear_projection: {unet_pt.config.use_linear_projection}")

    # Check actual attention shapes in each block
    for i, block in enumerate(unet_pt.down_blocks):
        if hasattr(block, 'attentions') and len(block.attentions) > 0:
            attn = block.attentions[0]
            tb = attn.transformer_blocks[0]
            q_weight = tb.attn1.to_q.weight
            print(f"\n  down_blocks[{i}]:")
            print(f"    channels: {unet_pt.config.block_out_channels[i]}")
            print(f"    to_q weight shape: {q_weight.shape}")
            print(f"    num_heads: {tb.attn1.heads}")
            if hasattr(tb.attn1, 'head_dim'):
                print(f"    head_dim: {tb.attn1.head_dim}")

    # Mid block
    if hasattr(unet_pt.mid_block, 'attentions'):
        attn = unet_pt.mid_block.attentions[0]
        tb = attn.transformer_blocks[0]
        print(f"\n  mid_block:")
        print(f"    to_q weight shape: {tb.attn1.to_q.weight.shape}")
        print(f"    num_heads: {tb.attn1.heads}")

    print("\n\nMLX UNet attention config:")
    print("  attention_head_dim=8 (used as divisor to get num_heads)")
    for i, ch in enumerate([320, 640, 1280, 1280]):
        if i < 3:  # CrossAttn blocks
            num_heads_mlx = ch // 8
            dim_head_mlx = ch // num_heads_mlx
            print(f"  down_blocks[{i}]: ch={ch}, num_heads={num_heads_mlx}, dim_head={dim_head_mlx}")

    # Compare: what should it be vs what MLX computes
    pt_head_dims = unet_pt.config.attention_head_dim
    if isinstance(pt_head_dims, list):
        print("\n  MISMATCH ANALYSIS:")
        for i, (ch, hd) in enumerate(zip(unet_pt.config.block_out_channels, pt_head_dims)):
            pt_heads = ch // hd
            mlx_heads = ch // 8
            print(f"    Block {i}: PT has {pt_heads} heads (dim_head={hd}), "
                  f"MLX has {mlx_heads} heads (dim_head=8)")
            if pt_heads != mlx_heads:
                print(f"      *** DIFFERENT number of heads!")
                print(f"      inner_dim is same ({ch}) but attention pattern differs")

    del unet_pt
    gc.collect()


# ============================================================================
# Part 3: Timestep Embedding Detail
# ============================================================================

def compare_timestep_embedding_detail():
    separator("TIMESTEP EMBEDDING DETAIL (flip_sin_to_cos)")

    import torch
    import mlx.core as mx

    # Diffusers default for SD 2.1: flip_sin_to_cos=True, downscale_freq_shift=0
    from diffusers.models.embeddings import get_timestep_embedding as pt_get_timestep_embedding

    from hunyuanpaintpbr_mlx.unet.blocks_mlx import get_timestep_embedding as mlx_get_timestep_embedding

    t = 500.0
    dim = 320

    # PyTorch
    emb_pt = pt_get_timestep_embedding(
        torch.tensor([t]), dim, flip_sin_to_cos=True, downscale_freq_shift=0
    ).numpy()[0]

    # MLX
    emb_mlx = np.array(mlx_get_timestep_embedding(mx.array([t]), dim))[0]

    print(f"  PT  first 4 values: {emb_pt[:4]}")
    print(f"  MLX first 4 values: {emb_mlx[:4]}")
    print(f"  PT  values 160-163: {emb_pt[160:164]}")
    print(f"  MLX values 160-163: {emb_mlx[160:164]}")

    max_diff = np.abs(emb_pt - emb_mlx).max()
    mean_diff = np.abs(emb_pt - emb_mlx).mean()
    print(f"\n  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if max_diff > 0.01:
        # Try the other ordering (swap sin/cos halves)
        emb_mlx_swapped = np.concatenate([emb_mlx[160:], emb_mlx[:160]])
        max_diff_swapped = np.abs(emb_pt - emb_mlx_swapped).max()
        print(f"  Max diff (swapped halves): {max_diff_swapped:.6e}")
        if max_diff_swapped < max_diff:
            print("  *** SIN/COS ORDERING IS SWAPPED!")
            print("  PT uses flip_sin_to_cos=True -> [cos, sin]")
            print("  MLX code may have different ordering")


# ============================================================================
# Part 4: UNet Forward Pass Comparison
# ============================================================================

def compare_unet_forward():
    separator("UNET FORWARD PASS COMPARISON")

    import torch
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from diffusers import UNet2DConditionModel
    from hunyuanpaintpbr_mlx.load_model import _enhance_unet
    from hunyuanpaintpbr_mlx.unet.unet_mlx import UNet2DConditionModelMLX
    from hunyuanpaintpbr_mlx.unet.blocks_mlx import get_timestep_embedding

    # --- Create deterministic inputs ---
    rng = np.random.RandomState(SEED)

    # --- Load PyTorch UNet first to check actual in_channels ---
    print("Loading PyTorch UNet...")
    unet_pt = UNet2DConditionModel.from_pretrained(PT_UNET_DIR, torch_dtype=torch.float32)
    unet_pt.eval()

    pt_in_ch = unet_pt.conv_in.weight.shape[1]
    print(f"  PT conv_in in_channels: {pt_in_ch}")

    # Create inputs with the correct number of channels
    sample_nhwc = rng.randn(1, LATENT_H, LATENT_W, pt_in_ch).astype(np.float32) * 0.1
    text_np = rng.randn(1, 77, 1024).astype(np.float32) * 0.1
    timestep_val = 500

    # PT uses NCHW
    sample_pt = torch.tensor(nhwc_to_nchw(sample_nhwc))
    text_pt = torch.tensor(text_np)
    timestep_pt = torch.tensor([timestep_val])

    # --- Collect PT intermediate outputs ---
    pt_intermediates = {}

    def hook_conv_in(module, inp, output):
        pt_intermediates["conv_in"] = output.detach()

    def hook_down_block_0(module, inp, output):
        if isinstance(output, tuple):
            pt_intermediates["down_block_0"] = output[0].detach()
        else:
            pt_intermediates["down_block_0"] = output.detach()

    def hook_mid_block(module, inp, output):
        pt_intermediates["mid_block"] = output.detach()

    h1 = unet_pt.conv_in.register_forward_hook(hook_conv_in)
    h2 = unet_pt.down_blocks[0].register_forward_hook(hook_down_block_0)
    h3 = unet_pt.mid_block.register_forward_hook(hook_mid_block)

    print("Running PyTorch forward pass...")
    with torch.no_grad():
        out_pt = unet_pt(sample_pt, timestep_pt, encoder_hidden_states=text_pt)
        out_pt_tensor = out_pt.sample

    h1.remove()
    h2.remove()
    h3.remove()

    pt_intermediates["final"] = out_pt_tensor.detach()

    # Convert all PT intermediates to NHWC numpy
    pt_results = {}
    for k, v in pt_intermediates.items():
        pt_results[k] = pt_to_nhwc(v)

    print(f"  PT final output shape (NHWC): {pt_results['final'].shape}")
    print(f"  PT final output stats: min={pt_results['final'].min():.4f} "
          f"max={pt_results['final'].max():.4f} mean={pt_results['final'].mean():.4f}")

    # Also get PT timestep embedding for comparison
    from diffusers.models.embeddings import get_timestep_embedding as pt_get_timestep_embedding
    pt_temb = pt_get_timestep_embedding(
        torch.tensor([timestep_val]).float(), 320,
        flip_sin_to_cos=True, downscale_freq_shift=0
    )
    pt_temb_proj = unet_pt.time_embedding(pt_temb)
    pt_temb_np = pt_temb.detach().cpu().numpy()
    pt_temb_proj_np = pt_temb_proj.detach().cpu().numpy()

    # Free PT model
    del unet_pt, out_pt, out_pt_tensor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- Load MLX UNet ---
    print("\nLoading MLX UNet...")
    unet_mlx = UNet2DConditionModelMLX(
        in_channels=pt_in_ch, out_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        cross_attention_dim=1024,
        attention_head_dim=8,
    )
    _enhance_unet(unet_mlx, cross_attention_dim=1024)

    # Load weights
    unet_w = dict(mx.load(os.path.join(MLX_WEIGHTS_DIR, "paint_unet.safetensors")))
    stripped = {}
    for k, v in unet_w.items():
        if k.startswith("unet."):
            stripped[k[5:]] = v

    normalized = {}
    for k, v in stripped.items():
        nk = k.replace(".to_out_mr.0.", ".to_out_mr.")
        nk = nk.replace(".to_out_albedo.0.", ".to_out_albedo.")
        normalized[nk] = v

    model_keys = set(k for k, _ in tree_flatten(unet_mlx.parameters()))
    matched = [(k, v) for k, v in normalized.items() if k in model_keys]
    unet_mlx.load_weights(matched)
    n_loaded = len(matched)
    n_total = len(stripped)
    print(f"  Loaded {n_loaded}/{n_total} MLX weights")

    # Show some unmatched keys
    matched_keys = set(k for k, _ in matched)
    unmatched = [k for k in normalized if k not in matched_keys and not k.startswith("learned_text_clip") and not k.startswith("image_proj")]
    if unmatched:
        print(f"  First 10 unmatched keys: {unmatched[:10]}")
    del unet_w, stripped, normalized

    # Cast to float32 for comparison
    float32_params = [(k, v.astype(mx.float32)) for k, v in tree_flatten(unet_mlx.parameters())]
    unet_mlx.load_weights(float32_params)
    del float32_params

    # --- MLX forward pass with intermediates ---
    sample_mlx = mx.array(sample_nhwc)
    text_mlx = mx.array(text_np)
    timestep_mlx = mx.array([timestep_val])

    mlx_results = {}

    # Timestep embedding comparison
    t_emb = get_timestep_embedding(timestep_mlx, unet_mlx.time_proj_dim)
    t_emb_np = np.array(t_emb)
    compare_tensors("timestep_sinusoidal", pt_temb_np, t_emb_np)

    emb = unet_mlx.time_embedding(t_emb)
    emb_np = np.array(emb)
    compare_tensors("timestep_projected", pt_temb_proj_np, emb_np)

    # Step 1: conv_in
    conv_in_out = unet_mlx.conv_in(sample_mlx)
    mx_sync(conv_in_out)
    mlx_results["conv_in"] = np.array(conv_in_out)

    # Step 2: First down block
    sample_running = conv_in_out
    down_block_res_samples = [sample_running]

    block = unet_mlx.down_blocks[0]
    if hasattr(block, "has_cross_attention") and block.has_cross_attention:
        sample_running, res = block(sample_running, emb, text_mlx)
    else:
        sample_running, res = block(sample_running, emb)
    down_block_res_samples.extend(res)
    mx_sync(sample_running)
    mlx_results["down_block_0"] = np.array(sample_running)

    # Continue through remaining down blocks
    for block in unet_mlx.down_blocks[1:]:
        if hasattr(block, "has_cross_attention") and block.has_cross_attention:
            sample_running, res = block(sample_running, emb, text_mlx)
        else:
            sample_running, res = block(sample_running, emb)
        down_block_res_samples.extend(res)

    # Step 3: Mid block
    sample_running = unet_mlx.mid_block(sample_running, emb, text_mlx)
    mx_sync(sample_running)
    mlx_results["mid_block"] = np.array(sample_running)

    # Step 4: Up blocks
    for block in unet_mlx.up_blocks:
        n_res = len(block.resnets)
        res_samples = down_block_res_samples[-n_res:]
        down_block_res_samples = down_block_res_samples[:-n_res]

        if hasattr(block, "has_cross_attention") and block.has_cross_attention:
            sample_running = block(sample_running, emb, res_samples, text_mlx)
        else:
            sample_running = block(sample_running, emb, res_samples)

    # Step 5: Output
    sample_running = unet_mlx.conv_act(unet_mlx.conv_norm_out(sample_running))
    sample_running = unet_mlx.conv_out(sample_running)
    mx_sync(sample_running)
    mlx_results["final"] = np.array(sample_running)

    print(f"\n  MLX final output shape: {mlx_results['final'].shape}")
    print(f"  MLX final output stats: min={mlx_results['final'].min():.4f} "
          f"max={mlx_results['final'].max():.4f} mean={mlx_results['final'].mean():.4f}")

    # --- Compare ---
    separator("LAYER-BY-LAYER COMPARISON (UNet)")

    for layer_name in ["conv_in", "down_block_0", "mid_block", "final"]:
        pt_arr = pt_results.get(layer_name)
        mlx_arr = mlx_results.get(layer_name)
        if pt_arr is None:
            print(f"  [{layer_name}] Missing from PT")
            continue
        if mlx_arr is None:
            print(f"  [{layer_name}] Missing from MLX")
            continue
        compare_tensors(layer_name, pt_arr, mlx_arr)

    del unet_mlx
    gc.collect()


# ============================================================================
# Part 5: VAE Comparison
# ============================================================================

def compare_vae():
    separator("VAE COMPARISON")

    import torch
    from diffusers import AutoencoderKL
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from hunyuanpaintpbr_mlx.vae_mlx import AutoencoderKLMLX

    # Create deterministic test image
    rng = np.random.RandomState(SEED)
    # Small image: 64x64x3 in [-1, 1]
    img_np = rng.randn(1, 64, 64, 3).astype(np.float32) * 0.5

    # --- PyTorch VAE ---
    print("Loading PyTorch VAE...")
    vae_pt = AutoencoderKL.from_pretrained(PT_VAE_DIR, torch_dtype=torch.float32)
    vae_pt.eval()

    img_pt = torch.tensor(nhwc_to_nchw(img_np))

    print("Running PyTorch VAE encode...")
    with torch.no_grad():
        posterior = vae_pt.encode(img_pt)
        latent_pt = posterior.latent_dist.mean * vae_pt.config.scaling_factor
        latent_pt_np = pt_to_nhwc(latent_pt)

    print(f"  PT latent shape: {latent_pt_np.shape}")
    print(f"  PT latent stats: min={latent_pt_np.min():.4f} max={latent_pt_np.max():.4f}")

    # Decode
    with torch.no_grad():
        decoded_pt = vae_pt.decode(latent_pt / vae_pt.config.scaling_factor).sample
        decoded_pt_np = pt_to_nhwc(decoded_pt)

    # Spot-check VAE weights before freeing PT model
    pt_sd = vae_pt.state_dict()

    del vae_pt, img_pt, latent_pt, decoded_pt
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- MLX VAE ---
    print("\nLoading MLX VAE...")
    vae_mlx = AutoencoderKLMLX()

    vae_w = dict(mx.load(os.path.join(MLX_WEIGHTS_DIR, "paint_vae.safetensors")))
    vae_mlx.load_weights(list(vae_w.items()))

    # Cast to float32
    float32_params = [(k, v.astype(mx.float32)) for k, v in tree_flatten(vae_mlx.parameters())]
    vae_mlx.load_weights(float32_params)
    del float32_params

    img_mlx = mx.array(img_np)

    print("Running MLX VAE encode...")
    latent_mlx = vae_mlx.encode(img_mlx)
    mx_sync(latent_mlx)
    latent_mlx_np = np.array(latent_mlx)

    print(f"  MLX latent shape: {latent_mlx_np.shape}")
    print(f"  MLX latent stats: min={latent_mlx_np.min():.4f} max={latent_mlx_np.max():.4f}")

    # Decode
    decoded_mlx = vae_mlx.decode(latent_mlx)
    mx_sync(decoded_mlx)
    decoded_mlx_np = np.array(decoded_mlx)

    # --- Compare ---
    separator("VAE ENCODE/DECODE COMPARISON")
    compare_tensors("vae_encode_latent", latent_pt_np, latent_mlx_np)
    compare_tensors("vae_decode_image", decoded_pt_np, decoded_mlx_np)

    # Also compare VAE weights directly
    print("\n  VAE weight spot-check:")

    vae_weight_keys = [
        "encoder.conv_in.weight",
        "encoder.down_blocks.0.resnets.0.conv1.weight",
        "decoder.conv_in.weight",
    ]

    for key in vae_weight_keys:
        if key in pt_sd and key in vae_w:
            pt_w = pt_sd[key].detach().cpu().float().numpy()
            mlx_w = np.array(vae_w[key].astype(mx.float32))
            if pt_w.ndim == 4:
                pt_w = pt_w.transpose(0, 2, 3, 1)
            compare_tensors(f"vae_weight:{key}", pt_w, mlx_w)

    del pt_sd, vae_w, vae_mlx
    gc.collect()


# ============================================================================
# Utility
# ============================================================================

def mx_sync(x):
    """Force MLX to materialize a lazy array."""
    import mlx.core as mx
    mx.eval(x)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  MLX vs PyTorch UNet/VAE Numerical Comparison")
    print("  Latent size: {}x{} (tiny for debugging)".format(LATENT_H, LATENT_W))
    print("=" * 70)

    # Run all comparisons
    sections = [
        ("Weight Comparison", compare_weights),
        ("Attention Head Dim Analysis", check_attention_head_dims),
        ("Timestep Embedding", compare_timestep_embedding_detail),
        ("UNet Forward Pass", compare_unet_forward),
        ("VAE Comparison", compare_vae),
    ]

    for name, func in sections:
        try:
            func()
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    separator("SUMMARY")
    print("  Check the output above for [FAIL] markers.")
    print("  Key things that could cause blue/cyan blob textures:")
    print("  1. attention_head_dim mismatch (PT uses per-block [5,10,20,20], MLX uses fixed 8)")
    print("  2. sin/cos ordering in timestep embedding")
    print("  3. Conv weight transposition errors")
    print("  4. Linear weight transposition (PT vs MLX convention)")
    print("  5. VAE encode/decode mismatch")
    print("  6. GroupNorm epsilon differences")
    print()
