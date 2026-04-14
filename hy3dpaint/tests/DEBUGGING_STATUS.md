# MLX vs PyTorch Numerical Validation - RESOLVED

## Current state

| Component | Diff (max_abs) | Status |
|---|---|---|
| VAE encode | bit-identical | ✅ MATCH |
| Conv layers | 0 | ✅ MATCH |
| GroupNorm | 7e-7 | ✅ MATCH (with pytorch_compatible=True) |
| Timestep embedding | 3e-5 | ✅ MATCH |
| ResNet block | 1e-5 | ✅ MATCH |
| Self-attention (attn0) | 6.8e-5 | ✅ MATCH |
| down_block_0 full | 1.6e-4 | ✅ MATCH |
| mid_block | 3.6e-4 | ✅ MATCH |
| up_block_0 | 1.7e-4 | ✅ MATCH |
| **Full UNet final** | **1.17e-5** | **✅ MATCH** |

## Root cause (identified and fixed)

**Diffusers `attention_head_dim` config is a misnomer — it specifies `num_heads`, not per-head dim.**

For SD 2.1 paint UNet, `attention_head_dim=[5, 10, 20, 20]` with
`block_out_channels=[320, 640, 1280, 1280]` means:
- 5 heads × 64 dim (block 0)
- 10 heads × 64 dim (block 1)
- 20 heads × 64 dim (blocks 2–3)

The MLX port originally interpreted the values as per-head dim and computed
`num_heads = channels // head_dim`, giving 64 heads × 5 dim per block —
the *opposite* of what diffusers does.

Because `inner_dim = num_heads × head_dim = channels` either way, weight
tensors load at the correct shape. The bug only manifests at runtime when
Q/K/V are reshaped for multi-head attention: `softmax(QK^T/√d)` operates
on a completely different grouping, producing numerically different outputs.

### Symptom chain
1. Single attention layer diverges by ~0.5 max_abs
2. ResNet after attn inherits the error, amplifies it (0.56)
3. Full UNet: 0.21 max_abs, ~3% relative norm error per pass
4. 15 denoising steps × CFG cumulative divergence → latents drift toward a
   neutral value
5. VAE decodes neutral latent to cyan/white blobs instead of textures

## Fix

Three one-line changes in `hunyuanpaintpbr_mlx/unet/unet_mlx.py`:

- Down blocks: `num_heads = self._attention_head_dims[i]`
  (was `ch // self._attention_head_dims[i]`)
- Mid block: `num_mid_heads = self._attention_head_dims[-1]`
  (was `mid_channels // self._attention_head_dims[-1]`)
- Up blocks: `num_heads = self._attention_head_dims[rev_idx]`
  (was `ch // self._attention_head_dims[rev_idx]`)

## Validation

```bash
cd /Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint
python3 tests/compare_mlx_pytorch.py
```

Final line should show `[final] max_abs=1.17e-05 [MATCH]`. All six layers
(conv_in, down_block_0, mid_block, up_block_0, conv_out, final) match.

## Next steps

1. Run end-to-end paint inference with the real mermaid mesh + astronaut
   texture and confirm textures are now coherent (not cyan/white).
2. If textures appear but quality lags, investigate remaining 2.5D paths
   (MDA, reference, multiview, DINO) — these were disabled in the above
   plain-UNet validation but re-engaged during actual inference.
