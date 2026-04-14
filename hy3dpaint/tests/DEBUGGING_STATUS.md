# MLX vs PyTorch Numerical Validation - Debug Status

## Current state

| Component | Diff (max_abs) | Status |
|---|---|---|
| VAE encode | bit-identical | ✅ MATCH |
| Conv layers | 0 | ✅ MATCH |
| GroupNorm | 7e-7 | ✅ MATCH (with pytorch_compatible=True) |
| Timestep embedding | 3e-5 | ✅ MATCH |
| ResNet block (in-UNet) | 1.1e-5 | ✅ MATCH |
| BasicTransformerBlock (standalone, same weights) | 0 | ✅ MATCH |
| Transformer2DModel (standalone, same weights) | 1e-5 | ✅ MATCH |
| Full UNet forward (PT vs MLX, same input) | **0.20** | ⚠ MISMATCH |

## Key insight

Each individual building block matches PyTorch perfectly when tested in isolation.
The Transformer2DModel standalone test produces **identical** output to PyTorch
diffusers `Transformer2DModel` (max_diff 1e-5).

But the full UNet forward pass shows ~0.20 max_abs error (~3% norm).

This means the error accumulates through the chain of layers (~24 transformer blocks
+ resnet blocks). Each layer's output is slightly off, and the next layer takes
that slightly-off input and produces a slightly-more-off output.

## Why the cyan/neutral output?

3% error per UNet pass × 15 denoising steps × CFG amplification → cumulative
divergence pushes the final latent toward a "neutral" value. The VAE decodes
this neutral latent to cyan/white.

## Likely causes (not yet identified)

1. **Subtle softmax precision**: fp16 weights cast to fp32, but intermediate
   activations might lose precision differently
2. **MLX vs PyTorch attention scaling**: minor float differences in how
   `softmax(QK^T / sqrt(d))` is computed
3. **FFN GEGLU implementation**: our GEGLU might compute `silu(gate) * value`
   while PyTorch does `gate * gelu(value)` (or similar order)
4. **Weight precision loss**: fp16 → fp32 cast doesn't recover lost precision

## Reproducing the issue

```bash
cd /Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint
python3 tests/compare_mlx_pytorch.py
```

Look for `[final]` line — should show `max_abs ~0.21`.

## Next steps for full quality

1. **Reduce per-layer error**: profile the FIRST CrossAttnDownBlock2D end-to-end
   PT vs MLX to see if 3-resnet+2-attention chain accumulates beyond what
   individual layer testing shows.

2. **Compare attention internals**: instrument attention to capture Q, K, V,
   QK^T, softmax, attention output. Compare each step PT vs MLX.

3. **Check FeedForward GEGLU**: Verify the order of operations matches diffusers.

4. **Try fp16 throughout**: PyTorch SD2.1 inference is typically fp16. Forcing
   fp32 might amplify some numerical paths.
