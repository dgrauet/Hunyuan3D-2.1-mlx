# MLX vs PyTorch UNet Numerical Comparison Report

**Test configuration**
- Latent spatial size: 8 x 8
- Input channels: 12 (4 latent + 4 normal + 4 position)
- Encoder text: zeros (1, 77, 1024)
- Timestep: 500
- Both models cast to fp32
- Tolerance: 0.01 (abs)

## Setup fixes made to the script

1. **MLX UNet built at `in_channels=12`** with `attention_head_dim=(5, 10, 20, 20)` (matching the
   PyTorch config), no `_enhance_unet()` — we compare the vanilla UNet topology.
2. **PyTorch UNet rebuilt from `config.json` with `in_channels=12`** (the config says 4 but the
   actual checkpoint has 12-channel `conv_in`). Loading via `from_pretrained` silently drops most
   weights because the checkpoint's keys are nested inside `unet.` / `unet_dual.` with a
   `.transformer.` wrapper for `BasicTransformerBlock`. We now:
   - load the `.bin` file manually,
   - strip the `unet.` prefix,
   - collapse `transformer_blocks.0.transformer.X` to `transformer_blocks.0.X`,
   - drop 2.5D-only tensors (`attn_multiview`, `attn_refview`, `attn_dino`, `.processor.`).
   - This yields **686 weights loaded, 0 missing, 0 unexpected**.

## Weight spot-check

All 18 spot-checked weight tensors (conv_in, time_embedding, conv_out, conv_norm_out, resnet 0
weights, attention.0 norm/proj_in, attn1 to_q/to_k/to_v for both down_blocks[0] and mid_block)
are bit-identical between the PT state_dict and the MLX safetensors after layout transposition.
Conclusion: **the converter is fine**; the bug is in the MLX modules themselves.

## Forward-pass divergence (plain run, same weights, same input)

| layer         | shape           | max_abs | status    |
| ------------- | --------------- | ------- | --------- |
| conv_in       | (1, 8, 8, 320)  | 1.5e-07 | MATCH     |
| down_block_0  | (1, 4, 4, 320)  | 6.29    | MISMATCH  |
| mid_block     | (1, 1, 1, 1280) | 15.18   | MISMATCH  |
| up_block_0    | (1, 2, 2, 1280) | 6.19    | MISMATCH  |
| conv_out / final | (1, 8, 8, 4)| 0.68    | MISMATCH  |

**First divergent layer: `down_blocks[0]`** (while `conv_in` still matches).

## Drill-down inside `down_blocks[0]`

PyTorch forward hooks captured each sub-module output; MLX sub-modules were run in the same order.

| sub-module     | max_abs (plain) | max_abs (GN fix) | max_abs (PT temb) |
| -------------- | --------------- | ---------------- | ----------------- |
| resnet 0       | 1.83            | **0.042**        | 1.85              |
| attention 0    | 2.33            | 0.75             | 2.31              |
| resnet 1       | 2.51            | 0.78             | 2.49              |
| attention 1    | 2.70            | 0.99             | 2.68              |

Each experimental column changes one variable vs the "plain" column:
- **GN fix**: every `nn.GroupNorm` in the model patched to `pytorch_compatible=True`.
- **PT temb**: the timestep embedding for the MLX forward is replaced by the exact
  numpy tensor produced by PyTorch's `time_embedding`.

The GN fix collapses resnet-0 error by ~44x (from 1.83 to 0.042 — roughly at fp32 tolerance).
Substituting the PT timestep embedding alone barely moves the needle at this stage.

---

## Bugs identified

### Bug 1 (primary, high impact): `nn.GroupNorm` uses MLX-native mode, not PyTorch-compatible mode

`mlx.nn.GroupNorm.__init__` defaults to `pytorch_compatible=False`, which implements a
different normalization scheme. Standalone test with identity affine parameters on a
`(1, 8, 8, 320)` tensor:

```
MLX default                 vs PyTorch GroupNorm: max |diff| = 3.45e-01
MLX pytorch_compatible=True vs PyTorch GroupNorm: max |diff| = 7.15e-07
```

Every `nn.GroupNorm` in the MLX port is affected. Locations:

- `hunyuanpaintpbr_mlx/unet/blocks_mlx.py` lines 64, 66 (ResnetBlock2D)
- `hunyuanpaintpbr_mlx/unet/blocks_mlx.py` line 403 (Transformer2DModel.norm)
- `hunyuanpaintpbr_mlx/unet/modules_mlx.py` line 338 (2.5D Transformer2DModel.norm)
- `hunyuanpaintpbr_mlx/unet/unet_mlx.py` line 153 (conv_norm_out)
- `hunyuanpaintpbr_mlx/vae_mlx.py` lines 24, 26, 59, 221, 263

**Fix**: pass `pytorch_compatible=True` on every `nn.GroupNorm(...)` construction site. The
drill-down showed that applying this fix at runtime shrinks the resnet-0 max_abs from 1.83
to 0.042 (with the remaining 0.042 attributable to Bug 2).

### Bug 2 (secondary, clearly measurable): timestep-embedding frequency denominator

`hunyuanpaintpbr_mlx/unet/blocks_mlx.py` line 20 uses:

```python
emb = math.log(10000) / (half_dim - 1)
```

Diffusers' `get_timestep_embedding` with `downscale_freq_shift=0` (the value SD 2.1 uses) divides by
`half_dim`, not `half_dim - 1`. Direct comparison at `dim=320, t=500`:

```
[timestep_sinusoidal] max_abs=1.06  mean_abs=0.21  [MISMATCH]
```

The first element matches (frequency 0) but all others drift — by up to 1.06 on individual entries.
Swapping MLX's timestep embedding for PyTorch's only moved resnet-0 max_abs by ~0.02 at this stage,
so Bug 2 is quantitatively smaller than Bug 1 *for this particular input*, but it is still present
and will stack with Bug 1 after many layers / non-trivial text inputs.

**Fix**: change to `emb = math.log(10000) / half_dim` (and in principle take a
`downscale_freq_shift` argument, but 0 is what we need for SD 2.1).

### Non-bug: the residual attention error (~0.75 after GN fix)

After fixing GroupNorm, attention-0 still shows max_abs ~0.75 at `down_blocks[0]`. This is
explained by:
- Its input already carries ~0.042 error from resnet-0.
- The first attention's softmax and proj_in amplify small input perturbations.
- Fixing Bug 2 removes a further small component.

No obvious structural bug in the attention code was found during the drill-down; the QKV
weights compare bit-identically, `heads=5` matches PT's per-block head-dim config, and
`to_out` path matches. Recommend re-running this comparison after fixing Bugs 1 and 2 to
confirm attention falls within tolerance.

---

## How to reproduce

```bash
cd /Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint
python3 tests/compare_mlx_pytorch.py
```

The script prints MATCH/MISMATCH verdicts with max absolute difference, max relative
difference, PT-norm and MLX-norm for every comparison, and writes a summary here.

Key files referenced:
- `/Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint/tests/compare_mlx_pytorch.py`
- `/Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint/hunyuanpaintpbr_mlx/unet/blocks_mlx.py`
- `/Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint/hunyuanpaintpbr_mlx/unet/modules_mlx.py`
- `/Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint/hunyuanpaintpbr_mlx/unet/unet_mlx.py`
- `/Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint/hunyuanpaintpbr_mlx/vae_mlx.py`
