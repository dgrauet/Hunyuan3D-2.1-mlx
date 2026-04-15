# Hunyuan3D-2.1 MLX — Forward Pass (Stage 2: Texture Synthesis)

End-to-end data flow from a reference image + 3D mesh to a textured GLB.
All tensors are in MLX (NHWC), all weights live on Apple Silicon's
unified memory.

```
                  REFERENCE IMAGE              MESH (.glb)
                   (PIL, 512×512)              (Trimesh, native UVs)
                          │                           │
                          │                           ▼
                          │                ┌─────────────────────┐
                          │                │  MeshRenderMLX      │
                          │                │  (Metal rasterizer) │
                          │                └─────────┬───────────┘
                          │                          │
                          │                  6 azim × 1 elev (+ top/bot)
                          │                          │
                          │           ┌──────────────┼─────────────────┐
                          │           ▼              ▼                 ▼
                          │     normal_maps    position_maps    alpha (face IDs)
                          │     (6, H, W, 3)   (6, H, W, 3)     used for view sel.
                          │           │              │
                          │           ▼              ▼
                          │     ┌───────────────────────────────┐
                          │     │  VAE.encode (×3, frozen)      │
                          │     │  → z_normal, z_pos (6, h, w, 4)│
                          │     │  + raw position_maps (no VAE) │
                          │     └──────────────┬────────────────┘
                          │                    │
                          │                    ▼
                          │     ┌─────────────────────────────────┐
                          │     │ calc_multires_voxel_idxs        │
                          │     │ → {seq_len: voxel_indices,      │
                          │     │              voxel_resolution}  │
                          │     │ for each level [4096,1024,256,64]│
                          │     └──────────────┬──────────────────┘
                          ▼                    │
                ┌─────────────────────┐        │
                │ DINOv2 (frozen)     │        │
                │ + ImageProjModel    │        │
                │ → dino_proj         │        │
                │   (1, N_tok, 1024)  │        │
                └──────────┬──────────┘        │
                           │                   │
                           │   ┌───────────────┘
                           │   │
                ┌──────────▼───▼──────────────────────────────┐
                │ extract_reference_features (run ONCE)        │
                │ → unet_dual (4-ch, vanilla SD2.1)            │
                │   forward(ref_latent, t=0, text_ref)          │
                │   capture_dict[block_id] = norm_pre_attn1   │
                │ → ref_features dict (16 transformer blocks)  │
                └──────────────┬───────────────────────────────┘
                               │
                       ┌───────┴───────┐
                       │  LEARNED      │
                       │  TEXT TOKENS  │
                       │  (77, 1024)   │
                       │  per material │
                       │  + ref + neg  │
                       └───────┬───────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│           DENOISING LOOP (15 steps, UniPC, v_prediction)          │
│                                                                   │
│   latents ~ N(0,1) shape (n_pbr*n_views, h, w, 4) = (12, h, w, 4)│
│                                                                   │
│   for t in scheduler.timesteps:                                   │
│     for chunk_view in chunks(n_views=6):                          │
│                                                                   │
│       unet_in = concat(latents, z_normal, z_pos) → (12, h, w, 12)│
│                                                                   │
│       ┌─────────── 3 CFG passes (PT-faithful) ───────────┐        │
│       │  pred_uncond = UNet(unet_in, text=NEG)            │        │
│       │       no DINO, no ref_features                    │        │
│       │  pred_ref    = UNet(unet_in, text=POS,            │        │
│       │       ref_features=ref_features)                  │        │
│       │       no DINO yet                                 │        │
│       │  pred_full   = UNet(unet_in, text=POS,            │        │
│       │       ref_features=ref_features,                  │        │
│       │       dino_features=dino_proj,                    │        │
│       │       position_voxel_indices=...)                  │        │
│       └───────────────────────────────────────────────────┘        │
│                                                                   │
│       view_scale = cam_mapping(azim) per view ∈ [1, 2, ..., 5]    │
│       guided = pred_uncond                                         │
│                + g · vs · (pred_ref  - pred_uncond)               │
│                + g · vs · (pred_full - pred_ref)                  │
│                                                                   │
│     latents = scheduler.step(noise_guided, t, latents)            │
└───────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                ┌────────────────────────────┐
                │ VAE.decode (per view)       │
                │ → albedo (6, 512, 512, 3)   │
                │   mr     (6, 512, 512, 3)   │
                └─────────────┬───────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │ RealESRGAN x4 (MLX)         │
                │ → 512² → 2048² per view     │
                └─────────────┬───────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────┐
                │  back_project (per view)               │
                │  for each view: project_texture,        │
                │     project_cos_map, boundary           │
                │  cos = view_weight * (cos**bake_exp)    │
                └─────────────┬──────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────┐
                │  fast_bake_texture (mode="face_wta")    │
                │  per UV face → pick view with max sum   │
                │  of cos over face's texels              │
                │  → atlas (2048, 2048, 3) + trust mask   │
                └─────────────┬──────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────┐
                │  uv_inpaint (mesh-aware)                │
                │  1. mesh_vertex_inpaint (Python port    │
                │     of meshVerticeInpaint):              │
                │       - seed vtx_color from texels      │
                │       - propagate via 3D adjacency      │
                │       - face barycentric raster +4 px   │
                │         conservative margin             │
                │  2. EDT nearest-fill for residual gutter│
                └─────────────┬──────────────────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │  set_texture(atlas)         │
                │  set_texture_mr(atlas_mr)   │
                │  save_mesh(.obj, downsample=False) │
                │  + trimesh export → .glb     │
                └────────────────────────────┘
```

## UNet (per CFG pass) — internal data flow

```
sample (B*n_pbr*n_views, h, w, 12)
   │
   ▼
conv_in → (B*n_pbr*n_views, h, w, 320)
   │
   ├─────────────────────── DownBlocks (3 cross-attn + 1 plain) ──────┐
   │                                                                  │
   │  CrossAttnDownBlock2D × 3:                                        │
   │    for each ResNet+Transformer pair:                              │
   │      hidden = ResNet(hidden, temb)                                │
   │      hidden = Transformer2DModel(hidden, text, **kwargs)          │
   │        norm + proj_in                                              │
   │        for each transformer_block:                                │
   │          ┌──────── BasicTransformerBlock ────────┐               │
   │          │  norm_hs = norm1(hidden)               │               │
   │          │                                        │               │
   │          │  ─ Step 1: SELF-ATTN (or MDA) ─        │               │
   │          │    if MDA enabled (n_views > 0):       │               │
   │          │      split albedo/MR by index          │               │
   │          │      albedo: attn1(norm_hs[albedo])    │               │
   │          │      mr: attn1.processor.{q,k,v,out}_mr│               │
   │          │    else: attn_out = attn1(norm_hs)     │               │
   │          │    hidden += attn_out                  │               │
   │          │                                        │               │
   │          │  ─ Step 2: REFERENCE-ATTN ─            │               │
   │          │    if ref_features and _block_id:      │               │
   │          │      query_albedo = norm_hs[:,0,...]   │               │
   │          │      ref_ctx = ref_features[block_id]  │               │
   │          │      Q,K shared, V per material:        │               │
   │          │        albedo: attn_refview.to_v        │               │
   │          │        mr: processor.to_v_mr            │               │
   │          │      sdpa(Q, K, V_per_mat)             │               │
   │          │      hidden += per-material output     │               │
   │          │                                        │               │
   │          │  ─ Step 3: MULTIVIEW-ATTN (RoPE 3D) ─  │               │
   │          │    if n_views > 1:                     │               │
   │          │      mv_input = norm_hs.reshape(B*n_pbr, n_views*L, C) │
   │          │      Q,K = rotate(Q,K, voxel_indices, voxel_res)       │
   │          │      mv_out = sdpa(Q, K, V) (flash-style)              │
   │          │      hidden += mv_out                  │               │
   │          │                                        │               │
   │          │  ─ Step 4: CROSS-ATTN (text) ─         │               │
   │          │    hidden += attn2(norm2(hidden), text)│               │
   │          │                                        │               │
   │          │  ─ Step 5: DINO CROSS-ATTN ─            │               │
   │          │    if dino_features:                    │               │
   │          │      hidden += attn_dino(norm_dino(h),  │               │
   │          │                          dino_features) │               │
   │          │                                        │               │
   │          │  ─ Step 6: FFN ─                        │               │
   │          │    hidden += ff(norm3(hidden))         │               │
   │          └────────────────────────────────────────┘               │
   │        proj_out + residual                                        │
   │      output_states.append(hidden)                                 │
   │    Downsample2D                                                   │
   │                                                                   │
   │  DownBlock2D × 1 (no attention):                                  │
   │    ResNet × 2                                                     │
   │                                                                   │
   ▼                                                                   │
mid_block (UNetMidBlock2DCrossAttn)                                    │
   ResNet → Transformer (same 6-step block) → ResNet                   │
   │                                                                   │
   ▼                                                                   │
   ├─────────────────── UpBlocks (1 plain + 3 cross-attn) ◀────────────┘
   │  UpBlock2D × 1 (concat skip + ResNet × 3 + Upsample)
   │  CrossAttnUpBlock2D × 3 (concat skip + ResNet+Transformer × 3 + Upsample)
   │
   ▼
conv_norm_out + SiLU + conv_out → (B*n_pbr*n_views, h, w, 4) noise prediction
```

## Critical invariants (learned the hard way)

Two classes of subtle bugs produced visibly degraded output for us; both
follow the same pattern — **a norm tensor is computed once and every
attention path in that step must reuse it, never recompute against the
updated hidden_states**:

1. **norm1 invariant**: in each transformer block, `norm1(hidden_states)`
   is computed once. Self-attn (or MDA), reference-attn, and multiview-
   attn all consume that single `norm_hs`. The block's hidden_states
   accumulates additive updates from each attention but `norm_hs` itself
   is NEVER recomputed inside the block.

2. **norm2 invariant**: same for `norm2`. Computed once. Text cross-attn
   AND DINO cross-attn both consume the pre-attn2 `norm2_hs`.
   Recomputing after attn2 feeds DINO a signal off by attn2's residual,
   which weakens the reference/style conditioning downstream.

Other PT-parity traps to watch for when porting:

- **VAE encode/decode scale**: `vae.encode()` multiplies by
  `scaling_factor` internally; `vae.decode()` divides by it. Callers
  must NOT scale again. (We doubled it once and conditioning latents
  ended up 30x too small.)
- **`position_voxel_indices` lookup key**: indexed by the *multiview*
  sequence length `n_views * L`, not per-view `L` — matches
  `multivew_hidden_states.shape[1]` in PT's modules.py.
- **Dual-stream reference UNet**: the checkpoint ships BOTH a main
  `unet.*` (2.5D) AND a vanilla `unet_dual.*` (plain SD 2.1) for
  reference feature extraction. Must load both.
- **3-pass CFG DINO/ref routing**: uncond has no DINO/no ref; ref has
  ref only; full has both. Passing DINO to both ref AND full collapses
  the `full - ref` guidance term.
- **DINOv2**: run at native 518x518 with fp32 weights — pos_embed
  interpolation to 224 amplifies ~450x through 40 blocks.
- **Per-material ref-attn V/out**: `attn_refview.processor.to_v_mr` /
  `to_out_mr` exist in the checkpoint and are required for non-albedo
  materials to get reference conditioning.
- **Diffusers `attention_head_dim`**: is a misnomer — it's the number
  of heads, not per-head dim. `head_dim = channels // attention_head_dim`.

## Tensor shapes for default config (resolution=512, 6 views, 2 PBR materials)

| Stage | Tensor | Shape | Notes |
|-------|--------|-------|-------|
| Conditioning | normal_maps (PIL) | 6 × (512, 512, 3) | RGB float [0,1] |
| Conditioning | position_maps (PIL) | 6 × (512, 512, 3) | RGB float [0,1] |
| Conditioning | reference (PIL) | (512, 512, 3) | white background |
| VAE encoded | z_normal | (6, 64, 64, 4) | scaled by 0.18215 |
| VAE encoded | z_pos | (6, 64, 64, 4) | scaled by 0.18215 |
| Voxel idx | level 0 | (1, 6×4096, 3), vox_res=512 | for L=4096 multiview attn |
| Voxel idx | level 1 | (1, 6×1024, 3), vox_res=256 | for L=1024 |
| Voxel idx | level 2 | (1, 6×256, 3), vox_res=128 | for L=256 |
| Voxel idx | level 3 | (1, 6×64, 3), vox_res=64 | for L=64 |
| DINOv2 | dino_proj | (1, 4 tokens, 1024) | from ImageProjModel |
| Ref features | per block | dict[16 blocks] → (1, 4096, C) at L0 | from unet_dual |
| Latent init | latents | (12, 64, 64, 4) | n_pbr × n_views |
| UNet input | unet_in | (12, 64, 64, 12) | latent + z_normal + z_pos |
| UNet output | noise_pred | (12, 64, 64, 4) | per-view, per-material |
| Decoded | albedo[i] | (512, 512, 3) | uint8 RGB |
| Super-res | albedo[i] | (2048, 2048, 3) | RealESRGAN x4 |
| Bake atlas | albedo atlas | (2048, 2048, 3) | float [0,1] |

## Key references in the codebase

| Component | File |
|-----------|------|
| Pipeline orchestration | `hy3dpaint/textureGenPipeline_mlx.py` |
| Inference loop, CFG, voxel calc | `hy3dpaint/hunyuanpaintpbr_mlx/inference.py` |
| Model loading + ref-feature extraction | `hy3dpaint/hunyuanpaintpbr_mlx/load_model.py` |
| UNet construction | `hy3dpaint/hunyuanpaintpbr_mlx/unet/unet_mlx.py` |
| BasicTransformerBlock + 6-step forward | `hy3dpaint/hunyuanpaintpbr_mlx/unet/blocks_mlx.py` |
| Per-material V/out, RoPE, MDA processors | `hy3dpaint/hunyuanpaintpbr_mlx/unet/attn_processor_mlx.py` |
| 3D voxel index computation | `hy3dpaint/hunyuanpaintpbr_mlx/unet/voxel_indices.py` |
| VAE | `hy3dpaint/hunyuanpaintpbr_mlx/vae_mlx.py` |
| DINOv2 + ImageProjModel | `hy3dpaint/hunyuanpaintpbr_mlx/dino_mlx.py` |
| UniPC scheduler (v_prediction) | `hy3dpaint/hunyuanpaintpbr_mlx/scheduler_mlx.py` |
| Metal rasterizer + bake | `hy3dpaint/DifferentiableRenderer/mesh_render_mlx.py` |
| Mesh-aware UV inpaint | `hy3dpaint/DifferentiableRenderer/mesh_inpaint_py.py` |
| RealESRGAN super-res | `hy3dpaint/utils/image_super_utils_mlx.py` |
