"""Full inference for HunyuanPaintPBR with CFG + multi-view.

Reproduces the PyTorch pipeline's denoising loop:
- 6-view generation (albedo + metallic-roughness)
- Classifier-free guidance (triple-batch: uncond / ref / full)
- Learned material text embeddings
- DINO feature conditioning
- View-dependent guidance scaling
"""

from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np
from PIL import Image

from .dino_mlx import preprocess_for_dino
from .load_model import HunyuanPaintModelMLX, extract_reference_features


def _cam_mapping(azim: float) -> float:
    """View-dependent guidance scale (per the original pipeline)."""
    azim = azim % 360
    if 0 <= azim < 90:
        return azim / 90.0 + 1.0
    elif 90 <= azim < 330:
        return 2.0
    else:
        return -azim / 90.0 + 5.0


def generate_multiview_pbr(
    model: HunyuanPaintModelMLX,
    normal_maps: List[np.ndarray],
    position_maps: List[np.ndarray],
    reference_image: np.ndarray,
    camera_azims: Optional[List[float]] = None,
    num_inference_steps: int = 15,
    guidance_scale: float = 3.0,
    view_size: int = 512,
    seed: int = 0,
) -> Dict[str, List[np.ndarray]]:
    """Generate multiview PBR textures.

    Args:
        model: Loaded HunyuanPaintModelMLX.
        normal_maps: List of N (H, W, 3) float32 normal renders in [0, 1].
        position_maps: List of N (H, W, 3) float32 position renders in [0, 1].
        reference_image: (H, W, 3) uint8 or float32 reference image.
        camera_azims: Azimuth angles for each view (for view-dependent guidance).
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale.
        view_size: Generation resolution.
        seed: Random seed.

    Returns:
        dict with 'albedo' and 'mr' keys, each a list of N (H, W, 3) uint8 arrays.
    """
    mx.random.seed(seed)

    n_views = len(normal_maps)
    n_pbr = 2  # albedo + mr
    latent_size = view_size // 8
    scaling_factor = 0.18215

    if camera_azims is None:
        camera_azims = [0.0] * n_views

    # ------------------------------------------------------------------
    # 1. Encode conditions
    # ------------------------------------------------------------------
    print("  Encoding conditions...")

    def encode_images(images):
        """Encode list of images to latent space."""
        arrays = []
        for img in images:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            from PIL import Image as PILImage
            pil = PILImage.fromarray((img * 255).astype(np.uint8)).resize(
                (view_size, view_size)
            )
            arr = np.array(pil).astype(np.float32) / 255.0
            arrays.append(arr)
        batch = mx.array(np.stack(arrays))  # (N, H, W, 3)
        # vae.encode already multiplies by self.scaling_factor internally;
        # do NOT re-multiply (was double-scaling latents to ~0.033 of the
        # expected magnitude, making ref_latents and z_normal/z_pos
        # essentially invisible to the UNet → washed-out diffusion output).
        return model.vae.encode(batch * 2 - 1)

    z_normal = encode_images(normal_maps)   # (N, h, w, 4)
    z_pos = encode_images(position_maps)    # (N, h, w, 4)
    mx.synchronize()

    # Multi-resolution voxel indices for 3D RoPE in multiview attention.
    # Built from the raw position maps (NOT the VAE-encoded latents).
    from .unet.voxel_indices import calc_multires_voxel_idxs
    pos_resized = []
    for pm in position_maps:
        if pm.dtype == np.uint8:
            pm = pm.astype(np.float32) / 255.0
        from PIL import Image as PILImage
        pil = PILImage.fromarray((pm * 255).astype(np.uint8)).resize(
            (view_size, view_size)
        )
        pos_resized.append(np.array(pil).astype(np.float32) / 255.0)
    # (N, H, W, 3) -> (1, N, 3, H, W)
    pos_np = np.stack(pos_resized).transpose(0, 3, 1, 2)[None]
    voxel_indices = calc_multires_voxel_idxs(pos_np)

    # ------------------------------------------------------------------
    # 2. DINO features
    # ------------------------------------------------------------------
    print("  Extracting DINO features...")
    if reference_image.dtype != np.uint8:
        ref_u8 = (reference_image * 255).astype(np.uint8)
    else:
        ref_u8 = reference_image
    dino_in = preprocess_for_dino(ref_u8)
    dino_feat = model.dino(dino_in)
    dino_proj = model.image_proj(dino_feat)  # (1, N_tokens, 1024)
    mx.synchronize()

    # ------------------------------------------------------------------
    # 3. Text embeddings for CFG
    # ------------------------------------------------------------------
    # Per-material learned tokens
    text_albedo = model.learned_text_clip["albedo"]  # (77, 1024)
    text_mr = model.learned_text_clip["mr"]
    text_neg = mx.zeros_like(text_albedo)

    # ------------------------------------------------------------------
    # 2b. Reference features — capture norm_hidden_states from each
    #     transformer block by running a single forward pass on the
    #     reference image latent.  Matches PyTorch mode "w".
    # ------------------------------------------------------------------
    print("  Extracting reference features...")
    ref_latent = encode_images([reference_image])  # (1, h, w, 4)
    text_ref = model.learned_text_clip["ref"]  # PT uses learned_text_clip_ref
    ref_text_batch = text_ref[None]  # (1, 77, 1024)
    # Use the dual-stream reference UNet when available — its weights are
    # different from the main UNet and are specifically trained to extract
    # ref-attention features. Falling back to the main UNet (as we did) gives
    # noticeably worse reference grounding.
    ref_unet = getattr(model, "unet_dual", None) or model.unet
    ref_features = extract_reference_features(
        ref_unet, ref_latent, ref_text_batch
    )
    mx.synchronize()

    # Build per-sample text: (N_pbr * N_views, 77, 1024)
    # Order: [albedo_v0, albedo_v1, ..., mr_v0, mr_v1, ...]
    text_full = mx.concatenate([
        mx.broadcast_to(text_albedo[None], (n_views, 77, 1024)),
        mx.broadcast_to(text_mr[None], (n_views, 77, 1024)),
    ], axis=0)  # (N_pbr*N_views, 77, 1024) = (12, 77, 1024)

    text_neg_batch = mx.broadcast_to(text_neg[None], (n_pbr * n_views, 77, 1024))

    # CFG triple: [negative, ref(=full), full]
    text_cfg = mx.concatenate([text_neg_batch, text_full, text_full], axis=0)
    # (36, 77, 1024)

    # DINO features for CFG: replicate for all 36 samples
    # But set to zero for negative (uncond) batch
    dino_for_cfg = mx.broadcast_to(
        dino_proj, (n_pbr * n_views, dino_proj.shape[1], dino_proj.shape[2])
    )
    dino_zeros = mx.zeros_like(dino_for_cfg)
    dino_cfg = mx.concatenate([dino_zeros, dino_for_cfg, dino_for_cfg], axis=0)

    # ------------------------------------------------------------------
    # 4. View-dependent guidance scale
    # ------------------------------------------------------------------
    view_scales = [_cam_mapping(a) for a in camera_azims]
    # Repeat for materials: [albedo_scales, mr_scales]
    view_scale_flat = np.array(view_scales * n_pbr, dtype=np.float32)
    view_scale = mx.array(view_scale_flat)[:, None, None, None]  # (12, 1, 1, 1)

    # ------------------------------------------------------------------
    # 5. Prepare condition latents for concat
    # ------------------------------------------------------------------
    # Replicate conditions for N_pbr materials
    z_normal_rep = mx.concatenate([z_normal] * n_pbr, axis=0)  # (12, h, w, 4)
    z_pos_rep = mx.concatenate([z_pos] * n_pbr, axis=0)        # (12, h, w, 4)

    # ------------------------------------------------------------------
    # 6. Denoising loop
    # ------------------------------------------------------------------
    latents = mx.random.normal((n_pbr * n_views, latent_size, latent_size, 4))
    model.scheduler.set_timesteps(num_inference_steps)

    print(f"  Denoising ({num_inference_steps} steps, {n_views} views, CFG={guidance_scale})...")

    for i, t in enumerate(model.scheduler.timesteps):
        t_int = int(t)

        # Process all n_views in a single forward so the multiview
        # attention path inside BasicTransformerBlock fires (it guards on
        # n_views > 1). Combined with the 3D RoPE positional encoding
        # from calc_multires_voxel_idxs and the norm_hs-reuse fix, the
        # cross-view consistency this provides is a big part of matching
        # PT's output.
        chunk_size = n_views
        t_arr = mx.array([t_int])
        noise_guided = mx.zeros_like(latents)

        for v_start in range(0, n_views, chunk_size):
            v_end = min(v_start + chunk_size, n_views)
            n_chunk = v_end - v_start

            # Gather chunk indices for both materials
            # Order: [albedo_v_start..v_end, mr_v_start..v_end]
            # Maps into full latent array ordered as
            # [albedo_0..N-1, mr_0..N-1]
            chunk_idx = list(range(v_start, v_end)) + [
                n_views + j for j in range(v_start, v_end)
            ]
            lat_chunk = latents[mx.array(chunk_idx)]           # (2*n_chunk, h, w, 4)
            zn_chunk = z_normal_rep[mx.array(chunk_idx)]
            zp_chunk = z_pos_rep[mx.array(chunk_idx)]
            unet_in = mx.concatenate([lat_chunk, zn_chunk, zp_chunk], axis=-1)

            n_c = len(chunk_idx)
            text_neg_c = text_neg_batch[:n_c]
            text_full_c = text_full[mx.array(chunk_idx)]
            dino_c = dino_for_cfg[:n_c]

            # 3 CFG passes (sequential to save memory).
            # Mirrors the PyTorch pipeline (hunyuanpaintpbr/pipeline.py:300+,
            # hunyuanpaintpbr/pipeline.py:685):
            #   uncond  -> ref_scale=0, dino=zero, prompt=neg
            #   ref     -> ref_scale=1, dino=zero, prompt=pos
            #   full    -> ref_scale=1, dino=full, prompt=pos
            # Then: noise = uncond + g*v*(ref - uncond) + g*v*(full - ref).
            # Earlier we passed DINO to BOTH ref and full, making
            # (full - ref) ~= 0 and dropping half of the guidance.

            # Slice voxel indices to the current chunk's views
            chunk_view_ids = list(range(v_start, v_end))
            chunk_voxel = {}
            for seq_len, vd in voxel_indices.items():
                vi_full = vd["voxel_indices"]  # (1, n_views * GxG, 3)
                gxg = vi_full.shape[1] // n_views
                # Pick this chunk's views
                vi_per_view = vi_full.reshape(1, n_views, gxg, 3)
                vi_chunk = vi_per_view[:, mx.array(chunk_view_ids), :, :]
                vi_chunk = vi_chunk.reshape(1, len(chunk_view_ids) * gxg, 3)
                chunk_voxel[len(chunk_view_ids) * gxg] = {
                    "voxel_indices": vi_chunk,
                    "voxel_resolution": vd["voxel_resolution"],
                }

            # uncond: no DINO, no ref features, neg text
            pred_uncond = model.unet(
                unet_in, t_arr, text_neg_c,
                n_views=n_chunk, n_pbr=n_pbr,
                position_voxel_indices=chunk_voxel,
            )
            mx.synchronize()

            # ref: ref_features ON, DINO OFF, pos text
            ctx_ref = dict(
                n_views=n_chunk, n_pbr=n_pbr,
                position_voxel_indices=chunk_voxel,
            )
            if ref_features is not None:
                ctx_ref["ref_features"] = ref_features
            pred_ref = model.unet(unet_in, t_arr, text_full_c, **ctx_ref)
            mx.synchronize()

            # full: ref_features ON, DINO ON, pos text
            ctx_full = dict(
                n_views=n_chunk, n_pbr=n_pbr,
                dino_features=dino_c,
                position_voxel_indices=chunk_voxel,
            )
            if ref_features is not None:
                ctx_full["ref_features"] = ref_features
            pred_full = model.unet(unet_in, t_arr, text_full_c, **ctx_full)
            mx.synchronize()

            vs = view_scale[mx.array(chunk_idx)]
            guided = (
                pred_uncond
                + guidance_scale * vs * (pred_ref - pred_uncond)
                + guidance_scale * vs * (pred_full - pred_ref)
            )

            # Scatter chunk results back to correct positions in the
            # full latent array.  chunk_idx maps each chunk sample to
            # its position in the [albedo_0..N-1, mr_0..N-1] layout.
            noise_guided[mx.array(chunk_idx)] = guided

        # Scheduler step
        latents = model.scheduler.step(noise_guided, t_int, latents)
        mx.synchronize()

        if i % 5 == 0 or i == num_inference_steps - 1:
            r = float(latents.max() - latents.min())
            print(f"    step {i}/{num_inference_steps}: t={t_int}, range={r:.1f}")

    # ------------------------------------------------------------------
    # 7. Decode (one image at a time to avoid OOM)
    # ------------------------------------------------------------------
    print("  Decoding...")
    all_images = []
    n_total = n_pbr * n_views
    for idx in range(n_total):
        # latents already at scaling_factor magnitude; vae.decode handles
        # the divide internally — pass z as-is.
        z_i = latents[idx : idx + 1]
        dec_i = model.vae.decode(z_i)
        img_i = mx.clip((dec_i + 1) / 2, 0, 1)
        mx.synchronize()
        all_images.append(np.array(img_i[0]))
        if (idx + 1) % n_views == 0:
            print(f"    decoded {idx + 1}/{n_total}")

    result = {
        "albedo": [
            (all_images[i] * 255).astype(np.uint8)
            for i in range(n_views)
        ],
        "mr": [
            (all_images[n_views + i] * 255).astype(np.uint8)
            for i in range(n_views)
        ],
    }
    return result
