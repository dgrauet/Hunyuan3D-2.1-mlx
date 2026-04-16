"""Full Stage 2 pipeline: mesh + reference image → textured GLB.

Exercises every MLX stage end-to-end:
  1. Remesh + UV-wrap (CPU/xatlas)
  2. Render normal + position maps (MLX Metal)
  3. Multiview PBR diffusion (MLX — UNet, VAE, DINO, scheduler)
  4. Back-project + bake UV texture (MLX Metal)
  5. UV inpaint gaps (OpenCV)
  6. Save OBJ + convert to GLB

Usage:
    cd /Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint
    python3 tests/test_e2e_mesh.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CASE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "assets", "case_2"
)
MESH_PATH = os.path.join(CASE_DIR, "mesh.glb")
# IMPORTANT: the reference image must match the mesh content. case_2 is a
# mermaid, so we use assets/case_2/image.png. Passing a mismatched reference
# (e.g. an astronaut on a mermaid mesh) asks the diffusion to map incompatible
# geometry — the multiview views may look plausible individually but
# back-projecting them onto the UV atlas of an unrelated mesh yields fragmented
# noise.
REF_IMAGE_PATH = os.path.join(CASE_DIR, "image.png")
OUT_DIR = "/tmp/full/e2e_mesh"
OUTPUT_OBJ = os.path.join(OUT_DIR, "textured_mermaid.obj")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    from textureGenPipeline_mlx import (
        Hunyuan3DPaintConfigMLX,
        Hunyuan3DPaintPipelineMLX,
    )

    # PT reference defaults (demo.py + pipeline.py): max_num_view=6,
    # resolution=512, render_size=2048, texture_size=4096,
    # num_inference_steps=15, guidance_scale=3.0. Matched here EXCEPT
    # texture_size=2048 — the Metal rasterizer times out at 4096 on
    # laptop GPUs (kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
    # because we don't tile the bake. 2048 keeps PT parity everywhere
    # else while staying within the Metal command-buffer budget.
    cfg = Hunyuan3DPaintConfigMLX(max_num_view=6, resolution=512)
    cfg.texture_size = 2048
    cfg.use_mlx_diffusion = True

    print("[1/2] Building pipeline...")
    t0 = time.time()
    pipeline = Hunyuan3DPaintPipelineMLX(cfg)
    print(f"      ready in {time.time() - t0:.1f}s")

    print(f"[2/2] Texturing {MESH_PATH}")
    t0 = time.time()
    out_path = pipeline(
        mesh_path=MESH_PATH,
        image_path=REF_IMAGE_PATH,
        output_mesh_path=OUTPUT_OBJ,
        use_remesh=False,  # mesh is already clean
        save_glb=True,
    )
    print(f"      textured in {time.time() - t0:.1f}s")
    print(f"\nOutput: {out_path}")
    glb = out_path.replace(".obj", ".glb")
    if os.path.exists(glb):
        size_mb = os.path.getsize(glb) / 1024 / 1024
        print(f"        {glb}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
