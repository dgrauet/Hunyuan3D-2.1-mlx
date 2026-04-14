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

MESH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "case_2", "mesh.glb"
)
REF_IMAGE_PATH = "/tmp/full/ComfyUI_temp_cpzbr_00007_.png"
OUT_DIR = "/tmp/full/e2e_mesh"
OUTPUT_OBJ = os.path.join(OUT_DIR, "textured_astronaut.obj")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    from textureGenPipeline_mlx import (
        Hunyuan3DPaintConfigMLX,
        Hunyuan3DPaintPipelineMLX,
    )

    # Lower render/texture size for speed on laptop. The default (2048 render,
    # 4096 texture) is fine on a beefier box.
    cfg = Hunyuan3DPaintConfigMLX(max_num_view=6, resolution=512)
    cfg.render_size = 1024
    cfg.texture_size = 2048
    cfg.use_mlx_diffusion = True
    cfg.mlx_num_inference_steps = 15

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
