"""End-to-end paint inference on the real mermaid mesh + astronaut reference.

Validates that the full MLX pipeline produces coherent textures after the
attention-head-dim fix (commit message for context). Outputs 6-view albedo
and MR images to /tmp/full/e2e_fixed/.

Usage:
    cd /Users/dgrauet/Work/Hunyuan3D-2.1-mlx/hy3dpaint
    python3 tests/test_e2e_paint.py
"""

import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MESH_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "case_2", "mesh.glb"
)
REF_IMAGE_PATH = "/tmp/full/ComfyUI_temp_cpzbr_00007_.png"
# HF repo ID (auto-downloaded via huggingface_hub) or local path override
# via HUNYUAN3D_MLX_WEIGHTS_DIR env var.
WEIGHTS_SOURCE = "dgrauet/hunyuan3d-2.1-mlx"
OUT_DIR = "/tmp/full/e2e_fixed"

VIEW_SIZE = 512
N_VIEWS = 6
NUM_STEPS = 15
CFG = 3.0
AZIMS = [0, 60, 120, 180, 240, 300]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Render conditions ---
    print(f"[1/4] Rendering {N_VIEWS} views at {VIEW_SIZE}px from {MESH_PATH}")
    import trimesh
    from DifferentiableRenderer.mesh_render_mlx import MeshRenderMLX

    mesh = trimesh.load(MESH_PATH, force="mesh")
    renderer = MeshRenderMLX(default_resolution=VIEW_SIZE, shader_type="face")
    renderer.load_mesh(mesh)

    normal_maps = []
    position_maps = []
    for i, az in enumerate(AZIMS):
        n = renderer.render_normal(0, az, bg_color=(0.5, 0.5, 0.5))
        p = renderer.render_position(0, az, bg_color=(0.5, 0.5, 0.5))
        normal_maps.append(np.asarray(n, dtype=np.float32))
        position_maps.append(np.asarray(p, dtype=np.float32))
        Image.fromarray((n * 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, f"cond_normal_{i}_az{az}.png")
        )
        Image.fromarray((p * 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, f"cond_position_{i}_az{az}.png")
        )
    print(f"      rendered {len(normal_maps)} normal + position maps")

    # --- Reference image ---
    ref = np.array(Image.open(REF_IMAGE_PATH).convert("RGB"))
    print(f"[2/4] Reference image: {ref.shape}, dtype={ref.dtype}")

    # --- Load model ---
    print("[3/4] Loading HunyuanPaintPBR MLX weights...")
    from hunyuanpaintpbr_mlx.load_model import HunyuanPaintModelMLX

    t0 = time.time()
    model = HunyuanPaintModelMLX.from_pretrained(WEIGHTS_SOURCE)
    print(f"      loaded in {time.time() - t0:.1f}s")

    # --- Generate ---
    print(f"[4/4] Running denoising ({NUM_STEPS} steps, CFG={CFG})...")
    from hunyuanpaintpbr_mlx.inference import generate_multiview_pbr

    t0 = time.time()
    result = generate_multiview_pbr(
        model=model,
        normal_maps=normal_maps,
        position_maps=position_maps,
        reference_image=ref,
        camera_azims=AZIMS,
        num_inference_steps=NUM_STEPS,
        guidance_scale=CFG,
        view_size=VIEW_SIZE,
        seed=42,
    )
    print(f"      denoise+decode in {time.time() - t0:.1f}s")

    # --- Save ---
    for i, (al, mr) in enumerate(zip(result["albedo"], result["mr"])):
        Image.fromarray(al).save(
            os.path.join(OUT_DIR, f"albedo_{i}_az{AZIMS[i]}.png")
        )
        Image.fromarray(mr).save(
            os.path.join(OUT_DIR, f"mr_{i}_az{AZIMS[i]}.png")
        )
    print(f"\nOutputs written to {OUT_DIR}")

    # Report rough stats on the first albedo view to catch cyan-blob regressions
    a0 = result["albedo"][0].astype(np.float32) / 255.0
    mean_rgb = a0.reshape(-1, 3).mean(axis=0)
    std_rgb = a0.reshape(-1, 3).std(axis=0)
    print(f"      albedo[0] mean RGB: {mean_rgb.round(3).tolist()}  "
          f"std: {std_rgb.round(3).tolist()}")


if __name__ == "__main__":
    main()
