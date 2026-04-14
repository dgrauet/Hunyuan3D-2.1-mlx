"""End-to-end image → shape → textured GLB on Apple Silicon (MLX).

Chains Stage 1 (shape generation) and Stage 2 (PBR texture synthesis) into
a single fully-MLX pipeline. The only external asset is the reference image.

Usage:
    python3 tests/test_stage1_to_stage2.py path/to/image.png [out_dir]

Output:
    out_dir/shape.glb     — Stage 1 output (untextured mesh)
    out_dir/textured.glb  — Stage 2 output (PBR textured mesh)
    out_dir/textured.obj  — OBJ form + material_0.png + material.mtl
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hy3dshape"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hy3dpaint"))

DEFAULT_IMAGE = "/tmp/full/ComfyUI_temp_cpzbr_00007_.png"
DEFAULT_OUT = "/tmp/full/stage1_to_stage2"


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)

    shape_glb = os.path.join(out_dir, "shape.glb")
    textured_obj = os.path.join(out_dir, "textured.obj")

    # --- Stage 1: image -> mesh ---
    print(f"[Stage 1] Shape generation from {image_path}")
    from hy3dshape.pipeline_mlx import ShapePipeline

    t0 = time.time()
    shape_pipe = ShapePipeline.from_pretrained("dgrauet/hunyuan3d-2.1-mlx")
    print(f"  pipeline ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    mesh = shape_pipe(
        image_path,
        num_inference_steps=50,
        guidance_scale=7.5,
        octree_resolution=256,
        seed=42,
    )
    print(f"  shape generated in {time.time() - t0:.1f}s "
          f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
    mesh.export(shape_glb)
    print(f"  saved: {shape_glb}")

    # Free the DiT before loading Stage 2 (both are memory-heavy)
    del shape_pipe
    import gc
    gc.collect()

    # --- Stage 2: mesh + image -> textured GLB ---
    print(f"\n[Stage 2] PBR texture synthesis")
    from textureGenPipeline_mlx import (
        Hunyuan3DPaintConfigMLX,
        Hunyuan3DPaintPipelineMLX,
    )

    cfg = Hunyuan3DPaintConfigMLX(max_num_view=6, resolution=512)
    cfg.render_size = 1024
    cfg.texture_size = 2048
    cfg.mlx_num_inference_steps = 15

    t0 = time.time()
    paint_pipe = Hunyuan3DPaintPipelineMLX(cfg)
    print(f"  pipeline ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    paint_pipe(
        mesh_path=shape_glb,
        image_path=image_path,
        output_mesh_path=textured_obj,
        use_remesh=False,
        save_glb=True,
    )
    print(f"  texture synthesized in {time.time() - t0:.1f}s")

    textured_glb = textured_obj.replace(".obj", ".glb")
    if os.path.exists(textured_glb):
        size_mb = os.path.getsize(textured_glb) / 1024 / 1024
        print(f"\nDone. {textured_glb}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
