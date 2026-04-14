#!/usr/bin/env python3
"""Convert RealESRGAN (RRDBNet) PyTorch weights to MLX format.

Usage:
    python convert_realesrgan.py --input path/to/RealESRGAN_x4plus.pth --output path/to/realesrgan_mlx.safetensors

The script:
  1. Loads the PyTorch .pth checkpoint.
  2. Remaps the keys from PyTorch's Sequential body (body.0, body.1, ...) to MLX list
     indexing (body.0, body.1, ...) -- these happen to match.
  3. Transposes Conv2d weights from PyTorch layout (O, I, H, W) to MLX layout (O, H, W, I).
  4. Saves the result as .safetensors or .npz.
"""

import argparse
import re
from pathlib import Path

import mlx.core as mx
import numpy as np


def convert_weights(input_path: str, output_path: str) -> None:
    """Convert PyTorch RRDBNet weights to MLX format.

    Args:
        input_path: Path to PyTorch .pth file.
        output_path: Path to output .safetensors or .npz file.
    """
    import torch

    # Load PyTorch checkpoint
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    # Some checkpoints wrap state_dict under a key
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    mlx_weights = {}
    for key, value in state_dict.items():
        arr = value.numpy()

        # Conv2d weights: PyTorch (O, I, H, W) -> MLX (O, H, W, I)
        if "weight" in key and arr.ndim == 4:
            arr = np.transpose(arr, (0, 2, 3, 1))

        mlx_weights[key] = mx.array(arr)

    # Save
    if output_path.endswith(".npz"):
        mx.savez(output_path, **mlx_weights)
    elif output_path.endswith(".safetensors"):
        mx.save_safetensors(output_path, mlx_weights)
    else:
        raise ValueError(f"Unsupported output format: {output_path}. Use .safetensors or .npz")

    print(f"Converted {len(mlx_weights)} tensors from {input_path} -> {output_path}")

    # Print some stats
    conv_count = sum(1 for k in mlx_weights if "weight" in k and mlx_weights[k].ndim == 4)
    print(f"  Conv2d weight tensors transposed: {conv_count}")
    total_params = sum(v.size for v in mlx_weights.values())
    print(f"  Total parameters: {total_params:,}")


def main():
    parser = argparse.ArgumentParser(description="Convert RealESRGAN PyTorch weights to MLX format")
    parser.add_argument("--input", "-i", required=True, help="Path to PyTorch .pth checkpoint")
    parser.add_argument("--output", "-o", required=True, help="Output path (.safetensors or .npz)")
    args = parser.parse_args()

    convert_weights(args.input, args.output)


if __name__ == "__main__":
    main()
