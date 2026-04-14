# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""MLX port of RealESRGAN's RRDBNet architecture for image super-resolution.

This is a faithful port of basicsr.archs.rrdbnet_arch.RRDBNet from PyTorch to MLX.
The architecture consists of Residual-in-Residual Dense Blocks (RRDB) used in ESRGAN.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image


def _nearest_upsample_2x(x: mx.array) -> mx.array:
    """Nearest-neighbor 2x upsampling for NHWC tensors.

    Args:
        x: Input array of shape (N, H, W, C).

    Returns:
        Upsampled array of shape (N, 2*H, 2*W, C).
    """
    N, H, W, C = x.shape
    # Repeat along H: (N, H, 1, W, C) -> (N, H, 2, W, C) -> (N, 2H, W, C)
    x = mx.expand_dims(x, axis=2)
    x = mx.repeat(x, repeats=2, axis=2)
    x = x.reshape(N, H * 2, W, C)
    # Repeat along W: (N, 2H, W, 1, C) -> (N, 2H, W, 2, C) -> (N, 2H, 2W, C)
    x = mx.expand_dims(x, axis=3)
    x = mx.repeat(x, repeats=2, axis=3)
    x = x.reshape(N, H * 2, W * 2, C)
    return x


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB block in ESRGAN.

    Contains 5 conv layers with dense connections where each layer receives
    the concatenated outputs of all previous layers.

    Args:
        num_feat: Channel number of intermediate features. Default: 64.
        num_grow_ch: Channels for each growth. Default: 32.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # Dense connections: each layer sees all previous outputs concatenated
        # Concatenation is along the channel axis (axis=-1 for NHWC)
        x1 = nn.leaky_relu(self.conv1(x), negative_slope=0.2)
        x2 = nn.leaky_relu(self.conv2(mx.concatenate([x, x1], axis=-1)), negative_slope=0.2)
        x3 = nn.leaky_relu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)), negative_slope=0.2)
        x4 = nn.leaky_relu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)), negative_slope=0.2)
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        # Residual scaling by 0.2
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Contains 3 ResidualDenseBlock with a residual connection.

    Args:
        num_feat: Channel number of intermediate features.
        num_grow_ch: Channels for each growth. Default: 32.
    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Residual scaling by 0.2
        return out * 0.2 + x


class RRDBNetMLX(nn.Module):
    """Networks consisting of Residual in Residual Dense Block (ESRGAN).

    MLX port of basicsr.archs.rrdbnet_arch.RRDBNet.
    Uses NHWC (channels-last) layout as required by MLX.

    Args:
        num_in_ch: Channel number of inputs. Default: 3.
        num_out_ch: Channel number of outputs. Default: 3.
        scale: Upsampling scale factor. Default: 4.
        num_feat: Channel number of intermediate features. Default: 64.
        num_block: Number of RRDB blocks in the trunk. Default: 23.
        num_grow_ch: Channels for each growth in RDB. Default: 32.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.body = [RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
        self.conv_body = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        # Upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, H, W, C) in NHWC layout.

        Returns:
            Super-resolved output of shape (N, H*scale, W*scale, num_out_ch).
        """
        # pixel_unshuffle for scale 1 or 2 (not needed for scale 4)
        if self.scale == 2:
            feat = self._pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = self._pixel_unshuffle(x, 4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        # Upsample 2x twice for 4x total
        feat = nn.leaky_relu(self.conv_up1(_nearest_upsample_2x(feat)), negative_slope=0.2)
        feat = nn.leaky_relu(self.conv_up2(_nearest_upsample_2x(feat)), negative_slope=0.2)
        out = self.conv_last(nn.leaky_relu(self.conv_hr(feat), negative_slope=0.2))
        return out

    @staticmethod
    def _pixel_unshuffle(x: mx.array, scale: int) -> mx.array:
        """Inverse of pixel shuffle: rearrange spatial dims into channels (NHWC).

        Args:
            x: Input of shape (N, H, W, C).
            scale: Downscaling factor.

        Returns:
            Output of shape (N, H//scale, W//scale, C*scale*scale).
        """
        N, H, W, C = x.shape
        h, w = H // scale, W // scale
        x = x.reshape(N, h, scale, w, scale, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (N, h, w, scale, scale, C)
        x = x.reshape(N, h, w, C * scale * scale)
        return x


class imageSuperNetMLX:
    """MLX wrapper for RealESRGAN super-resolution, matching imageSuperNet interface.

    Args:
        weights_path: Path to converted MLX weights (.safetensors or .npz).
    """

    def __init__(self, weights_path: str):
        self.model = RRDBNetMLX(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        if weights_path.endswith(".npz"):
            weights = dict(mx.load(weights_path))
        elif weights_path.endswith(".safetensors"):
            weights = dict(mx.load(weights_path))
        else:
            raise ValueError(f"Unsupported weight format: {weights_path}")

        self.model.load_weights(list(weights.items()))
        mx.eval(self.model.parameters())

    def __call__(self, image: Image.Image) -> Image.Image:
        """Super-resolve a PIL Image.

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Super-resolved PIL Image (RGB).
        """
        # Convert PIL Image to NHWC float32 array normalized to [0, 1]
        img_np = np.array(image).astype(np.float32) / 255.0
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img_mx = mx.array(img_np[np.newaxis, ...])

        # Run inference
        output = self.model(img_mx)
        mx.eval(output)

        # Convert back to PIL Image
        output_np = np.array(output[0])  # Remove batch dim
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(output_np)
