"""Flow matching Euler discrete scheduler for MLX.

MLX port of the FlowMatchEulerDiscreteScheduler from schedulers.py.
Timesteps are reversed compared to standard diffusers convention.
"""

import mlx.core as mx
import numpy as np


class FlowMatchEulerDiscreteScheduler:
    """Euler discrete scheduler for flow matching.

    Implements the denoising schedule used by Hunyuan3D-2.1 DiT.
    Timesteps go from 0 to 1 (reversed from standard diffusers convention).

    Args:
        num_train_timesteps: Number of training timesteps.
        shift: Timestep shift factor.
    """

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigmas = None
        self.timesteps = None
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int, sigmas=None):
        """Set the discrete timesteps for inference.

        Matches the original PyTorch scheduler: ascending sigmas from ~0 to ~1,
        with a terminal sigma=1.0 appended for the last Euler step.

        Args:
            num_inference_steps: Number of denoising steps.
            sigmas: Optional custom sigma schedule (numpy array).
        """
        if sigmas is None:
            # Original logic: linspace from sigma_max to sigma_min where
            # sigma_max = sigmas[0] ≈ 0.001 (near clean) and
            # sigma_min = sigmas[-1] = 1.0 (full noise) — confusing names!
            # Result: ASCENDING sigmas from ~0.001 to ~1.0
            timesteps = np.linspace(
                1, self.num_train_timesteps, num_inference_steps, dtype=np.float32
            )
            sigmas = timesteps / self.num_train_timesteps
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        else:
            sigmas = np.asarray(sigmas, dtype=np.float32)
            sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

        timesteps = sigmas * self.num_train_timesteps

        self.timesteps = mx.array(timesteps, dtype=mx.float32)
        # Append terminal sigma=1.0 (matches original torch.cat([sigmas, ones(1)]))
        self.sigmas = mx.array(
            np.concatenate([sigmas, np.ones(1, dtype=np.float32)]), dtype=mx.float32
        )
        self._step_index = None
        self.num_inference_steps = num_inference_steps

    def _init_step_index(self, timestep):
        """Initialize step index for the first step."""
        if self._step_index is None:
            self._step_index = 0

    def step(self, model_output: mx.array, timestep: mx.array, sample: mx.array) -> mx.array:
        """Perform one Euler step.

        Args:
            model_output: Predicted velocity from the model.
            timestep: Current timestep (used to init step index).
            sample: Current noisy sample.

        Returns:
            Denoised sample after one step.
        """
        if self._step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        self._step_index += 1
        return prev_sample

    def add_noise(self, original: mx.array, noise: mx.array, sigma: mx.array) -> mx.array:
        """Add noise to sample at given sigma level (flow matching interpolation)."""
        return sigma * noise + (1.0 - sigma) * original
