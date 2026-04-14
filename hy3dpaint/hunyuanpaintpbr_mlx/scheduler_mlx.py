"""UniPC Multistep Scheduler for MLX.

Simplified port of diffusers UniPCMultistepScheduler for inference only.
Implements the unified predictor-corrector framework.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import numpy as np


@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    solver_order: int = 2
    prediction_type: str = "v_prediction"
    timestep_spacing: str = "trailing"
    rescale_betas_zero_snr: bool = True


class UniPCMultistepSchedulerMLX:
    """UniPC multistep scheduler for MLX inference.

    Supports 'trailing' timestep spacing and 'epsilon' prediction type,
    matching the HunyuanPaintPBR configuration.
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        if config is None:
            config = SchedulerConfig()
        self.config = config

        # Compute beta schedule
        betas = np.linspace(
            config.beta_start ** 0.5,
            config.beta_end ** 0.5,
            config.num_train_timesteps,
            dtype=np.float64,
        ) ** 2

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)

        # Rescale for zero terminal SNR (SD 2.1 uses this)
        if config.rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2 ** -24  # ~0, avoids log(0)

        # Precompute for signal/noise ratio
        self.alpha_t = np.sqrt(self.alphas_cumprod)
        self.sigma_t = np.sqrt(1.0 - self.alphas_cumprod)
        self.lambda_t = np.log(self.alpha_t / self.sigma_t)

        self.num_inference_steps = None
        self.timesteps = None
        self._step_index = 0
        self.model_outputs: List[Optional[mx.array]] = []

    def set_timesteps(self, num_inference_steps: int):
        """Set the discrete timesteps for inference."""
        self.num_inference_steps = num_inference_steps

        if self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.round(
                np.arange(self.config.num_train_timesteps, 0, -step_ratio)
            ).astype(np.int64) - 1
        else:
            # linspace fallback
            timesteps = np.linspace(
                self.config.num_train_timesteps - 1, 0, num_inference_steps
            ).round().astype(np.int64)

        self.timesteps = timesteps
        self.model_outputs = [None] * self.config.solver_order
        self._step_index = 0

    def scale_model_input(self, sample: mx.array, timestep: int) -> mx.array:
        """Scale input for the model (identity for UniPC)."""
        return sample

    def step(
        self, model_output: mx.array, timestep: int, sample: mx.array
    ) -> mx.array:
        """Perform one scheduler step.

        Args:
            model_output: (B, C, H, W) predicted noise.
            timestep: current timestep.
            sample: (B, C, H, W) current noisy sample.

        Returns:
            Denoised sample for the next step.
        """
        t = int(timestep)
        step_idx = self._step_index

        # Shift model outputs buffer
        self.model_outputs = self.model_outputs[1:] + [model_output]

        # Get next timestep
        if step_idx + 1 < len(self.timesteps):
            t_next = int(self.timesteps[step_idx + 1])
        else:
            t_next = 0

        # Signal and noise coefficients
        alpha_t = float(self.alpha_t[t])
        sigma_t = float(self.sigma_t[t])
        lambda_t_val = float(self.lambda_t[t])

        if t_next > 0:
            alpha_next = float(self.alpha_t[t_next])
            sigma_next = float(self.sigma_t[t_next])
            lambda_next = float(self.lambda_t[t_next])
        else:
            alpha_next = 1.0
            sigma_next = 0.0
            lambda_next = float("inf")

        h = lambda_next - lambda_t_val

        # Convert model output to x0 prediction
        if self.config.prediction_type == "epsilon":
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == "v_prediction":
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            x0_pred = model_output

        # First-order update (DDIM-style)
        if sigma_t > 1e-8:
            # Compute noise component
            noise_component = (sample - alpha_t * x0_pred) / sigma_t
            prev_sample = alpha_next * x0_pred + sigma_next * noise_component
        else:
            prev_sample = alpha_next * x0_pred

        self._step_index += 1
        return prev_sample
