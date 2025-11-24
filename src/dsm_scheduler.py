"""
DSM Scheduler - Single fixed noise level for Denoising Score Matching.

Unlike DDPM which uses a noise schedule over many timesteps, DSM uses
a single fixed noise level σ for training.
"""
import torch
import numpy as np


class DSMScheduler:
    """
    Simple scheduler for DSM with a single fixed noise level.

    Args:
        sigma: Fixed noise level (typically 0.1 to 1.0 depending on data scale)
               For images normalized to [-1, 1], σ ≈ 0.3 to 0.5 works well
    """

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def add_noise(self, x0, noise=None):
        """
        Add Gaussian noise to clean data.

        x̃ = x + σε where ε ~ N(0, I)

        Args:
            x0: Clean data tensor
            noise: Optional pre-generated noise (if None, will generate)

        Returns:
            Tuple of (noisy_data, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        noisy = x0 + self.sigma * noise

        return noisy, noise

    def __repr__(self):
        return f"DSMScheduler(sigma={self.sigma})"
