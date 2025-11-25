"""
NCSN Scheduler - Multiple noise levels for Noise Conditional Score Networks.

Uses a geometric sequence of noise levels from σ_max to σ_min.
"""
import torch
import numpy as np


class NCSNScheduler:
    """
    Multi-scale noise scheduler for NCSN.

    Args:
        num_scales: Number of noise levels L (typically 10-232)
        sigma_min: Smallest noise level (e.g., 0.01)
        sigma_max: Largest noise level (e.g., 1.0 for [-1,1] images)
        schedule_type: 'geometric' or 'cosine'
    """

    def __init__(
        self,
        num_scales: int = 10,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        schedule_type: str = 'geometric'
    ):
        self.num_scales = num_scales
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.schedule_type = schedule_type

        if schedule_type == 'geometric':
            # Geometric sequence: σ_i = σ_max * (σ_min/σ_max)^(i/(L-1))
            # This gives σ_0 = σ_max, σ_{L-1} = σ_min
            self.sigmas = torch.tensor(
                np.geomspace(sigma_max, sigma_min, num_scales),
                dtype=torch.float32
            )
        elif schedule_type == 'cosine':
            # Cosine schedule (inspired by improved DDPM)
            # More noise levels in the middle range for better coverage
            steps = np.arange(num_scales)
            # Map to [0, pi/2] for cosine - REVERSED so we go high to low
            alphas = np.cos(((steps / (num_scales - 1)) + 0.008) / 1.008 * np.pi / 2) ** 2
            # Convert to sigma values
            # Level 0 should be sigma_max, level L-1 should be sigma_min
            sigmas = sigma_max + (sigma_min - sigma_max) * (1 - alphas)
            self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}. Use 'geometric' or 'cosine'.")

        # Also store as numpy for convenience
        self.sigmas_np = self.sigmas.numpy()

    def get_sigma(self, level_idx: torch.Tensor) -> torch.Tensor:
        """
        Get sigma value for given noise level indices.

        Args:
            level_idx: Tensor of noise level indices (0 to num_scales-1)
                       0 = highest noise (σ_max), L-1 = lowest noise (σ_min)

        Returns:
            Tensor of sigma values
        """
        return self.sigmas[level_idx]

    def add_noise(
        self,
        x0: torch.Tensor,
        level_idx: torch.Tensor,
        noise: torch.Tensor = None
    ) -> tuple:
        """
        Add noise at specified noise level.

        VE (Variance Exploding): x̃ = x + σ_i * ε  where ε ~ N(0, I)

        Args:
            x0: Clean data (batch, C, H, W)
            level_idx: Noise level indices (batch,)
            noise: Optional pre-generated noise

        Returns:
            (noisy_data, noise, sigma)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Get sigma for each sample in batch
        sigma = self.sigmas.to(x0.device)[level_idx]  # (batch,)
        sigma = sigma.view(-1, 1, 1, 1)  # (batch, 1, 1, 1) for broadcasting

        noisy = x0 + sigma * noise

        return noisy, noise, sigma.squeeze()

    def __repr__(self):
        return (f"NCSNScheduler(num_scales={self.num_scales}, "
                f"sigma_min={self.sigma_min}, sigma_max={self.sigma_max}, "
                f"schedule_type='{self.schedule_type}')")


