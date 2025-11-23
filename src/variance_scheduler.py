"""
Variance/noise scheduling for diffusion models.
"""
import torch
import numpy as np
import math


class VarianceScheduler:
    """
    Variance scheduler for diffusion models.

    Manages the forward diffusion process where variance stays at 1
    while the signal drifts to 0 over T timesteps.

    Args:
        num_timesteps: Number of diffusion timesteps (T)
        beta_start: Starting variance (small noise)
        beta_end: Ending variance (large noise)
        schedule_type: 'linear' or 'cosine'
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'linear'
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        # Generate beta schedule
        if schedule_type == 'linear':
            self.betas = self._linear_schedule()
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule()
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")

        # Compute alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # Compute cumulative product: alpha_bar_t = prod(alpha_i) for i=1 to t
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Forward process coefficients
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Posterior variance for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _linear_schedule(self) -> np.ndarray:
        """Linear beta schedule from beta_start to beta_end."""
        return np.linspace(self.beta_start, self.beta_end, self.num_timesteps)

    def _cosine_schedule(self, s: float = 0.008) -> np.ndarray:
        """
        Cosine beta schedule as in "Improved DDPM".

        Args:
            s: Small offset to prevent beta from being too small near t=0
        """
        steps = self.num_timesteps + 1
        x = np.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = np.cos(((x / self.num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)

    def forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None
    ) -> tuple:
        """
        Forward diffusion process: add noise to x_0 to get x_t.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        This keeps variance at 1: Var(x_t) = alpha_bar_t + (1 - alpha_bar_t) = 1
        While the mean drifts: E[x_t] = sqrt(alpha_bar_t) * x_0 -> 0 as t -> T

        Args:
            x_0: Original data (batch, channels, height, width)
            t: Timestep indices (batch,)
            noise: Optional noise (generated if None)

        Returns:
            (x_t, noise): Noisy data and the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for timestep t
        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def _extract(
        self,
        arr: np.ndarray,
        timesteps: torch.Tensor,
        broadcast_shape: tuple
    ) -> torch.Tensor:
        """
        Extract values from array at timesteps and reshape for broadcasting.

        Args:
            arr: NumPy array of shape (num_timesteps,)
            timesteps: Timestep indices (batch,)
            broadcast_shape: Target shape (batch, channels, height, width)

        Returns:
            Tensor of shape (batch, 1, 1, 1) for broadcasting
        """
        device = timesteps.device
        res = torch.from_numpy(arr).to(device)[timesteps].float()

        # Reshape to (batch, 1, 1, 1)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]

        return res
