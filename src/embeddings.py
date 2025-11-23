"""
Time and positional embedding implementations.
"""
import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion models.

    Encodes timesteps using sinusoidal functions of different frequencies.
    Based on the positional encoding from "Attention is All You Need".

    Args:
        embed_dim: Dimension of the embedding (must be even)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for given timesteps.

        Args:
            timesteps: Tensor of shape (batch_size,) containing timestep values

        Returns:
            Embeddings of shape (batch_size, embed_dim)
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2

        # Compute the frequency scaling factors
        # freq = 1 / (10000^(2i/d)) where i is the dimension index
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

        # Outer product: (batch_size, 1) x (1, half_dim) -> (batch_size, half_dim)
        emb = timesteps[:, None] * emb[None, :]

        # Concatenate sin and cos embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb
