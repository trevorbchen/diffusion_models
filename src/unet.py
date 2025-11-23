"""
Contains U-Net model architecture for diffusion models.
"""
import torch
import torch.nn as nn
from embeddings import SinusoidalTimeEmbedding
from model_components import ResidualBlock, AttentionBlock, Downsample, Upsample, WeightStandardizedConv2d


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input channels (e.g., 3 for RGB images)
        out_channels: Number of output channels
        dim: Base channel dimension
        dim_mults: Tuple of dimension multipliers for each level
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 64,
        dim_mults: tuple = (1, 2, 4, 8),
        dropout: float = 0.0
    ):
        super().__init__()

        # Time embedding: sinusoidal -> linear -> GELU -> linear
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial convolution
        self.init_conv = WeightStandardizedConv2d(in_channels, dim, kernel_size=7, padding=3)

        # Calculate dimensions for each level
        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Down blocks: ResBlock -> ResBlock -> LinearAttention -> Conv2d -> Downsample
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_in, time_dim, dropout=dropout),
                ResidualBlock(dim_in, dim_in, time_dim, dropout=dropout),
                AttentionBlock(dim_in),
                WeightStandardizedConv2d(dim_in, dim_out, 3, padding=1),
                Downsample(dim_out, dim_out) if not is_last else nn.Identity()
            ]))

        # Middle: ResBlock -> ResBlock -> Attention
        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_dim, dropout=dropout)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(mid_dim)

        # Up blocks: ResBlock -> ResBlock -> LinearAttention -> Conv2d -> Upsample
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            # Reverse the dimensions for upsampling (go from larger to smaller)
            dim_in, dim_out = dim_out, dim_in

            self.ups.append(nn.ModuleList([
                ResidualBlock(
                    dim_in + dim_in,
                    dim_in,
                    time_dim,
                    dropout=dropout,
                    downsample=WeightStandardizedConv2d(dim_in + dim_in, dim_in, 1)  # 1x1 conv to match channels
                ),
                ResidualBlock(dim_in, dim_in, time_dim, dropout=dropout),
                AttentionBlock(dim_in),
                WeightStandardizedConv2d(dim_in, dim_out, 3, padding=1),
                Upsample(dim_out, dim_out) if not is_last else nn.Identity()
            ]))

        # Final output
        self.final_res = ResidualBlock(dim, dim, time_dim, dropout=dropout)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

        # Initialize final conv to small values (HuggingFace approach)
        # Model starts by predicting small noise, learns gradually
        nn.init.normal_(self.final_conv.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            time: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Time embedding
        t = self.time_mlp(time)

        # Initial conv
        x = self.init_conv(x)

        # Store skip connections from downsampling
        skip_connections = []

        # Downsampling path
        for res1, res2, attn, conv, downsample in self.downs:
            x = res1(x, t)
            x = res2(x, t)
            x = attn(x)
            x = conv(x)
            skip_connections.append(x.clone())  # Save copy for skip connection AFTER conv
            x = downsample(x)

        # Middle block
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        x = self.mid_attn(x)

        # Upsampling path
        for res1, res2, attn, conv, upsample in self.ups:
            skip = skip_connections.pop()  # Get corresponding skip connection
            x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
            x = res1(x, t)
            x = res2(x, t)
            x = attn(x)
            x = conv(x)
            x = upsample(x)

        # Final output
        x = self.final_res(x, t)
        x = self.final_conv(x)
        return x
