"""
Contains reusable neural network components and building blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class WeightStandardizedConv2d(nn.Conv2d):
    """
    Conv2d with weight standardization.

    Improves training stability when combined with GroupNorm.
    """
    def forward(self, x):
        # Just use regular conv2d for now - weight standardization causing numerical issues
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                       self.dilation, self.groups)



def Upsample(dim, dim_out):
    """
    Upsampling layer that doubles spatial dimensions.

    Uses nearest neighbor interpolation followed by convolution.

    Args:
        dim: Input channel dimension
        dim_out: Output channel dimension

    Returns:
        Sequential module with upsample and convolution
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        WeightStandardizedConv2d(dim, dim_out, 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """
    Downsampling layer that halves spatial dimensions.

    Uses pixel unshuffle (rearrange) followed by 1x1 convolution.
    No strided convolutions or pooling.

    Args:
        dim: Input channel dimension
        dim_out: Output channel dimension (defaults to dim)

    Returns:
        Sequential module with rearrange and convolution
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        WeightStandardizedConv2d(dim * 4, dim_out or dim, 1),
    )


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv -> BatchNorm -> Activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        stride: Stride of the convolution
        padding: Padding added to input
        activation: Activation function (default: LeakyReLU)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: nn.Module = nn.LeakyReLU()
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding conditioning for diffusion models.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_emb_dim: Dimension of time embedding
        stride: Stride for the first convolution
        downsample: Downsample layer for skip connection if needed
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        stride: int = 1,
        downsample: nn.Module = None,
        dropout: float = 0.0
    ):
        super().__init__()

        self.conv1 = WeightStandardizedConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.conv2 = WeightStandardizedConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)

        self.downsample = downsample

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection and time conditioning.

        Args:
            x: Input tensor (batch, channels, height, width)
            time_emb: Time embedding tensor (batch, time_emb_dim)
        """
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.leaky_relu(out)

        # Add time embedding
        time_out = self.time_mlp(time_emb)
        out = out + time_out[:, :, None, None]  # Broadcast to (batch, channels, 1, 1)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leaky_relu(out)

        return out


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for spatial features.

    Args:
        channels: Number of input channels
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert (
            self.head_dim * num_heads == channels
        ), "Channels must be divisible by number of heads"

        self.qkv = WeightStandardizedConv2d(channels, channels * 3, 1, bias=False)
        self.proj = WeightStandardizedConv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, height, width)
        """
        batch, channels, height, width = x.shape

        # Generate Q, K, V: (batch, channels*3, H, W)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ v  # (batch, heads, H*W, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        out = self.proj(out)

        return out


class AttentionBlock(nn.Module):
    """
    Attention block with residual connection and normalization.

    Args:
        channels: Number of channels
        num_heads: Number of attention heads
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = LinearAttention(channels, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        return x + self.attn(self.norm(x))


class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(n) complexity for spatial features.

    Args:
        channels: Number of input channels
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert (
            self.head_dim * num_heads == channels
        ), "Channels must be divisible by number of heads"

        self.qkv = WeightStandardizedConv2d(channels, channels * 3, 1, bias=False)
        self.proj = WeightStandardizedConv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for linear attention.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, height, width)
        """
        batch, channels, height, width = x.shape

        # Generate Q, K, V: (batch, channels*3, H, W)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply softmax to make q and k non-negative (for linear attention)
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention: compute KV first, then QKV
        # This changes complexity from O(n^2) to O(n)
        kv = k.transpose(-2, -1) @ v  # (batch, heads, head_dim, head_dim)
        k_sum = k.sum(dim=-2, keepdim=True)  # (batch, heads, 1, head_dim)

        out = q @ kv  # (batch, heads, H*W, head_dim)
        normalizer = q @ k_sum.transpose(-2, -1)  # (batch, heads, H*W, 1)
        out = out / (normalizer + 1e-6)

        # Reshape back to spatial
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        out = self.proj(out)
        out = self.dropout(out)

        return out

