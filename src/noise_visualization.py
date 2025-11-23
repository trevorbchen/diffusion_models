# -*- coding: utf-8 -*-
"""
Visualization utilities for diffusion process and noise schedules.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from variance_scheduler import VarianceScheduler


def load_sample_from_dataset(data_dir: str = "../data", index: int = 0) -> torch.Tensor:
    """
    Load a sample tensor from the training dataset.

    Args:
        data_dir: Path to data directory
        index: Index of sample to load

    Returns:
        Image tensor of shape (C, H, W)
    """
    data_path = Path(data_dir)

    # Try to load .pt or .pth files (exclude batch files)
    tensor_files = [f for f in list(data_path.glob("*.pt")) + list(data_path.glob("*.pth"))
                    if "batch" not in f.name]

    if len(tensor_files) == 0:
        raise FileNotFoundError(f"No tensor files found in {data_dir}")

    # Load the requested sample
    sample_file = tensor_files[min(index, len(tensor_files) - 1)]
    sample = torch.load(sample_file)

    # Handle dict case (in case there are still some)
    if isinstance(sample, dict):
        sample = sample["images"][0] if "images" in sample else sample[list(sample.keys())[0]]

    return sample


def visualize_noise_schedule(
    scheduler: VarianceScheduler,
    save_path: str = None
):
    """
    Visualize the beta, alpha, and alpha_bar schedules.

    Args:
        scheduler: VarianceScheduler instance
        save_path: Optional path to save figure
    """
    timesteps = np.arange(scheduler.num_timesteps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Beta schedule
    axes[0].plot(timesteps, scheduler.betas, linewidth=2)
    axes[0].set_title(f'Beta Schedule ({scheduler.schedule_type})')
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('�_t')
    axes[0].grid(True, alpha=0.3)

    # Alpha schedule
    axes[1].plot(timesteps, scheduler.alphas, linewidth=2)
    axes[1].set_title('Alpha (1 - �_t)')
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel('�_t')
    axes[1].grid(True, alpha=0.3)

    # Alpha bar (cumulative product)
    axes[2].plot(timesteps, scheduler.alphas_cumprod, linewidth=2, color='red')
    axes[2].set_title('Alpha Bar (Cumulative Product)')
    axes[2].set_xlabel('Timestep t')
    axes[2].set_ylabel('�_t')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_forward_diffusion(
    image: torch.Tensor,
    scheduler: VarianceScheduler,
    timesteps: list = [0, 100, 250, 500, 750, 999],
    save_path: str = None
):
    """
    Visualize forward diffusion process at different timesteps.

    Args:
        image: Image tensor of shape (C, H, W)
        scheduler: VarianceScheduler instance
        timesteps: List of timesteps to visualize
        save_path: Optional path to save figure
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add batch dimension

    num_steps = len(timesteps)
    fig, axes = plt.subplots(1, num_steps, figsize=(3 * num_steps, 3))

    if num_steps == 1:
        axes = [axes]

    for idx, t in enumerate(timesteps):
        # Create timestep tensor
        t_tensor = torch.tensor([t], dtype=torch.long)

        # Apply forward diffusion
        x_t, noise = scheduler.forward_diffusion(image, t_tensor)

        # Convert to displayable format
        x_t = x_t.squeeze(0).cpu()

        # Normalize to [0, 1]
        x_t_normalized = (x_t - x_t.min()) / (x_t.max() - x_t.min() + 1e-8)

        # Display
        if x_t.shape[0] == 1:  # Grayscale
            axes[idx].imshow(x_t_normalized.squeeze(0), cmap='gray')
        else:  # RGB
            axes[idx].imshow(x_t_normalized.permute(1, 2, 0))

        # Add title with signal-to-noise info
        alpha_bar = scheduler.alphas_cumprod[t]
        axes[idx].set_title(f't={t}\n�_t={alpha_bar:.3f}')
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_signal_to_noise_ratio(
    scheduler: VarianceScheduler,
    save_path: str = None
):
    """
    Visualize signal and noise coefficients over timesteps.

    Args:
        scheduler: VarianceScheduler instance
        save_path: Optional path to save figure
    """
    timesteps = np.arange(scheduler.num_timesteps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Signal vs Noise magnitude
    ax1.plot(timesteps, scheduler.sqrt_alphas_cumprod, label='Signal: �_t', linewidth=2)
    ax1.plot(timesteps, scheduler.sqrt_one_minus_alphas_cumprod,
             label='Noise: (1-�_t)', linewidth=2)
    ax1.set_title('Signal vs Noise Coefficients')
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('Coefficient')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SNR
    snr = scheduler.sqrt_alphas_cumprod / (scheduler.sqrt_one_minus_alphas_cumprod + 1e-8)
    ax2.plot(timesteps, snr, linewidth=2, color='purple')
    ax2.set_title('Signal-to-Noise Ratio')
    ax2.set_xlabel('Timestep t')
    ax2.set_ylabel('SNR')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def compare_schedules(
    image: torch.Tensor,
    timestep: int = 500,
    save_path: str = None
):
    """
    Compare linear vs cosine schedules on the same image.

    Args:
        image: Image tensor of shape (C, H, W)
        timestep: Timestep to compare at
        save_path: Optional path to save figure
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Create both schedulers
    linear_scheduler = VarianceScheduler(schedule_type='linear')
    cosine_scheduler = VarianceScheduler(schedule_type='cosine')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original
    original = image.squeeze(0).cpu()
    if original.shape[0] == 1:
        axes[0].imshow(original.squeeze(0), cmap='gray')
    else:
        axes[0].imshow(original.permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Linear schedule
    t_tensor = torch.tensor([timestep], dtype=torch.long)
    x_t_linear, _ = linear_scheduler.forward_diffusion(image, t_tensor)
    x_t_linear = x_t_linear.squeeze(0).cpu()
    x_t_linear = (x_t_linear - x_t_linear.min()) / (x_t_linear.max() - x_t_linear.min() + 1e-8)

    if x_t_linear.shape[0] == 1:
        axes[1].imshow(x_t_linear.squeeze(0), cmap='gray')
    else:
        axes[1].imshow(x_t_linear.permute(1, 2, 0))
    axes[1].set_title(f'Linear (t={timestep})')
    axes[1].axis('off')

    # Cosine schedule
    x_t_cosine, _ = cosine_scheduler.forward_diffusion(image, t_tensor)
    x_t_cosine = x_t_cosine.squeeze(0).cpu()
    x_t_cosine = (x_t_cosine - x_t_cosine.min()) / (x_t_cosine.max() - x_t_cosine.min() + 1e-8)

    if x_t_cosine.shape[0] == 1:
        axes[2].imshow(x_t_cosine.squeeze(0), cmap='gray')
    else:
        axes[2].imshow(x_t_cosine.permute(1, 2, 0))
    axes[2].set_title(f'Cosine (t={timestep})')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_original_grid(data_dir: str = "../data", grid_size: tuple = (7, 5)):
    """
    Visualize a grid of original images from the dataset.

    Args:
        data_dir: Path to data directory
        grid_size: Tuple of (rows, cols) for grid
    """
    rows, cols = grid_size
    num_images = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_images):
        try:
            sample = load_sample_from_dataset(data_dir, index=i)

            # Display
            if sample.shape[0] == 1:  # Grayscale
                # Denormalize from [-1, 1] to [0, 1]
                img = (sample.squeeze(0) + 1) / 2
                axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            else:  # RGB
                img = (sample + 1) / 2
                axes[i].imshow(img.permute(1, 2, 0))

            axes[i].axis('off')
        except:
            axes[i].axis('off')

    plt.suptitle('Original Fashion MNIST Images', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Show grid of original images
    print("Showing original images grid (7x5)...")
    visualize_original_grid()

    # Load sample
    try:
        sample = load_sample_from_dataset()

        # Linear schedule diffusion
        print("Visualizing LINEAR schedule diffusion...")
        linear_scheduler = VarianceScheduler(num_timesteps=1000, schedule_type='linear')
        visualize_forward_diffusion(
            sample,
            linear_scheduler,
            timesteps=[0, 50, 100, 500, 999]
        )

        # Cosine schedule diffusion
        print("Visualizing COSINE schedule diffusion...")
        cosine_scheduler = VarianceScheduler(num_timesteps=1000, schedule_type='cosine')
        visualize_forward_diffusion(
            sample,
            cosine_scheduler,
            timesteps=[0, 50, 100, 500, 999]
        )

    except FileNotFoundError as e:
        print(f"Could not load sample: {e}")
        print("Please add .pt or .pth tensor files to the data/ directory")
