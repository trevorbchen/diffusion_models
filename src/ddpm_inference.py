"""
Visualize DDPM denoising process step by step.
"""
import torch
import matplotlib.pyplot as plt
from unet import UNet
from variance_scheduler import VarianceScheduler
from ddpm import DDPM
import numpy as np


def visualize_denoising_steps(model_path, num_samples=4, device='cuda', seed=None):
    """
    Visualize the denoising process step by step.

    Args:
        model_path: Path to saved model checkpoint
        num_samples: Number of samples to generate
        device: Device to use
        seed: Random seed for reproducibility (if None, no seed is set)
    """
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model_config = checkpoint['model_config']
    model = UNet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create scheduler
    scheduler_config = checkpoint['scheduler_config']
    scheduler = VarianceScheduler(**scheduler_config)

    print(f"Scheduler: T={scheduler.num_timesteps} steps")

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    # Start from pure noise
    x = torch.randn(num_samples, 1, 28, 28).to(device)

    # Reverse diffusion - save snapshots
    timesteps = list(range(scheduler.num_timesteps))[::-1]

    # Save every N steps
    save_interval = scheduler.num_timesteps // 10
    snapshots = []
    snapshot_timesteps = []

    print("\nDenoising process:")
    for i, t in enumerate(timesteps):
        # Save snapshot
        if t % save_interval == 0 or t == 0:
            snapshot = x.cpu().numpy()
            snapshot = (snapshot + 1) / 2  # [-1, 1] -> [0, 1]
            snapshot = np.clip(snapshot, 0, 1)
            snapshots.append(snapshot)
            snapshot_timesteps.append(t)
            print(f"  t={t:3d}: Saved snapshot")

        # Predict noise
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)

        with torch.no_grad():
            predicted_noise = model(x, t_batch)

        # Get alpha values (convert to tensor)
        alpha_t = torch.tensor(scheduler.alphas[t], device=device)
        alpha_bar_t = torch.tensor(scheduler.alphas_cumprod[t], device=device)

        # Compute posterior mean
        if t > 0:
            alpha_bar_t_prev = torch.tensor(scheduler.alphas_cumprod[t - 1], device=device)
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)

        # Predicted x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Compute mean
        beta_t = torch.tensor(scheduler.betas[t], device=device)
        mean = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t) * pred_x0
        mean += (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t) * x

        if t > 0:
            # Add noise
            noise = torch.randn_like(x)
            variance = torch.tensor(scheduler.posterior_variance[t], device=device)
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

    # Plot snapshots
    print("\nCreating visualization...")
    n_steps = len(snapshots)
    fig, axes = plt.subplots(num_samples, n_steps, figsize=(n_steps * 2, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('DDPM Denoising Process: Pure Noise → Clean Image', fontsize=14)

    for i in range(num_samples):
        for j in range(n_steps):
            axes[i, j].imshow(snapshots[j][i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f't={snapshot_timesteps[j]}', fontsize=10)

    plt.tight_layout()

    # Save
    import os
    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_denoising_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.show()

    return snapshots, snapshot_timesteps


if __name__ == "__main__":
    import sys

    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "../models/ddpm_final.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    snapshots, timesteps = visualize_denoising_steps(model_path, num_samples=4, device=device, seed=42)

    print(f"\nCaptured {len(snapshots)} snapshots at timesteps: {timesteps}")
    print("Each row shows one sample being gradually denoised from random noise to a clean image!")
