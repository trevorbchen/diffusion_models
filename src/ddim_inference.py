"""
DDIM (Denoising Diffusion Implicit Models) inference script.
Uses deterministic sampling for faster generation with fewer steps.
"""
import torch
import matplotlib.pyplot as plt
from unet import UNet
from variance_scheduler import VarianceScheduler
import numpy as np


def ddim_sample(model, scheduler, num_samples=4, num_steps=None, eta=0.0, device='cuda', seed=None):
    """
    Generate samples using DDIM sampling.

    DDIM allows for deterministic (eta=0) or stochastic (eta>0) sampling
    with fewer steps than DDPM.

    Args:
        model: Trained UNet model
        scheduler: Variance scheduler
        num_samples: Number of samples to generate
        num_steps: Number of sampling steps (if None, uses all training timesteps)
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM-like)
        device: Device to use
        seed: Random seed for reproducibility (if None, no seed is set)

    Returns:
        Generated images and snapshots
    """
    model.eval()

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    # Start from pure noise
    x = torch.randn(num_samples, 1, 28, 28).to(device)

    # Create timestep sequence
    total_timesteps = scheduler.num_timesteps

    if num_steps is None:
        # Use ALL timesteps (same as DDPM)
        timesteps = list(range(total_timesteps))[::-1]
    else:
        # Use evenly spaced steps from the full diffusion process
        step_size = total_timesteps // num_steps
        timesteps = list(range(0, total_timesteps, step_size))[::-1]

        # Ensure we end at t=0
        if timesteps[-1] != 0:
            timesteps.append(0)

    # Save snapshots
    save_interval = max(1, len(timesteps) // 10)
    snapshots = []
    snapshot_timesteps = []

    print(f"\nDDIM sampling with {len(timesteps)} steps (eta={eta}):")
    print(f"Timesteps: {timesteps[:5]}...{timesteps[-5:]}")

    for i, t in enumerate(timesteps):
        # Save snapshot
        if i % save_interval == 0 or t == 0:
            snapshot = x.cpu().numpy()
            snapshot = (snapshot + 1) / 2  # [-1, 1] -> [0, 1]
            snapshot = np.clip(snapshot, 0, 1)
            snapshots.append(snapshot)
            snapshot_timesteps.append(t)
            print(f"  Step {i}/{len(timesteps)}, t={t:3d}: Saved snapshot")

        # Predict noise
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)

        with torch.no_grad():
            predicted_noise = model(x, t_batch)

        # Get alpha values
        alpha_bar_t = torch.tensor(scheduler.alphas_cumprod[t], device=device)

        # Get next timestep
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_t_prev = torch.tensor(scheduler.alphas_cumprod[t_prev], device=device)
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)

        # Predict x_0 (original image)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # DDIM update rule
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t)) * predicted_noise

        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt

        # Add stochastic noise if eta > 0
        if eta > 0 and t > 0:
            noise = torch.randn_like(x)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
            x_prev = x_prev + sigma_t * noise

        x = x_prev

    return x, snapshots, snapshot_timesteps


def visualize_ddim_denoising(model_path, num_samples=4, num_steps=50, eta=0.0, device='cuda', seed=None):
    """
    Visualize the DDIM denoising process step by step.

    Args:
        model_path: Path to saved model checkpoint
        num_samples: Number of samples to generate
        num_steps: Number of DDIM sampling steps
        eta: Stochasticity parameter (0 = deterministic, 1 = stochastic)
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

    # Create scheduler
    scheduler_config = checkpoint['scheduler_config']
    scheduler = VarianceScheduler(**scheduler_config)

    print(f"Model trained with T={scheduler.num_timesteps} timesteps")
    print(f"DDIM sampling with {num_steps} steps")

    # Generate samples
    final_samples, snapshots, timesteps = ddim_sample(
        model, scheduler, num_samples, num_steps, eta, device, seed
    )

    # Plot snapshots
    print("\nCreating visualization...")
    n_steps = len(snapshots)
    fig, axes = plt.subplots(num_samples, n_steps, figsize=(n_steps * 2, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    title = f'DDIM Denoising Process (eta={eta}): Pure Noise → Clean Image'
    if eta == 0:
        title += ' [Deterministic]'
    fig.suptitle(title, fontsize=14)

    for i in range(num_samples):
        for j in range(n_steps):
            axes[i, j].imshow(snapshots[j][i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f't={timesteps[j]}', fontsize=10)

    plt.tight_layout()

    # Save
    import os
    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_ddim_steps{num_steps}_eta{eta}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.show()

    return final_samples, snapshots, timesteps


def compare_ddim_speeds(model_path, num_samples=4, device='cuda', seed=None):
    """
    Compare DDIM sampling with different numbers of steps.

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

    # Create scheduler
    scheduler_config = checkpoint['scheduler_config']
    scheduler = VarianceScheduler(**scheduler_config)

    print(f"Model trained with T={scheduler.num_timesteps} timesteps\n")

    # Test different step counts
    step_counts = [10, 25, 50, scheduler.num_timesteps]
    all_samples = []

    import time

    for num_steps in step_counts:
        print(f"\nTesting DDIM with {num_steps} steps...")
        start_time = time.time()

        samples, _, _ = ddim_sample(
            model, scheduler, num_samples, num_steps, eta=0.0, device=device, seed=seed
        )

        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f}s ({elapsed/num_samples:.2f}s per sample)")

        all_samples.append(samples.cpu().numpy())

    # Visualize comparison
    print("\nCreating comparison visualization...")
    fig, axes = plt.subplots(num_samples, len(step_counts), figsize=(len(step_counts) * 3, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('DDIM: Quality vs Speed Tradeoff', fontsize=14)

    for i in range(num_samples):
        for j, num_steps in enumerate(step_counts):
            img = (all_samples[j][i, 0] + 1) / 2
            img = np.clip(img, 0, 1)
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'{num_steps} steps', fontsize=12)

    plt.tight_layout()

    # Save
    import os
    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_ddim_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")

    plt.show()


if __name__ == "__main__":
    import sys

    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "../models/ddpm_final.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Visualize DDIM denoising using ALL timesteps (deterministic, eta=0)
    print("="*60)
    print("DDIM Deterministic Sampling (eta=0, ALL timesteps)")
    print("="*60)
    samples, snapshots, timesteps = visualize_ddim_denoising(
        model_path, num_samples=4, num_steps=None, eta=0.0, device=device, seed=42
    )

    print(f"\nGenerated {len(snapshots)} snapshots at timesteps: {timesteps}")

    print("\nDone! DDIM with eta=0 and all timesteps is deterministic.")
