"""
DSM (Denoising Score Matching) inference script.
Uses Langevin dynamics for sampling from the score-based model.
"""
import torch
import matplotlib.pyplot as plt
from unet import UNet
from variance_scheduler import VarianceScheduler
import numpy as np
from tqdm import tqdm


def dsm_sample(model, scheduler, num_samples=4, num_steps=None, device='cuda', seed=None):
    """
    Generate samples using annealed Langevin dynamics.

    For score-based models, we use annealed Langevin dynamics:
    x_{t-1} = x_t + epsilon_t * s_theta(x_t, t) + sqrt(2 * epsilon_t) * z

    where s_theta is the predicted score and z ~ N(0, I)

    Args:
        model: Trained UNet model (score predictor)
        scheduler: Variance scheduler
        num_samples: Number of samples to generate
        num_steps: Number of sampling steps (if None, uses all training timesteps)
        device: Device to use
        seed: Random seed for reproducibility (if None, no seed is set)

    Returns:
        Generated images, snapshots, and timesteps
    """
    model.eval()

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    # Start from pure noise
    x = torch.randn(num_samples, 1, 28, 28).to(device)

    # Create timestep sequence (reverse order for denoising)
    total_timesteps = scheduler.num_timesteps

    if num_steps is None:
        # Use ALL timesteps
        timesteps = list(range(total_timesteps))[::-1]
    else:
        # Use evenly spaced steps
        step_size = total_timesteps // num_steps
        timesteps = list(range(0, total_timesteps, step_size))[::-1]

        # Ensure we end at t=0
        if timesteps[-1] != 0:
            timesteps.append(0)

    # Save snapshots
    save_interval = max(1, len(timesteps) // 10)
    snapshots = []
    snapshot_timesteps = []

    print(f"\nDSM Langevin sampling with {len(timesteps)} steps:")
    print(f"Timesteps: {timesteps[:5]}...{timesteps[-5:]}")

    # Annealed Langevin dynamics
    for i, t in enumerate(timesteps):
        # Save snapshot
        if i % save_interval == 0 or t == 0:
            snapshot = x.cpu().numpy()
            snapshot = (snapshot + 1) / 2  # [-1, 1] -> [0, 1]
            snapshot = np.clip(snapshot, 0, 1)
            snapshots.append(snapshot)
            snapshot_timesteps.append(t)
            print(f"  Step {i}/{len(timesteps)}, t={t:3d}: Saved snapshot")

        # Create batch of timesteps
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)

        with torch.no_grad():
            # Predict score output from model
            model_output = model(x, t_batch)

        # Get noise level (sigma_t)
        alpha_bar_t = torch.tensor(scheduler.alphas_cumprod[t], device=device)
        sigma_t = torch.sqrt(1 - alpha_bar_t)

        # Model was trained with loss E[||σ·score + ε||²]
        # Model learns: σ·model_output ≈ -ε, so model_output ≈ -ε/σ
        # The model output IS the score (gradient), use it directly
        score = model_output

        # Annealed step size: η_t ∝ σ_t^2
        # At high noise levels (early), take larger steps
        # At low noise levels (late), take smaller steps
        base_epsilon = 0.8  # Larger base step size for more aggressive denoising
        epsilon_t = base_epsilon * (sigma_t ** 2)

        # Langevin update with properly scaled step
        x = x + epsilon_t * score

        if t > 0:
            # Add noise for stochastic sampling (except at last step)
            z = torch.randn_like(x)
            x = x + torch.sqrt(2 * epsilon_t) * z

        # No clipping during intermediate steps - let Langevin dynamics explore freely

    # Final clipping to valid range
    x = torch.clamp(x, -1, 1)

    return x, snapshots, snapshot_timesteps


def visualize_dsm_denoising(model_path, num_samples=4, num_steps=50, device='cuda', seed=None):
    """
    Visualize the DSM denoising process step by step using Langevin dynamics.

    Args:
        model_path: Path to saved model checkpoint
        num_samples: Number of samples to generate
        num_steps: Number of Langevin sampling steps
        device: Device to use
        seed: Random seed for reproducibility (if None, no seed is set)
    """
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Verify model type
    model_type = checkpoint.get('model_type', 'unknown')
    if model_type != 'DSM-EBM':
        print(f"Warning: Model type is '{model_type}', expected 'DSM-EBM'")

    # Create model
    model_config = checkpoint['model_config']
    model = UNet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Create scheduler
    scheduler_config = checkpoint['scheduler_config']
    scheduler = VarianceScheduler(**scheduler_config)

    print(f"Model trained with T={scheduler.num_timesteps} timesteps")
    print(f"DSM Langevin sampling with {num_steps} steps")

    # Generate samples
    final_samples, snapshots, timesteps = dsm_sample(
        model, scheduler, num_samples, num_steps, device, seed
    )

    # Plot snapshots
    print("\nCreating visualization...")
    n_steps = len(snapshots)
    fig, axes = plt.subplots(num_samples, n_steps, figsize=(n_steps * 2, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('DSM Langevin Denoising Process: Pure Noise → Clean Image', fontsize=14)

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
    output_path = f"{base_name}_dsm_langevin_steps{num_steps}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.show()

    return final_samples, snapshots, timesteps


def compare_dsm_speeds(model_path, num_samples=4, device='cuda', seed=None):
    """
    Compare DSM Langevin sampling with different numbers of steps.

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
        print(f"\nTesting DSM Langevin with {num_steps} steps...")
        start_time = time.time()

        samples, _, _ = dsm_sample(
            model, scheduler, num_samples, num_steps, device=device, seed=seed
        )

        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f}s ({elapsed/num_samples:.2f}s per sample)")

        all_samples.append(samples.cpu().numpy())

    # Visualize comparison
    print("\nCreating comparison visualization...")
    fig, axes = plt.subplots(num_samples, len(step_counts), figsize=(len(step_counts) * 3, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('DSM Langevin: Quality vs Speed Tradeoff', fontsize=14)

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
    output_path = f"{base_name}_dsm_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")

    plt.show()


def generate_grid(model_path, num_samples=16, num_steps=50, device='cuda', seed=None):
    """
    Generate a grid of samples from the DSM model.

    Args:
        model_path: Path to saved model checkpoint
        num_samples: Number of samples to generate (will be arranged in a grid)
        num_steps: Number of Langevin sampling steps
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

    print(f"Generating {num_samples} samples with {num_steps} steps...")

    # Generate samples
    samples, _, _ = dsm_sample(model, scheduler, num_samples, num_steps, device, seed)

    # Convert to numpy and normalize
    samples = samples.cpu().numpy()
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    samples = np.clip(samples, 0, 1)

    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    fig.suptitle(f'DSM Generated Samples ({num_steps} steps)', fontsize=14)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')

    plt.tight_layout()

    # Save
    import os
    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_dsm_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid to {output_path}")

    plt.show()


if __name__ == "__main__":
    import sys

    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "../models/dsm_ebm_final.pt"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Visualize DSM denoising using ALL timesteps
    print("="*60)
    print("DSM Langevin Dynamics Sampling (ALL timesteps)")
    print("="*60)
    samples, snapshots, timesteps = visualize_dsm_denoising(
        model_path, num_samples=4, num_steps=None, device=device, seed=42
    )

    print(f"\nGenerated {len(snapshots)} snapshots at timesteps: {timesteps}")

    # Generate a grid of samples
    print("\n" + "="*60)
    print("Generating Sample Grid")
    print("="*60)
    generate_grid(model_path, num_samples=16, num_steps=50, device=device, seed=42)

    print("\nDone! DSM uses stochastic Langevin dynamics for sampling.")
