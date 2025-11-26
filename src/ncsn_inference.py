"""
NCSN Inference - Annealed Langevin Dynamics.

Progressively samples from high noise to low noise levels.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from unet import UNet
from ncsn_scheduler import NCSNScheduler


def annealed_langevin_dynamics(
    model,
    scheduler,
    num_samples=16,
    image_size=(1, 28, 28),
    steps_per_level=500,
    step_size=None,
    temperature=1.0,
    device='cuda',
    seed=None,
    save_snapshots=True
):
    """
    Generate samples using annealed Langevin dynamics.

    Args:
        model: Trained NCSN model
        scheduler: NCSNScheduler
        num_samples: Number of samples to generate
        image_size: (C, H, W)
        steps_per_level: Number of Langevin steps per noise level (500 default)
        step_size: Step size for Langevin dynamics (if None, uses adaptive scaling)
        temperature: Noise temperature (1.0 = standard)
        device: Device
        seed: Random seed
        save_snapshots: Whether to save intermediate states

    Returns:
        final_samples: Generated images
        snapshots: List of intermediate states (if save_snapshots=True)
        snapshot_info: List of (level, step, sigma) for each snapshot
    """
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    num_levels = scheduler.num_scales
    sigma_min = scheduler.sigma_min
    sigma_max = scheduler.sigma_max

    print(f"\nAnnealed Langevin Dynamics:")
    print(f"  {num_levels} noise levels: {sigma_max:.3f} -> {sigma_min:.3f}")
    print(f"  {steps_per_level} steps per level")
    print(f"  Total steps: {num_levels * steps_per_level}")

    # Initialize from N(0, sigma_max^2)
    x = torch.randn(num_samples, *image_size, device=device) * sigma_max

    snapshots = []
    snapshot_info = []

    # Iterate: high noise -> low noise
    for level in range(num_levels):
        sigma = scheduler.sigmas[level].item()

        # CRITICAL FIX: Adaptive step size that scales with sigma^2
        if step_size is None:
            epsilon = 0.1 * (sigma ** 2)  # Scale with sigma^2 - much more effective!
        else:
            epsilon = step_size

        level_idx = torch.full((num_samples,), level,
                               dtype=torch.long, device=device)

        # Save snapshot at start of level (every other level to reduce memory)
        if save_snapshots and level % 2 == 0:
            snap = x.detach().cpu().numpy()
            snap = (snap + 1) / 2
            snap = np.clip(snap, 0, 1)
            snapshots.append(snap)
            snapshot_info.append((level, 0, sigma))

        # Langevin dynamics at this noise level
        for step in range(steps_per_level):
            with torch.no_grad():
                score = model(x, level_idx)

            # Langevin update: x <- x + epsilon*score + sqrt(2*epsilon)*z
            # Last steps of last level: gradually reduce noise
            if level == num_levels - 1 and step >= steps_per_level - 10:
                # Gradually reduce noise in final 10 steps
                noise_scale = (steps_per_level - step) / 10.0
                noise = torch.randn_like(x) * temperature * noise_scale
            else:
                noise = torch.randn_like(x) * temperature

            x = x + epsilon * score + np.sqrt(2 * epsilon) * noise

        print(f"  Level {level+1}/{num_levels}: sigma={sigma:.4f}, epsilon={epsilon:.5f}")

    # Final snapshot
    if save_snapshots:
        snap = x.detach().cpu().numpy()
        snap = (snap + 1) / 2
        snap = np.clip(snap, 0, 1)
        snapshots.append(snap)
        snapshot_info.append((num_levels, steps_per_level, sigma_min))

    return torch.clamp(x, -1, 1), snapshots, snapshot_info


def visualize_annealed_sampling(
    model_path,
    num_samples=4,
    steps_per_level=500,
    device='cuda',
    seed=42
):
    """
    Visualize the annealed Langevin sampling process.
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Verify model type
    model_type = checkpoint.get('model_type', 'unknown')
    if model_type != 'NCSN':
        print(f"Warning: Expected 'NCSN', got '{model_type}'")

    # Load model
    model_config = checkpoint['model_config']
    model = UNet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load scheduler config
    sched_config = checkpoint['scheduler_config']
    scheduler = NCSNScheduler(**sched_config)

    print(f"Loaded NCSN with {scheduler.num_scales} noise levels")
    print(f"Sigma range: [{scheduler.sigma_min}, {scheduler.sigma_max}]")

    # Print noise schedule to verify direction
    print(f"\nNoise schedule verification:")
    for i in range(min(5, scheduler.num_scales)):
        print(f"  Level {i}: sigma = {scheduler.sigmas[i].item():.4f}")
    print(f"  ...")
    print(f"  Level {scheduler.num_scales-1}: sigma = {scheduler.sigmas[-1].item():.4f}")

    # Generate samples
    samples, snapshots, info = annealed_langevin_dynamics(
        model, scheduler,
        num_samples=num_samples,
        steps_per_level=steps_per_level,
        device=device,
        seed=seed
    )

    # Visualize
    n_snaps = len(snapshots)
    fig, axes = plt.subplots(num_samples, n_snaps,
                              figsize=(n_snaps * 2, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('NCSN Annealed Langevin: High Noise -> Low Noise', fontsize=14)

    for i in range(num_samples):
        for j in range(n_snaps):
            axes[i, j].imshow(snapshots[j][i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                level, step, sigma = info[j]
                axes[i, j].set_title(f'sigma={sigma:.3f}', fontsize=10)

    plt.tight_layout()

    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_annealed_sampling.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {output_path}")
    plt.show()

    return samples, snapshots


def generate_grid(
    model_path,
    num_samples=64,
    steps_per_level=500,
    device='cuda',
    seed=42
):
    """Generate a grid of NCSN samples."""
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    model = UNet(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    scheduler = NCSNScheduler(**checkpoint['scheduler_config'])

    print(f"Generating {num_samples} samples...")

    samples, _, _ = annealed_langevin_dynamics(
        model, scheduler,
        num_samples=num_samples,
        steps_per_level=steps_per_level,
        device=device,
        seed=seed,
        save_snapshots=False
    )

    # Create grid
    samples = samples.cpu().numpy()
    samples = (samples + 1) / 2
    samples = np.clip(samples, 0, 1)

    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    fig.suptitle('NCSN Generated Samples', fontsize=14)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                axes[i, j].imshow(samples[idx, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')

    plt.tight_layout()

    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid to {output_path}")
    plt.show()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "../models/ncsn_final.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}\n")

    print("="*60)
    print("NCSN ANNEALED LANGEVIN SAMPLING")
    print("="*60)

    # Visualize sampling process
    visualize_annealed_sampling(model_path, num_samples=4,
                                 steps_per_level=500, device=device)

    # Generate grid
    print("\n" + "="*60)
    print("Generating Sample Grid")
    print("="*60)
    generate_grid(model_path, num_samples=64,
                  steps_per_level=500, device=device)

    print("\nDone! NCSN uses annealed Langevin: high sigma -> low sigma.")
