"""
DSM (Denoising Score Matching) inference script - CORRECTED for pure DSM.

Uses simple Langevin dynamics at SINGLE FIXED noise level σ.
"""
import torch
import matplotlib.pyplot as plt
from unet import UNet
from dsm_scheduler import DSMScheduler  # Your scheduler - single fixed σ!
import numpy as np
from tqdm import tqdm


def dsm_sample(model, scheduler, num_samples=4, num_steps=1000, eta=0.1,
               device='cuda', seed=None):
    """
    Generate samples using Langevin dynamics at single fixed noise level.

    Pure DSM sampling (NO annealing, NO timesteps):
    1. Start from x_0 ~ N(0, σ²I)
    2. Run Langevin: x_{k+1} = x_k + η·∇ log p_σ(x_k) + √(2η)·z
    3. Return final x after K steps

    Args:
        model: Trained UNet model (score predictor)
        scheduler: DSMScheduler with fixed sigma
        num_samples: Number of samples to generate
        num_steps: Number of Langevin steps (1000-5000 for quality)
        eta: Step size (typically 0.01-0.1)
        device: Device to use
        seed: Random seed for reproducibility

    Returns:
        Generated images, snapshots, and step indices
    """
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    # Get the SINGLE fixed sigma from scheduler
    sigma = scheduler.sigma
    print(f"\nPure DSM Langevin sampling:")
    print(f"  Fixed noise level: σ = {sigma}")
    print(f"  Number of steps: {num_steps}")
    print(f"  Step size: η = {eta}")

    # Start from noise at level σ: x_0 ~ N(0, σ²I)
    x = torch.randn(num_samples, 1, 28, 28).to(device) * sigma

    # Dummy timestep (pure DSM doesn't use timesteps!)
    dummy_t = torch.zeros(num_samples, dtype=torch.long, device=device)

    # Save snapshots
    save_interval = max(1, num_steps // 10)
    snapshots = []
    snapshot_steps = []

    # Simple Langevin dynamics at fixed σ
    for step in range(num_steps):
        # Save snapshot
        if step % save_interval == 0 or step == num_steps - 1:
            snapshot = x.cpu().numpy()
            snapshot = (snapshot + 1) / 2  # [-1, 1] -> [0, 1]
            snapshot = np.clip(snapshot, 0, 1)
            snapshots.append(snapshot)
            snapshot_steps.append(step)
            if step % (save_interval * 2) == 0:
                print(f"  Step {step}/{num_steps}: Saved snapshot")

        with torch.no_grad():
            # Predict score: ∇_x log p_σ(x)
            predicted_score = model(x, dummy_t)

        # Langevin update: x_{k+1} = x_k + η·score + √(2η)·z
        x = x + eta * predicted_score

        # Add stochastic noise
        z = torch.randn_like(x)
        x = x + np.sqrt(2 * eta) * z

    # Final clipping
    x = torch.clamp(x, -1, 1)

    return x, snapshots, snapshot_steps


def visualize_dsm_sampling(model_path, num_samples=4, num_steps=1000, eta=0.1,
                           device='cuda', seed=None):
    """
    Visualize the pure DSM sampling process.

    Args:
        model_path: Path to saved model checkpoint
        num_samples: Number of samples to generate
        num_steps: Number of Langevin steps
        eta: Step size
        device: Device to use
        seed: Random seed
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Verify model type
    model_type = checkpoint.get('model_type', 'unknown')
    if model_type != 'DSM':
        print(f"Warning: Model type is '{model_type}', expected 'DSM'")

    # Load sigma from checkpoint
    sigma = checkpoint['scheduler_config']['sigma']
    print(f"Loaded σ = {sigma} from checkpoint")

    # Create model
    model_config = checkpoint['model_config']
    model = UNet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Create scheduler with the SAME sigma as training
    scheduler = DSMScheduler(sigma=sigma)

    print(f"\nPure DSM model with fixed σ = {sigma}")
    print(f"Sampling with {num_steps} Langevin steps")

    # Generate samples
    final_samples, snapshots, steps = dsm_sample(
        model, scheduler, num_samples, num_steps, eta, device, seed
    )

    # Plot snapshots
    print("\nCreating visualization...")
    n_steps = len(snapshots)
    fig, axes = plt.subplots(num_samples, n_steps, figsize=(n_steps * 2, num_samples * 2))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Pure DSM Langevin Sampling (σ={sigma}, η={eta})', fontsize=14)

    for i in range(num_samples):
        for j in range(n_steps):
            axes[i, j].imshow(snapshots[j][i, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'step={steps[j]}', fontsize=10)

    plt.tight_layout()

    # Save
    import os
    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_pure_dsm_steps{num_steps}_eta{eta}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    plt.show()

    return final_samples, snapshots, steps


def compare_step_sizes(model_path, num_samples=4, num_steps=1000, device='cuda', seed=None):
    """
    Compare different step sizes for Langevin dynamics.
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    sigma = checkpoint['scheduler_config']['sigma']

    model_config = checkpoint['model_config']
    model = UNet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    scheduler = DSMScheduler(sigma=sigma)

    print(f"\nTesting different step sizes with {num_steps} steps each:")

    # Test different step sizes
    step_sizes = [0.01, 0.05, 0.1, 0.2]
    all_samples = []

    import time

    for eta in step_sizes:
        print(f"\nTesting η = {eta}...")
        start_time = time.time()

        samples, _, _ = dsm_sample(
            model, scheduler, num_samples, num_steps, eta, device, seed
        )

        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f}s ({elapsed/num_samples:.2f}s per sample)")

        all_samples.append(samples.cpu().numpy())

    # Visualize comparison
    print("\nCreating comparison visualization...")
    fig, axes = plt.subplots(num_samples, len(step_sizes),
                            figsize=(len(step_sizes) * 3, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'Pure DSM: Step Size Comparison (σ={sigma}, {num_steps} steps)',
                 fontsize=14)

    for i in range(num_samples):
        for j, eta in enumerate(step_sizes):
            img = (all_samples[j][i, 0] + 1) / 2
            img = np.clip(img, 0, 1)
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f'η = {eta}', fontsize=12)

    plt.tight_layout()

    import os
    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_pure_dsm_step_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")

    plt.show()


def generate_grid(model_path, num_samples=16, num_steps=1000, eta=0.1,
                 device='cuda', seed=None):
    """
    Generate a grid of samples from the pure DSM model.
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    sigma = checkpoint['scheduler_config']['sigma']

    model_config = checkpoint['model_config']
    model = UNet(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    scheduler = DSMScheduler(sigma=sigma)

    print(f"Generating {num_samples} samples with {num_steps} steps...")

    # Generate samples
    samples, _, _ = dsm_sample(
        model, scheduler, num_samples, num_steps, eta, device, seed
    )

    # Convert to numpy and normalize
    samples = samples.cpu().numpy()
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    samples = np.clip(samples, 0, 1)

    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    fig.suptitle(f'Pure DSM Generated Samples (σ={sigma}, η={eta}, {num_steps} steps)',
                 fontsize=14)

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx, 0], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')

    plt.tight_layout()

    import os
    base_name = os.path.splitext(model_path)[0]
    output_path = f"{base_name}_pure_dsm_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved grid to {output_path}")

    plt.show()


if __name__ == "__main__":
    import sys

    # Get model path from command line or use default
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "../models/dsm_final.pt"  # Note: dsm_final.pt, not dsm_ebm_final.pt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("="*60)
    print("PURE DSM SAMPLING - CORRECTED")
    print("="*60)
    print("Single fixed noise level σ - simple Langevin dynamics")
    print("="*60 + "\n")

    # Visualize sampling process
    samples, snapshots, steps = visualize_dsm_sampling(
        model_path, num_samples=4, num_steps=1000, eta=0.1, device=device, seed=42
    )

    print(f"\nGenerated {len(snapshots)} snapshots at steps: {steps}")

    # Generate a grid of samples
    print("\n" + "="*60)
    print("Generating Sample Grid")
    print("="*60)
    generate_grid(model_path, num_samples=16, num_steps=1000, eta=0.1, device=device, seed=42)

    print("\nDone! Pure DSM uses simple Langevin dynamics at fixed σ.")
