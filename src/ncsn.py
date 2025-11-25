"""
NCSN (Noise Conditional Score Networks) training.

Key difference from DSM: Model is conditioned on noise level sigma.
Uses multi-scale noise levels and sigma-squared weighting.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from pathlib import Path
import numpy as np

from unet import UNet
from ncsn_scheduler import NCSNScheduler
from ncsn_dataset import create_ncsn_dataloader


class NCSN:
    """
    Noise Conditional Score Network.

    Args:
        model: UNet that takes (x, level_idx) and outputs score
        scheduler: NCSNScheduler with noise levels
        device: Device to use
    """

    def __init__(self, model, scheduler, device='cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device

    def ncsn_loss(self, clean_images, level_idx):
        """
        Compute NCSN denoising score matching loss.

        Loss formula (with sigma^2 weighting):
            L = E[sigma^2 * ||s_theta(x + sigma*epsilon, sigma) + epsilon/sigma||^2]

        Args:
            clean_images: Clean images (batch, C, H, W)
            level_idx: Noise level indices (batch,)

        Returns:
            Weighted loss value
        """
        # Add noise at the specified levels
        noisy_images, noise, sigma = self.scheduler.add_noise(clean_images, level_idx)

        # Model predicts score conditioned on noise level
        # We pass level_idx as the "timestep" - UNet already supports this!
        predicted_score = self.model(noisy_images, level_idx)

        # Target: -epsilon/sigma
        sigma_expanded = sigma.view(-1, 1, 1, 1)
        target = -noise / sigma_expanded

        # Per-sample loss (MSE)
        loss_per_sample = ((predicted_score - target) ** 2).mean(dim=(1, 2, 3))

        # Weight by sigma^2 to balance gradients across scales
        # (high sigma has larger scores, low sigma has smaller scores)
        weights = sigma ** 2

        # Weighted mean
        loss = (weights * loss_per_sample).mean()

        return loss

    def train_step(self, batch, optimizer):
        """Single training step."""
        self.model.train()
        optimizer.zero_grad()

        clean_images = batch['image'].to(self.device)
        level_idx = batch['level_idx'].to(self.device)

        loss = self.ncsn_loss(clean_images, level_idx)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples, image_size=(1, 28, 28),
               steps_per_level=100, step_size=None, temperature=1.0):
        """
        Annealed Langevin dynamics sampling.

        For each noise level (high to low):
            1. Run T steps of Langevin at that level
            2. Move to next (lower) noise level

        Args:
            num_samples: Number of samples
            image_size: (C, H, W)
            steps_per_level: Langevin steps at each noise level
            step_size: Fixed step size (if None, uses adaptive formula)
            temperature: Final noise temperature (1.0 = standard)

        Returns:
            Generated samples
        """
        self.model.eval()

        # Start from N(0, sigma_max^2)
        sigma_max = self.scheduler.sigma_max
        x = torch.randn(num_samples, *image_size).to(self.device) * sigma_max

        # Iterate through noise levels: high (0) to low (L-1)
        for level in range(self.scheduler.num_scales):
            sigma = self.scheduler.sigmas[level].item()

            # Step size: use fixed or adaptive
            # Fixed is more stable than sigma^2 scaling
            if step_size is None:
                # Simple fixed step size
                epsilon = 0.00002  # 2e-5, works well in practice
            else:
                epsilon = step_size

            level_idx = torch.full((num_samples,), level,
                                   dtype=torch.long, device=self.device)

            # Langevin dynamics at this noise level
            for step in range(steps_per_level):
                # Predict score
                score = self.model(x, level_idx)

                # Langevin update
                # Last step of last level: no noise added
                if level == self.scheduler.num_scales - 1 and step == steps_per_level - 1:
                    noise = 0
                else:
                    noise = torch.randn_like(x) * temperature

                x = x + epsilon * score + np.sqrt(2 * epsilon) * noise

        return torch.clamp(x, -1, 1)


def train_ncsn(
    epochs=10,
    batch_size=128,
    learning_rate=1e-3,
    num_scales=10,
    sigma_min=0.01,
    sigma_max=1.0,
    schedule_type='geometric',
    device='cuda',
    save_dir='../models',
    wandb_project='ncsn-fashion-mnist'
):
    """
    Train NCSN model.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Training NCSN with {num_scales} noise levels")
    print(f"Sigma range: [{sigma_min}, {sigma_max}]")
    print(f"Schedule type: {schedule_type}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        mode='offline',
        config={
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_scales': num_scales,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'schedule_type': schedule_type,
            'model_type': 'NCSN',
            'architecture': 'UNet',
            'model_dim': 64,
            'dim_mults': (1, 2, 4)
        }
    )

    # Create scheduler
    scheduler = NCSNScheduler(
        num_scales=num_scales,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        schedule_type=schedule_type
    )
    print(f"\nNoise levels (geometric sequence):")
    for i, sigma in enumerate(scheduler.sigmas_np):
        print(f"  Level {i}: sigma = {sigma:.4f}")

    # Create model (same UNet - it already handles conditioning!)
    model = UNet(
        in_channels=1,
        out_channels=1,
        dim=64,
        dim_mults=(1, 2, 4),
        dropout=0.1
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create NCSN trainer
    ncsn = NCSN(model, scheduler, device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dataloaders
    train_loader = create_ncsn_dataloader(
        split='train',
        batch_size=batch_size,
        num_scales=num_scales,
        augment=True,
        shuffle=True
    )
    test_loader = create_ncsn_dataloader(
        split='test',
        batch_size=batch_size,
        num_scales=num_scales,
        augment=False,
        shuffle=False
    )

    # LR scheduler (cosine annealing)
    total_steps = epochs * len(train_loader)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.01
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"LR scheduler: Cosine annealing from {learning_rate} to {learning_rate * 0.01}")

    # Training loop
    print("\n" + "="*50)
    print("Starting NCSN training...")
    print("="*50 + "\n")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Training
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc="Training (NCSN)")
        for batch in pbar:
            loss = ncsn.train_step(batch, optimizer)
            train_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            lr_scheduler.step()

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Testing
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing (NCSN)"):
                clean = batch['image'].to(device)
                level_idx = batch['level_idx'].to(device)
                loss = ncsn.ncsn_loss(clean, level_idx)
                test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)

        # Log
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'learning_rate': current_lr,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss
        })

        # Generate samples every epoch
        if (epoch + 1) % 1 == 0:
            print("Generating samples...")
            samples = ncsn.sample(num_samples=16, steps_per_level=100)
            wandb.log({
                'generated_samples': [wandb.Image(img.cpu()) for img in samples]
            })

    # Save model
    print("\n" + "="*50)
    print("Training complete! Saving model...")
    print("="*50)

    final_model_path = save_path / 'ncsn_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_config': {
            'num_scales': num_scales,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'schedule_type': schedule_type
        },
        'model_config': {
            'in_channels': 1,
            'out_channels': 1,
            'dim': 64,
            'dim_mults': (1, 2, 4),
            'dropout': 0.1
        },
        'model_type': 'NCSN'
    }, final_model_path)

    print(f"Model saved to: {final_model_path}")
    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    train_ncsn(
        epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        num_scales=10,
        sigma_min=0.01,
        sigma_max=1.0,
        schedule_type='cosine',  # 'geometric' or 'cosine'
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='../models',
        wandb_project='ncsn-fashion-mnist'
    )
