"""
DSM (Denoising Score Matching) EBM training script.

True DSM uses a SINGLE FIXED noise level σ, not a schedule.

Training:
- Add noise: x̃ = x + σε where ε ~ N(0, I)
- Train model to predict score: ∇_x log p_σ(x)
- Loss: E[||score + ε/σ||²]

Sampling:
- Use Langevin dynamics at the single noise level
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from pathlib import Path

from unet import UNet
from dsm_scheduler import DSMScheduler
from evaluation import DiffusionEvaluator
from torchvision import datasets, transforms


class DSM:
    """
    Denoising Score Matching with single fixed noise level.

    Args:
        model: UNet model (predicts score/gradient)
        scheduler: DSMScheduler with fixed sigma
        device: Device to use
    """

    def __init__(self, model, scheduler, device='cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device
        self.sigma = scheduler.sigma

    def dsm_loss(self, noise, predicted_score):
        """
        True DSM loss with single fixed noise level σ.

        Formulation (from eq 3.3.4):
            L_DSM(φ; σ) = (1/2) E[||s_φ(x̃; σ) + ε/σ||²]

        where:
        - x̃ = x + σε (noisy data)
        - s_φ is the predicted score ∇_x log p_σ(x)
        - σ is the FIXED noise level
        - ε ~ N(0, I) is the added noise

        The optimal score satisfies: s_φ(x̃; σ) = -ε/σ

        Args:
            noise: Added noise ϵ ~ N(0, I)
            predicted_score: Model's predicted score s_φ(x̃; σ)

        Returns:
            DSM loss value (without 1/2 factor, doesn't affect optimization)
        """
        # DSM loss: MSE(predicted_score, -ε/σ) = E[||s_φ(x̃; σ) + ε/σ||²]
        target = -noise / self.sigma
        loss = nn.functional.mse_loss(predicted_score, target)

        return loss

    def train_step(self, batch, optimizer):
        """
        Single training step with DSM loss.

        Args:
            batch: Clean image tensor (batch_size, C, H, W)
            optimizer: Optimizer

        Returns:
            Loss value
        """
        self.model.train()
        optimizer.zero_grad()

        # Get clean images (simple tensor from DSM dataloader)
        clean_images = batch.to(self.device)

        # Add noise: x̃ = x + σε
        noisy_images, noise = self.scheduler.add_noise(clean_images)

        # DSM doesn't use timesteps - pass a dummy timestep (all zeros)
        batch_size = clean_images.shape[0]
        dummy_timestep = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Predict score (model output represents the score function)
        predicted_score = self.model(noisy_images, dummy_timestep)

        # Compute DSM loss
        loss = self.dsm_loss(noise, predicted_score)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples, image_size=(1, 28, 28), num_steps=1000, eta=0.1):
        """
        Generate samples using Langevin dynamics at single noise level.

        For DSM with fixed σ, we use simple Langevin dynamics:
        x_{k+1} = x_k + η·∇_x log p_σ(x_k) + √(2η)·z

        where z ~ N(0, I)

        Args:
            num_samples: Number of samples to generate
            image_size: Size of images (C, H, W)
            num_steps: Number of Langevin steps (more steps = better quality)
            eta: Step size for Langevin dynamics

        Returns:
            Generated images tensor (num_samples, C, H, W)
        """
        self.model.eval()

        # Start from data corrupted with noise at level σ
        x = torch.randn(num_samples, *image_size).to(self.device) * self.sigma

        # Dummy timestep (DSM doesn't use timesteps)
        dummy_timestep = torch.zeros(num_samples, dtype=torch.long, device=self.device)

        # Convert eta to tensor
        eta_tensor = torch.tensor(eta, device=self.device)

        # Langevin dynamics
        for step in tqdm(range(num_steps), desc="Langevin Sampling"):
            # Predict score ∇_x log p_σ(x)
            predicted_score = self.model(x, dummy_timestep)

            # Langevin update: x_{k+1} = x_k + η·score + √(2η)·z
            x = x + eta_tensor * predicted_score

            # Add noise for stochastic sampling
            z = torch.randn_like(x)
            x = x + torch.sqrt(2 * eta_tensor) * z

        # Final clipping to valid range
        x = torch.clamp(x, -1, 1)

        return x


def train_dsm(
    epochs=6,
    batch_size=32,
    learning_rate=3e-4,
    sigma=0.5,
    device='cuda',
    save_dir='../models',
    wandb_project='dsm-ebm-fashion-mnist'
):
    """
    Train DSM EBM model with single fixed noise level.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        sigma: Fixed noise level (typically 0.3-0.5 for images in [-1,1])
        device: Device to use
        save_dir: Directory to save models
        wandb_project: Wandb project name
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Training DSM with single fixed noise level")
    print(f"Training for {epochs} epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Noise level (sigma): {sigma}")

    # Initialize wandb (offline mode)
    wandb.init(
        project=wandb_project,
        mode='offline',
        config={
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'sigma': sigma,
            'architecture': 'UNet',
            'model_type': 'DSM',
            'loss': 'Denoising Score Matching',
            'model_dim': 64,
            'dim_mults': (1, 2, 4)
        }
    )

    # Create DSM scheduler (single fixed noise level)
    scheduler = DSMScheduler(sigma=sigma)

    # Create model (same UNet architecture as DDPM)
    # Fashion MNIST: 1 channel, 28x28
    model = UNet(
        in_channels=1,
        out_channels=1,  # Output is score (same dim as input)
        dim=64,
        dim_mults=(1, 2, 4),
        dropout=0.1
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create DSM
    dsm = DSM(model, scheduler, device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create DSM dataloaders (clean images only, no wasted VP noise computation)
    from dsm_dataset import create_dsm_dataloader

    train_loader = create_dsm_dataloader(
        split='train',
        batch_size=batch_size,
        augment=True,
        shuffle=True,
        num_workers=0
    )

    test_loader = create_dsm_dataloader(
        split='test',
        batch_size=batch_size,
        augment=False,
        shuffle=False,
        num_workers=0
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Cosine annealing learning rate scheduler
    total_steps = epochs * len(train_loader)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.01  # End at 1% of initial LR
    )
    print(f"LR scheduler: Cosine annealing from {learning_rate} to {learning_rate * 0.01}")

    # Training loop
    print("\n" + "="*50)
    print("Starting DSM training...")
    print("="*50 + "\n")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Training
        model.train()
        train_losses = []

        progress_bar = tqdm(train_loader, desc=f"Training (DSM)")
        for batch in progress_bar:
            loss = dsm.train_step(batch, optimizer)
            train_losses.append(loss)
            progress_bar.set_postfix({'dsm_loss': f'{loss:.4f}'})

            # Step the learning rate scheduler after each batch
            lr_scheduler.step()

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Testing
        model.eval()
        test_losses = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing (DSM)"):
                clean_images = batch.to(device)

                # Add noise with DSM scheduler
                noisy_images, noise = dsm.scheduler.add_noise(clean_images)

                # Dummy timestep
                batch_size = clean_images.shape[0]
                dummy_timestep = torch.zeros(batch_size, dtype=torch.long, device=device)

                predicted_score = model(noisy_images, dummy_timestep)
                loss = dsm.dsm_loss(noise, predicted_score)
                test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)

        # Log metrics
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Train DSM Loss: {avg_train_loss:.4f}")
        print(f"  Test DSM Loss: {avg_test_loss:.4f}")

        wandb.log({
            'epoch': epoch + 1,
            'learning_rate': current_lr,
            'train_dsm_loss': avg_train_loss,
            'test_dsm_loss': avg_test_loss
        })

        # Generate and log sample images every epoch
        if (epoch + 1) % 1 == 0:
            print("Generating samples...")
            generated_samples = dsm.sample(num_samples=16, image_size=(1, 28, 28), num_steps=500)
            sample_grid = generated_samples.cpu()
            wandb.log({
                'generated_samples': [wandb.Image(img) for img in sample_grid]
            })

    # Save final model
    print("\n" + "="*50)
    print("Training complete! Saving final model...")
    print("="*50)

    final_model_path = save_path / 'dsm_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_config': {
            'sigma': sigma
        },
        'model_config': {
            'in_channels': 1,
            'out_channels': 1,
            'dim': 64,
            'dim_mults': (1, 2, 4),
            'dropout': 0.1
        },
        'model_type': 'DSM'
    }, final_model_path)

    print(f"Model saved to: {final_model_path}")

    # Finish wandb
    wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    train_dsm(
        epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        sigma=0.5,  # Fixed noise level for DSM
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='../models',
        wandb_project='dsm-fashion-mnist'
    )
