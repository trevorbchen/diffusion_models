"""
DSM (Denoising Score Matching) EBM training script.

Unlike DDPM which predicts noise, DSM trains an energy-based model to estimate
the score function (gradient of log-likelihood) using denoising score matching.

The key difference is the loss function:
- DDPM: MSE loss between predicted noise and actual noise
- DSM: Denoising score matching loss that trains the model to estimate gradients
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
from dataset import create_dataloader
from variance_scheduler import VarianceScheduler
from evaluation import DiffusionEvaluator


class DSM:
    """
    Denoising Score Matching EBM for image generation.

    The model learns to estimate the score function (gradient of log p(x))
    through denoising score matching instead of predicting noise directly.

    Args:
        model: UNet model (predicts score/gradient)
        scheduler: Variance scheduler
        device: Device to use
    """

    def __init__(self, model, scheduler, device='cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device

    def dsm_loss(self, noise, timesteps, predicted_score):
        """
        Denoising Score Matching loss (simplified version).

        Formulation: L_DSM = E[||σ·s_φ(x_t; σ) + ϵ||²]

        where:
        - x_t = x + σϵ (noisy data)
        - s_φ is the predicted score
        - σ is the noise level at timestep t
        - ϵ is the added noise

        This formulation is stable across different noise levels.

        Args:
            noise: Added noise ϵ ~ N(0, I)
            timesteps: Timestep indices
            predicted_score: Model's predicted score s_φ(x_t; σ)

        Returns:
            DSM loss value
        """
        # Get noise levels (sigma_t) for each timestep
        # sigma_t^2 = 1 - alpha_bar_t (variance of noise at time t)
        batch_size = noise.shape[0]

        # Extract sigma values for the batch
        sigmas = []
        for t in timesteps:
            alpha_bar_t = self.scheduler.alphas_cumprod[t.item()]
            sigma_t = torch.sqrt(torch.tensor(1 - alpha_bar_t, device=self.device))
            sigmas.append(sigma_t)

        sigma_batch = torch.stack(sigmas).view(batch_size, 1, 1, 1)

        # Simple DSM loss: E[||σ·score + ε||²]
        # No division by σ²
        loss = torch.mean((predicted_score * sigma_batch + noise) ** 2)

        return loss

    def train_step(self, batch, optimizer):
        """
        Single training step with DSM loss.

        Args:
            batch: Dictionary with 'noisy_image', 'timestep', 'noise', 'original'
            optimizer: Optimizer

        Returns:
            Loss value
        """
        self.model.train()
        optimizer.zero_grad()

        # Get data
        noisy_images = batch['noisy_image'].to(self.device)
        timesteps = batch['timestep'].to(self.device)
        noise = batch['noise'].to(self.device)

        # Add small input noise for stability, especially at early timesteps
        # This prevents numerical issues when sigma is very small
        input_noise_scale = 0.05
        noisy_images = noisy_images + input_noise_scale * torch.randn_like(noisy_images)

        # Predict score (model output represents the score function)
        predicted_score = self.model(noisy_images, timesteps)

        # Compute DSM loss
        loss = self.dsm_loss(noise, timesteps, predicted_score)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples, image_size=(1, 28, 28), num_steps=None):
        """
        Generate samples using Langevin dynamics.

        For score-based models, we use annealed Langevin dynamics:
        x_{t-1} = x_t + epsilon_t * s_theta(x_t, t) + sqrt(2 * epsilon_t) * z

        where s_theta is the predicted score and z ~ N(0, I)

        Args:
            num_samples: Number of samples to generate
            image_size: Size of images (C, H, W)
            num_steps: Number of sampling steps (if None, uses all timesteps)

        Returns:
            Generated images tensor (num_samples, C, H, W)
        """
        self.model.eval()

        # Start from pure noise
        x = torch.randn(num_samples, *image_size).to(self.device)

        # Create timestep sequence
        total_timesteps = self.scheduler.num_timesteps
        if num_steps is None:
            timesteps = list(range(total_timesteps))[::-1]
        else:
            step_size = total_timesteps // num_steps
            timesteps = list(range(0, total_timesteps, step_size))[::-1]
            if timesteps[-1] != 0:
                timesteps.append(0)

        # Annealed Langevin dynamics
        for i, t in enumerate(tqdm(timesteps, desc="Sampling (Langevin)")):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=self.device)

            # Predict score s_φ(x_n; σ)
            predicted_score = self.model(x, t_batch)

            # Get noise level σ_t from the same scheduler used in training
            alpha_bar_t = torch.tensor(self.scheduler.alphas_cumprod[t], device=self.device)
            sigma_t = torch.sqrt(1 - alpha_bar_t)

            # Step size: Use constant step size for Langevin dynamics
            eta = torch.tensor(0.05, device=self.device)

            # Langevin dynamics: x_{n+1} = x_n + η·s_φ(x_n; σ) + √(2η)·ε_n
            x = x + eta * predicted_score

            if t > 0:
                # Add noise: √(2η)·ε_n where ε_n ~ N(0, I)
                z = torch.randn_like(x)
                x = x + torch.sqrt(2 * eta) * z

            # Clip to reasonable range
            x = torch.clamp(x, -1.5, 1.5)

        # Final clipping to valid range
        x = torch.clamp(x, -1, 1)

        return x


def train_dsm(
    epochs=6,
    batch_size=32,
    learning_rate=3e-4,
    num_timesteps=500,
    schedule_type='linear',
    device='cuda',
    save_dir='../models',
    wandb_project='dsm-ebm-fashion-mnist'
):
    """
    Train DSM EBM model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        num_timesteps: Number of diffusion timesteps
        schedule_type: 'linear' or 'cosine'
        device: Device to use
        save_dir: Directory to save models
        wandb_project: Wandb project name
    """
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Training DSM EBM with Denoising Score Matching")
    print(f"Training for {epochs} epochs")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Timesteps: {num_timesteps}")
    print(f"Schedule: {schedule_type}")

    # Initialize wandb (offline mode)
    wandb.init(
        project=wandb_project,
        mode='offline',
        config={
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_timesteps': num_timesteps,
            'schedule_type': schedule_type,
            'architecture': 'UNet',
            'model_type': 'DSM-EBM',
            'loss': 'Denoising Score Matching',
            'model_dim': 64,
            'dim_mults': (1, 2, 4)
        }
    )

    # Create variance scheduler
    beta_end = 0.35 if num_timesteps <= 100 else 0.02

    scheduler = VarianceScheduler(
        num_timesteps=num_timesteps,
        beta_start=1e-4,
        beta_end=beta_end,
        schedule_type=schedule_type
    )

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

    # Create dataloaders
    train_loader = create_dataloader(
        split='train',
        batch_size=batch_size,
        scheduler=scheduler,
        augment=True,
        shuffle=True,
        num_workers=0
    )

    test_loader = create_dataloader(
        split='test',
        batch_size=batch_size,
        scheduler=scheduler,
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

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = DiffusionEvaluator(train_loader, device=device)

    # Training loop
    print("\n" + "="*50)
    print("Starting DSM EBM training...")
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

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Testing
        model.eval()
        test_losses = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing (DSM)"):
                noisy_images = batch['noisy_image'].to(device)
                timesteps = batch['timestep'].to(device)
                noise = batch['noise'].to(device)

                predicted_score = model(noisy_images, timesteps)
                loss = dsm.dsm_loss(noise, timesteps, predicted_score)
                test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)

        # Generate samples for evaluation
        print("Generating samples for evaluation...")
        generated_samples = dsm.sample(num_samples=100, image_size=(1, 28, 28))

        # Evaluate
        metrics = evaluator.evaluate(generated_samples, num_nn_samples=100)

        # Log metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train DSM Loss: {avg_train_loss:.4f}")
        print(f"  Test DSM Loss: {avg_test_loss:.4f}")
        print(f"  Mean NN Distance: {metrics['mean_nn_distance']:.4f}")
        print(f"  Overfit %: {metrics['overfit_percentage']:.2f}%")

        wandb.log({
            'epoch': epoch + 1,
            'train_dsm_loss': avg_train_loss,
            'test_dsm_loss': avg_test_loss,
            'mean_nn_distance': metrics['mean_nn_distance'],
            'min_nn_distance': metrics['min_nn_distance'],
            'max_nn_distance': metrics['max_nn_distance'],
            'overfit_percentage': metrics['overfit_percentage']
        })

        # Log sample images
        sample_grid = generated_samples[:16].cpu()
        wandb.log({
            'generated_samples': [wandb.Image(img) for img in sample_grid]
        })

    # Save final model
    print("\n" + "="*50)
    print("Training complete! Saving final model...")
    print("="*50)

    final_model_path = save_path / 'dsm_ebm_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_config': {
            'num_timesteps': num_timesteps,
            'beta_start': 1e-4,
            'beta_end': beta_end,
            'schedule_type': schedule_type
        },
        'model_config': {
            'in_channels': 1,
            'out_channels': 1,
            'dim': 64,
            'dim_mults': (1, 2, 4),
            'dropout': 0.1
        },
        'model_type': 'DSM-EBM'
    }, final_model_path)

    print(f"Model saved to: {final_model_path}")

    # Finish wandb
    wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    train_dsm(
        epochs=3,
        batch_size=32,
        learning_rate=1e-3,
        num_timesteps=50,
        schedule_type='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='../models',
        wandb_project='dsm-ebm-fashion-mnist'
    )
