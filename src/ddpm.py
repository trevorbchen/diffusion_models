"""
DDPM (Denoising Diffusion Probabilistic Models) training script.
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


class DDPM:
    """
    DDPM training and sampling.

    Args:
        model: UNet model
        scheduler: Variance scheduler
        device: Device to use
    """

    def __init__(self, model, scheduler, device='cuda'):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.device = device

    def train_step(self, batch, optimizer):
        """
        Single training step.

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

        # Predict noise
        predicted_noise = self.model(noisy_images, timesteps)

        # MSE loss on noise prediction
        loss = nn.functional.mse_loss(predicted_noise, noise)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent explosion (use higher threshold)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples, image_size=(1, 28, 28)):
        """
        Generate samples using DDPM sampling.

        Args:
            num_samples: Number of samples to generate
            image_size: Size of images (C, H, W)

        Returns:
            Generated images tensor (num_samples, C, H, W)
        """
        self.model.eval()

        # Start from pure noise
        x = torch.randn(num_samples, *image_size).to(self.device)

        # Reverse diffusion process
        timesteps = list(range(self.scheduler.num_timesteps))[::-1]

        for t in tqdm(timesteps, desc="Sampling"):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=self.device)

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # Get alpha values (convert to tensor)
            alpha_t = torch.tensor(self.scheduler.alphas[t], device=self.device)
            alpha_bar_t = torch.tensor(self.scheduler.alphas_cumprod[t], device=self.device)

            # Compute posterior mean
            if t > 0:
                alpha_bar_t_prev = torch.tensor(self.scheduler.alphas_cumprod[t - 1], device=self.device)
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=self.device)

            # Predicted x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            # Compute mean
            beta_t = torch.tensor(self.scheduler.betas[t], device=self.device)
            mean = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t) * pred_x0
            mean += (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t) * x

            if t > 0:
                # Add noise
                noise = torch.randn_like(x)
                variance = torch.tensor(self.scheduler.posterior_variance[t], device=self.device)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean

        return x


def train_ddpm(
    epochs=6,
    batch_size=32,
    learning_rate=3e-4,
    num_timesteps=500,
    schedule_type='linear',
    device='cuda',
    save_dir='../models',
    wandb_project='ddpm-fashion-mnist'
):
    """
    Train DDPM model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (constant)
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
            'model_dim': 64,
            'dim_mults': (1, 2, 4)
        }
    )

    # Create variance scheduler
    # For 50 timesteps, need beta_end=0.35 to fully destroy signal
    # For 1000 timesteps, use beta_end=0.02 (standard)
    beta_end = 0.35 if num_timesteps <= 100 else 0.02

    scheduler = VarianceScheduler(
        num_timesteps=num_timesteps,
        beta_start=1e-4,
        beta_end=beta_end,
        schedule_type=schedule_type
    )

    # Create model
    # Fashion MNIST: 1 channel, 28x28
    # Model dimensions: 64 -> 64, 128, 256, 512
    model = UNet(
        in_channels=1,
        out_channels=1,
        dim=64,
        dim_mults=(1, 2, 4),
        dropout=0.1
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create DDPM
    ddpm = DDPM(model, scheduler, device)

    # Optimizer (Adam with constant learning rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataloaders (randomized)
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

    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = DiffusionEvaluator(train_loader, device=device)

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Training
        model.train()
        train_losses = []

        progress_bar = tqdm(train_loader, desc=f"Training")
        for batch in progress_bar:
            loss = ddpm.train_step(batch, optimizer)
            train_losses.append(loss)
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Testing
        model.eval()
        test_losses = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                noisy_images = batch['noisy_image'].to(device)
                timesteps = batch['timestep'].to(device)
                noise = batch['noise'].to(device)

                predicted_noise = model(noisy_images, timesteps)
                loss = nn.functional.mse_loss(predicted_noise, noise)
                test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)

        # Generate samples for evaluation (100 samples for FID)
        print("Generating samples for evaluation...")
        generated_samples = ddpm.sample(num_samples=100, image_size=(1, 28, 28))

        # Evaluate
        metrics = evaluator.evaluate(generated_samples, num_nn_samples=100)

        # Log metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {avg_test_loss:.4f}")
        print(f"  Mean NN Distance: {metrics['mean_nn_distance']:.4f}")
        print(f"  Overfit %: {metrics['overfit_percentage']:.2f}%")

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'mean_nn_distance': metrics['mean_nn_distance'],
            'min_nn_distance': metrics['min_nn_distance'],
            'max_nn_distance': metrics['max_nn_distance'],
            'overfit_percentage': metrics['overfit_percentage']
        })

        # Log sample images
        sample_grid = generated_samples[:16].cpu()  # First 16 samples
        wandb.log({
            'generated_samples': [wandb.Image(img) for img in sample_grid]
        })

    # Save final model
    print("\n" + "="*50)
    print("Training complete! Saving final model...")
    print("="*50)

    final_model_path = save_path / 'ddpm_final.pt'
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
        }
    }, final_model_path)

    print(f"Model saved to: {final_model_path}")

    # Finish wandb
    wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    train_ddpm(
        epochs=3,
        batch_size=32,
        learning_rate=1e-3,
        num_timesteps=50,
        schedule_type='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir='../models',
        wandb_project='ddpm-fashion-mnist'
    )
