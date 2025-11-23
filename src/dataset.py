"""
Dataset and DataLoader for diffusion model training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import torchvision.transforms as transforms
from variance_scheduler import VarianceScheduler


class DiffusionDataset(Dataset):
    """
    Dataset for diffusion model training.

    Applies random horizontal flips and forward diffusion.

    Args:
        split: 'train' or 'test'
        scheduler: VarianceScheduler instance
        augment: Whether to apply random horizontal flips
    """

    def __init__(
        self,
        split: str = 'train',
        scheduler: VarianceScheduler = None,
        augment: bool = True
    ):
        print(f"Loading Fashion MNIST {split} dataset...")
        self.dataset = load_dataset("fashion_mnist")[split]

        if scheduler is None:
            self.scheduler = VarianceScheduler(num_timesteps=1000, schedule_type='linear')
        else:
            self.scheduler = scheduler

        self.augment = augment
        self.num_timesteps = self.scheduler.num_timesteps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            noisy_image: Image with noise added at timestep t
            timestep: The timestep (0 to T-1)
            noise: The noise that was added
            original: The original image
        """
        # Get image (no label needed)
        image = self.dataset[idx]["image"]

        # Convert to tensor: (1, 28, 28)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = (image_array - 0.5) / 0.5  # Normalize to [-1, 1]
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        # Random horizontal flip for augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])

        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (1,)).item()
        t_tensor = torch.tensor([t], dtype=torch.long)

        # Apply forward diffusion
        noisy_image, noise = self.scheduler.forward_diffusion(
            image_tensor.unsqueeze(0),  # Add batch dim
            t_tensor
        )

        # Remove batch dimension
        noisy_image = noisy_image.squeeze(0)

        return {
            'noisy_image': noisy_image,
            'timestep': t,
            'noise': noise.squeeze(0),
            'original': image_tensor
        }


def create_dataloader(
    split: str = 'train',
    batch_size: int = 32,
    scheduler: VarianceScheduler = None,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
):
    """
    Create a DataLoader for diffusion training.

    Args:
        split: 'train' or 'test'
        batch_size: Batch size
        scheduler: VarianceScheduler instance
        augment: Whether to apply random horizontal flips
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading

    Returns:
        DataLoader instance
    """
    dataset = DiffusionDataset(split=split, scheduler=scheduler, augment=augment)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataset and dataloader
    print("Creating dataloader...")

    # Linear scheduler
    linear_scheduler = VarianceScheduler(num_timesteps=1000, schedule_type='linear')

    # Create train dataloader
    train_loader = create_dataloader(
        split='train',
        batch_size=8,
        scheduler=linear_scheduler,
        augment=True,
        shuffle=True
    )

    # Test a batch
    print("\nTesting dataloader...")
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Noisy images shape: {batch['noisy_image'].shape}")
    print(f"Timesteps shape: {batch['timestep'].shape}")
    print(f"Timesteps: {batch['timestep']}")
    print(f"Noise shape: {batch['noise'].shape}")
    print(f"Original shape: {batch['original'].shape}")

    print("\nDataLoader created successfully!")
    print(f"Total batches in train: {len(train_loader)}")
    print(f"Total samples: {len(train_loader.dataset)}")
