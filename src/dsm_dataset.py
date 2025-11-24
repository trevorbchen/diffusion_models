"""
Dataset and DataLoader for DSM (Denoising Score Matching) training.

For DSM, we only need clean images - noise is added during training
with the DSMScheduler at a single fixed sigma level.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np


class DSMDataset(Dataset):
    """
    Simple dataset for DSM training that returns only clean images.

    No timesteps, no pre-computed noise - just clean images.
    Noise will be added during training using DSMScheduler.

    Args:
        split: 'train' or 'test'
        augment: Whether to apply random horizontal flips
    """

    def __init__(self, split: str = 'train', augment: bool = True):
        print(f"Loading Fashion MNIST {split} dataset...")
        self.dataset = load_dataset("fashion_mnist")[split]
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            Clean image tensor normalized to [-1, 1]
        """
        # Get image
        image = self.dataset[idx]["image"]

        # Convert to tensor: (1, 28, 28)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = (image_array - 0.5) / 0.5  # Normalize to [-1, 1]
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        # Random horizontal flip for augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])

        return image_tensor


def create_dsm_dataloader(
    split: str = 'train',
    batch_size: int = 128,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
):
    """
    Create a DataLoader for DSM training.

    Args:
        split: 'train' or 'test'
        batch_size: Batch size
        augment: Whether to apply random horizontal flips
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading

    Returns:
        DataLoader instance that yields clean image batches
    """
    dataset = DSMDataset(split=split, augment=augment)

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
    print("Creating DSM dataloader...")

    # Create train dataloader
    train_loader = create_dsm_dataloader(
        split='train',
        batch_size=8,
        augment=True,
        shuffle=True
    )

    # Test a batch
    print("\nTesting dataloader...")
    batch = next(iter(train_loader))

    print(f"Batch shape: {batch.shape}")
    print(f"Batch type: {type(batch)}")
    print(f"Value range: [{batch.min():.2f}, {batch.max():.2f}]")

    print("\nDataLoader created successfully!")
    print(f"Total batches in train: {len(train_loader)}")
    print(f"Total samples: {len(train_loader.dataset)}")
