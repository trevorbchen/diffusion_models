"""
Dataset for NCSN training.

Returns clean images with randomly sampled noise level indices.
Actual noise is added during training for flexibility.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np


class NCSNDataset(Dataset):
    """
    Dataset for NCSN that samples a random noise level per image.

    Args:
        split: 'train' or 'test'
        num_scales: Number of noise levels (must match scheduler)
        augment: Whether to apply random horizontal flips
    """

    def __init__(
        self,
        split: str = 'train',
        num_scales: int = 10,
        augment: bool = True
    ):
        print(f"Loading Fashion MNIST {split} dataset...")
        self.dataset = load_dataset("fashion_mnist")[split]
        self.num_scales = num_scales
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - 'image': Clean image tensor (1, 28, 28) in [-1, 1]
                - 'level_idx': Random noise level index (0 to num_scales-1)
                                with importance sampling favoring low-sigma levels
        """
        # Get image
        image = self.dataset[idx]["image"]

        # Convert to tensor
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = (image_array - 0.5) / 0.5  # [-1, 1]
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)

        # Random horizontal flip
        if self.augment and torch.rand(1).item() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])

        # Importance sampling: oversample low-sigma levels (second half)
        # 50% of time sample from low-sigma levels (harder to learn)
        # 50% of time sample uniformly (coverage of all levels)
        if torch.rand(1).item() < 0.5:
            # Sample from second half (lower sigma levels)
            level_idx = torch.randint(self.num_scales // 2, self.num_scales, (1,)).item()
        else:
            # Sample uniformly across all levels
            level_idx = torch.randint(0, self.num_scales, (1,)).item()

        return {
            'image': image_tensor,
            'level_idx': level_idx
        }


def create_ncsn_dataloader(
    split: str = 'train',
    batch_size: int = 128,
    num_scales: int = 10,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
):
    """Create DataLoader for NCSN training."""
    dataset = NCSNDataset(split=split, num_scales=num_scales, augment=augment)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


