"""
Load and prepare Fashion MNIST dataset for diffusion model training.
"""
import torch
from datasets import load_dataset
from pathlib import Path
import numpy as np


def prepare_fashion_mnist(save_dir: str = "../data", num_samples: int = 35):
    """
    Load Fashion MNIST and convert to tensor format.

    Args:
        save_dir: Directory to save tensor files
        num_samples: Number of samples to save for visualization
    """
    print("Loading Fashion MNIST dataset...")
    dataset = load_dataset("fashion_mnist")

    # Get training split
    train_dataset = dataset["train"]

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving {num_samples} samples to {save_dir}...")

    for i in range(num_samples):
        # Get image (PIL Image)
        image = train_dataset[i]["image"]
        label = train_dataset[i]["label"]

        # Convert to tensor: (1, 28, 28) for grayscale
        # Normalize to [-1, 1] range (standard for diffusion models)
        image_array = np.array(image, dtype=np.float32) / 255.0  # [0, 1]
        image_array = (image_array - 0.5) / 0.5  # [-1, 1]

        # Create tensor with channel dimension
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # (1, 28, 28)

        # Save tensor
        torch.save(image_tensor, save_path / f"sample_{i}_label_{label}.pt")

    print(f"Saved {num_samples} samples successfully!")

    # Also save a batch for convenience
    print("Creating batch tensor...")
    batch_images = []
    batch_labels = []

    for i in range(100):
        image = train_dataset[i]["image"]
        label = train_dataset[i]["label"]

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = (image_array - 0.5) / 0.5

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        batch_images.append(image_tensor)
        batch_labels.append(label)

    batch_tensor = torch.stack(batch_images)  # (100, 1, 28, 28)
    labels_tensor = torch.tensor(batch_labels)

    torch.save({"images": batch_tensor, "labels": labels_tensor}, save_path / "batch_100.pt")
    print("Saved batch of 100 images!")

    return train_dataset


if __name__ == "__main__":
    prepare_fashion_mnist()
