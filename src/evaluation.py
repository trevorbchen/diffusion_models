"""
Evaluation metrics for diffusion models: Nearest Neighbor.
"""
import torch
import numpy as np
from tqdm import tqdm


def nearest_neighbor_distance(generated_images, train_images, k=1, metric='l2'):
    """
    Calculate nearest neighbor distances to detect overfitting.

    Args:
        generated_images: Generated images (N, C, H, W)
        train_images: Training images (M, C, H, W)
        k: Number of nearest neighbors
        metric: Distance metric ('l2' or 'l1')

    Returns:
        distances: Array of distances to k-nearest neighbors (N, k)
        indices: Indices of k-nearest neighbors (N, k)
    """
    n_generated = len(generated_images)
    n_train = len(train_images)

    # Flatten images
    gen_flat = generated_images.reshape(n_generated, -1)
    train_flat = train_images.reshape(n_train, -1)

    distances_list = []
    indices_list = []

    # Calculate distances in batches to save memory
    batch_size = 100

    for i in tqdm(range(0, n_generated, batch_size), desc="Computing nearest neighbors"):
        end_idx = min(i + batch_size, n_generated)
        gen_batch = gen_flat[i:end_idx]

        # Calculate distances to all training images
        if metric == 'l2':
            # Euclidean distance
            dists = torch.cdist(gen_batch, train_flat, p=2)
        elif metric == 'l1':
            # Manhattan distance
            dists = torch.cdist(gen_batch, train_flat, p=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Find k nearest neighbors
        topk_dists, topk_indices = torch.topk(dists, k, largest=False, dim=1)

        distances_list.append(topk_dists.cpu())
        indices_list.append(topk_indices.cpu())

    distances = torch.cat(distances_list, dim=0)
    indices = torch.cat(indices_list, dim=0)

    return distances.numpy(), indices.numpy()


class DiffusionEvaluator:
    """
    Evaluator for diffusion models.

    Calculates nearest neighbor metrics.
    """

    def __init__(self, train_dataloader, device='cuda'):
        """
        Args:
            train_dataloader: DataLoader for training data (for nearest neighbor)
            device: Device to use
        """
        self.device = device

        # Cache training images for nearest neighbor
        print("Caching training images...")
        self.train_images = []
        for batch in tqdm(train_dataloader, desc="Loading train images"):
            self.train_images.append(batch['original'])

        self.train_images = torch.cat(self.train_images, dim=0)
        print(f"Cached {len(self.train_images)} training images")

    def evaluate(self, generated_images, num_nn_samples=1000):
        """
        Evaluate generated images.

        Args:
            generated_images: Tensor of generated images (N, C, H, W)
            num_nn_samples: Number of samples to use for nearest neighbor

        Returns:
            Dictionary of metrics
        """
        print("\n=== Evaluation ===")

        # Calculate nearest neighbor distances
        print("Calculating nearest neighbor distances...")
        nn_samples = min(num_nn_samples, len(generated_images))
        sample_indices = torch.randperm(len(generated_images))[:nn_samples]
        sampled_generated = generated_images[sample_indices]

        distances, indices = nearest_neighbor_distance(
            sampled_generated.cpu(),
            self.train_images.cpu(),
            k=1,
            metric='l2'
        )

        mean_nn_dist = np.mean(distances)
        min_nn_dist = np.min(distances)
        max_nn_dist = np.max(distances)

        print(f"Nearest Neighbor Distance (mean): {mean_nn_dist:.4f}")
        print(f"Nearest Neighbor Distance (min): {min_nn_dist:.4f}")
        print(f"Nearest Neighbor Distance (max): {max_nn_dist:.4f}")

        # Check for potential overfitting (very small distances)
        threshold = 0.1  # Adjust based on your data
        overfit_count = np.sum(distances < threshold)
        overfit_percentage = 100 * overfit_count / len(distances)

        print(f"Potential overfitting: {overfit_count}/{len(distances)} ({overfit_percentage:.2f}%)")

        return {
            'mean_nn_distance': mean_nn_dist,
            'min_nn_distance': min_nn_dist,
            'max_nn_distance': max_nn_dist,
            'overfit_percentage': overfit_percentage
        }


if __name__ == "__main__":
    # Test evaluation
    from dataset import create_dataloader

    print("Creating test dataloader...")
    train_loader = create_dataloader(split='train', batch_size=64, shuffle=False, num_workers=0)

    print("Initializing evaluator...")
    evaluator = DiffusionEvaluator(train_loader, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Generate some fake samples (for testing)
    print("\nGenerating test samples...")
    fake_samples = torch.randn(100, 1, 28, 28) * 0.5

    # Evaluate
    metrics = evaluator.evaluate(fake_samples)

    print("\nEvaluation complete!")
    print(f"Metrics: {metrics}")