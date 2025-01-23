import torch

def sample_from_3dgrid(grid: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """
    Samples features from a 3D grid based on 3D coordinates.

    Args:
        grid (torch.Tensor): 3D grid with shape (1, channels, H, W, D).
        coordinates (torch.Tensor): 3D coordinates to sample from with shape (batch_size, num_points_per_batch, 3).

    Returns:
        torch.Tensor: Sampled features with shape (batch_size, num_points_per_batch, feature_channels).
    """
    # ----
    
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(
        grid.expand(batch_size, -1, -1, -1, -1),
        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=False,
    )
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


# unit test cases
grid1 = torch.rand((1, 3, 4, 4, 4))  # 3D grid with 3 channels, shape (1, 3, 4, 4, 4)
coordinates1 = torch.tensor([[[1.0, 1.0, 1.0], [2.5, 2.5, 2.5]]])  # 3D coordinates, shape (1, 2, 3)
print(sample_from_3dgrid(grid1, coordinates1))

grid2 = torch.rand((1, 2, 5, 5, 5))  # 3D grid with 2 channels, shape (1, 2, 5, 5, 5)
coordinates2 = torch.tensor([[[6.0, 6.0, 6.0], [-1.0, -1.0, -1.0]]])  # Out of bounds coordinates, shape (1, 2, 3)
print(sample_from_3dgrid(grid2, coordinates2))

grid3 = torch.rand((1, 4, 6, 6, 6))  # 3D grid with 4 channels, shape (1, 4, 6, 6, 6)
coordinates3 = torch.tensor([
    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    [[0.5, 0.5, 0.5], [4.5, 4.5, 4.5], [5.5, 5.5, 5.5]]
])  # 3D coordinates, shape (2, 3, 3)
print(sample_from_3dgrid(grid3, coordinates3))