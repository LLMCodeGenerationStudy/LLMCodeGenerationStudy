import torch
import numpy as np
from torch.nn import functional as F


def grid_distortion(images, strength=0.5):
    """
    Applies a grid-based distortion to a batch of images by perturbing the grid points.

    Arguments:
    images : torch.Tensor
        A tensor of shape [B, C, H, W] representing the batch of images.
    strength : float, optional
        The strength of the distortion, must be a value between 0 and 1. Default is 0.5.

    Returns:
    torch.Tensor
        A tensor of distorted images with the same shape as the input.
    """
    # ----

    B, C, H, W = images.shape

    # Determine the resolution of the grid
    num_steps = np.random.randint(8, 17)  # Random grid size between 8 and 16
    grid_steps = torch.linspace(-1, 1, num_steps)  # Steps from -1 to 1 for normalized coordinates

    grids = []  # To store grids for each image in the batch
    for b in range(B):
        # Generate distorted x-coordinates
        x_steps = torch.linspace(0, 1, num_steps)  # Equally spaced steps between 0 and 1
        # Add random perturbation scaled by the distortion strength
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1)
        x_steps = (x_steps * W).long()  # Scale steps to image width
        x_steps[0] = 0
        x_steps[-1] = W  # Ensure edges remain at the borders

        xs = [torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]) for i in
              range(num_steps - 1)]
        xs = torch.cat(xs, dim=0)  # Concatenate to form a single tensor

        # Generate distorted y-coordinates using similar steps as x-coordinates
        y_steps = torch.linspace(0, 1, num_steps)
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1)
        y_steps = (y_steps * H).long()
        y_steps[0] = 0
        y_steps[-1] = H

        ys = [torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]) for i in
              range(num_steps - 1)]
        ys = torch.cat(ys, dim=0)

        # Construct the distortion grid using meshgrid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        grid = torch.stack([grid_x, grid_y], dim=-1)  # Stack to form [H, W, 2] grid

        grids.append(grid)

    grids = torch.stack(grids, dim=0).to(images.device)  # Convert list to tensor and move to the same device as images

    # Apply grid sample to distort the images according to the generated grids
    images = F.grid_sample(images, grids, align_corners=False)

    return images

# unit test cases
print(grid_distortion(torch.rand([2, 3, 32, 32])))
print(grid_distortion(torch.rand([1, 1, 64, 64]), 0))
print(grid_distortion(torch.rand([1, 1, 64, 64]), 1))
print(grid_distortion(torch.rand([10, 3, 256, 256]), 0.3))