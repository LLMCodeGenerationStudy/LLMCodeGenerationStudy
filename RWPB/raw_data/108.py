import torch

def norm_grid(v_grid):
    """Normalize the grid values to the range [-1, 1].

    Args:
        v_grid (torch.Tensor): The input grid of shape (B, 2, H, W).

    Returns:
        torch.Tensor: The normalized grid of shape (B, H, W, 2).
    """
    # ----
    
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


# unit test cases
v_grid = torch.rand((2, 2, 4, 4))
print(norm_grid(v_grid))

v_grid = torch.tensor([[[[0, 1], [0, 1]], [[0, 0], [1, 1]]]])
print(norm_grid(v_grid))

v_grid = torch.rand((1, 2, 3, 5))
print(norm_grid(v_grid))