import torch

def generate_grids(lower_limit, upper_limit, grid_size):
    """
    Generate a grid of points within the specified limits.

    Args:
        lower_limit (torch.Tensor): Lower boundary of the grid.
        upper_limit (torch.Tensor): Upper boundary of the grid.
        grid_size (torch.Tensor): Number of grid points per dimension.

    Returns:
        tuple: Two tensors, each containing the lower and upper bounds of each grid cell.
    """
    # ----
    
    ndim = lower_limit.size(0)
    assert ndim == upper_limit.size(0)
    assert lower_limit.ndim == upper_limit.ndim == 1
    grids = [None] * ndim
    steps = (upper_limit - lower_limit) / grid_size
    for d in range(ndim):
        grids[d] = torch.linspace(
            lower_limit[d], upper_limit[d], grid_size[d] + 1, device=lower_limit.device
        )[: grid_size[d]]
    lower = torch.cartesian_prod(*grids)
    upper = lower + steps
    return lower, upper

# unit test cases

lower_limit = torch.tensor([0.0])
upper_limit = torch.tensor([1.0])
grid_size = torch.tensor([2])
print(generate_grids(lower_limit, upper_limit, grid_size))

lower_limit = torch.tensor([0.0, 0.0])
upper_limit = torch.tensor([1.0, 1.0])
grid_size = torch.tensor([2, 2])
print(generate_grids(lower_limit, upper_limit, grid_size))

lower_limit = torch.tensor([-1.0, 0.0, 2.0])
upper_limit = torch.tensor([1.0, 1.0, 5.0])
grid_size = torch.tensor([2, 3, 1])
print(generate_grids(lower_limit, upper_limit, grid_size))
