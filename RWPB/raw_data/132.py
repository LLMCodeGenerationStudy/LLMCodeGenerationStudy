import torch


def B_batch(x, grid, k=0, extend=True, device="cpu"):
    """
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    """
    # ----

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(
            x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device
        )
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:(-k)]
        ) * B_km1[
            :, 1:
        ]
    return value


# unit test cases
num_splines = 3
num_samples = 50
num_grid_points = 8
k = 2
x = torch.normal(0, 1, size=(num_splines, num_samples))
grids = torch.einsum('i,j->ij', torch.ones(num_splines,), torch.linspace(-1, 1, steps=num_grid_points))
result = B_batch(x, grids, k=k)
print(result)

num_splines = 2
num_samples = 30
num_grid_points = 5
k = 0
x = torch.linspace(-1, 1, steps=num_samples).repeat(num_splines, 1)
grids = torch.einsum('i,j->ij', torch.ones(num_splines,), torch.linspace(-1, 1, steps=num_grid_points))
result = B_batch(x, grids, k=k, extend=False)
print(result)


num_splines = 4
num_samples = 100
num_grid_points = 10
k = 4
x = torch.cat([
    torch.linspace(-1.5, -1, steps=num_samples // 2),
    torch.linspace(1, 1.5, steps=num_samples // 2)
]).repeat(num_splines, 1)
grids = torch.einsum('i,j->ij', torch.ones(num_splines,), torch.linspace(-1, 1, steps=num_grid_points))
result = B_batch(x, grids, k=k, extend=False)
print(result)