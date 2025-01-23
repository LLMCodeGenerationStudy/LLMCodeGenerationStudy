import torch

def mesh_grid(B, H, W):
    """Create a mesh grid.

    Args:
        B (int): Batch size.
        H (int): Height of the grid.
        W (int): Width of the grid.

    Returns:
        torch.Tensor: A tensor of shape (B, 2, H, W) containing the mesh grid coordinates.
    """
    # ----
    
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

# unit test cases
B=1
H=1
W=1
print(mesh_grid(B, H, W))

B=1
H=5
W=10
print(mesh_grid(B, H, W))

B=100
H=10
W=10
print(mesh_grid(B, H, W))

B=3
H=10
W=5
print(mesh_grid(B, H, W))
