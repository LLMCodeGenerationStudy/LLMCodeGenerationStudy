import torch
import numpy as np

def rotate_y(a, device=None):
    """
    Creates a 4x4 homogeneous transformation matrix for rotation around the y-axis in 3D space.

    Parameters:
    a (float): Rotation angle in radians.
    device (torch.device, optional): The device on which to create the tensor (e.g., 'cpu' or 'cuda'). Default is None, which means the tensor will be created on the default device.

    Returns:
    torch.Tensor: A 4x4 tensor representing the rotation transformation matrix around the y-axis.

    Process:
    1. Calculate the sine and cosine of the rotation angle.
    2. Construct a 4x4 matrix where the rotation is applied to the x and z coordinates, leaving the y coordinate unchanged.
    """
    # ----
    
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

# unit test cases
a = torch.pi / 4
print(rotate_y(a))

a = 0
print(rotate_y(a))

a = 2 * torch.pi
print(rotate_y(a))