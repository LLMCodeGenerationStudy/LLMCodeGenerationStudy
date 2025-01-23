import torch

def scale(s, device=None):
    """
    Creates a 4x4 homogeneous transformation matrix for uniform scaling in 3D space.

    Parameters:
    s (float): Scaling factor applied equally to the x, y, and z axes.
    device (torch.device, optional): The device on which to create the tensor (e.g., 'cpu' or 'cuda'). Default is None, which means the tensor will be created on the default device.

    Returns:
    torch.Tensor: A 4x4 tensor representing the scaling transformation matrix.

    Process:
    1. Construct a 4x4 identity matrix.
    2. Replace the diagonal elements with the scaling factor for the x, y, and z coordinates.
    """
    # ----
    
    return torch.tensor([[ s, 0, 0, 0], 
                         [ 0, s, 0, 0], 
                         [ 0, 0, s, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)


# unit test cases
a = torch.pi / 4
print(scale(a))

a = 0
print(scale(a))

a = 2 * torch.pi
print(scale(a))
