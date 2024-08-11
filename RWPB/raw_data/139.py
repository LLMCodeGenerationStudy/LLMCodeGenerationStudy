import torch
import numpy as np

def rotate_normalmap_by_angle_torch(normal_map, angle):
    """
    Rotates the normals in a normal map along the y-axis by a specified angle using PyTorch.

    Args:
        normal_map (torch.Tensor): A tensor of shape (H, W, 3) representing the normal map,
                                   where H is the height, W is the width, and each normal is a 3D vector with components in [-1, 1].
                                   This tensor should be on a CUDA device for GPU acceleration.
        angle (float): The rotation angle in degrees.

    Returns:
        torch.Tensor: The rotated normal map with the same shape as the input.
    """
    # ----
    
    # Convert the angle from degrees to radians and move it to the same device as the normal_map
    angle = torch.tensor(angle / 180 * np.pi).to(normal_map.device)

    # Define the rotation matrix for rotating around the y-axis
    R = torch.tensor([
        [torch.cos(angle), 0, torch.sin(angle)],  # X' = X*cos(theta) + Z*sin(theta)
        [0, 1, 0],                                # Y' = Y (no change)
        [-torch.sin(angle), 0, torch.cos(angle)]  # Z' = -X*sin(theta) + Z*cos(theta)
    ]).to(normal_map.device)

    # Apply the rotation matrix to each normal vector in the normal map
    # Reshape the normal map from (H, W, 3) to (H*W, 3) for matrix multiplication, then reshape back to (H, W, 3)
    return torch.matmul(normal_map.view(-1, 3), R.T).view(normal_map.shape)


# unit test cases
normal_map = torch.rand(10, 10, 3) * 2 - 1
angle = 0
print(rotate_normalmap_by_angle_torch(normal_map, angle))

normal_map = torch.tensor([[[0, 0, 1]]], dtype=torch.float32).repeat(5, 5, 1)
angle = 90
print(rotate_normalmap_by_angle_torch(normal_map, angle))

normal_map = torch.rand(50, 50, 3) * 2 - 1
angle = 360
print(rotate_normalmap_by_angle_torch(normal_map, angle))