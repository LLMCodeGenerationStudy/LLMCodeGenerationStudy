import numpy as np

def rotate_normalmap_by_angle(normal_map: np.ndarray, angle: float):
    """
    Rotates the normals in a normal map along the y-axis by a specified angle.

    Args:
        normal_map (np.ndarray): An array of shape (H, W, 3) representing the normal map,
                                 where H is the height, W is the width, and each normal is a 3D vector with components in [-1, 1].
        angle (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The rotated normal map with the same shape as the input.
    """
    # ----
    
    # Convert the angle from degrees to radians
    angle = angle / 180 * np.pi
    # Define the rotation matrix for rotating around the y-axis
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],  # X' = X*cos(theta) + Z*sin(theta)
        [0, 1, 0],                          # Y' = Y (no change)
        [-np.sin(angle), 0, np.cos(angle)]  # Z' = -X*sin(theta) + Z*cos(theta)
    ])
    # Apply the rotation matrix to each normal vector in the normal map
    # Reshape the normal map from (H, W, 3) to (H*W, 3) for matrix multiplication, then reshape back to (H, W, 3)
    return np.dot(normal_map.reshape(-1, 3), R.T).reshape(normal_map.shape)


# unit test cases
normal_map = np.array([[[0, 0, 1]]])
angle = 90
print(rotate_normalmap_by_angle(normal_map, angle))

normal_map = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
angle = 0
print(rotate_normalmap_by_angle(normal_map, angle))

normal_map = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
angle = 720  # Two full rotations
print(rotate_normalmap_by_angle(normal_map, angle))