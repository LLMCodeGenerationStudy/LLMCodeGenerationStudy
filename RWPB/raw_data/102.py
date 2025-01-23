import math
import numpy as np
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN



def rotate_about_z_axis(vector, theta):
    """
    Rotate a 3D vector about the z-axis by a specified angle.
    Args:
        - vector: numpy array of shape (3,), the 3D vector to be rotated
        - theta: float, angle of rotation in radians
    Returns:
        rotated_vector: numpy array of shape (3,), the rotated 3D vector
    """
    # ----

    return np.dot(np.array([
[np.cos(theta), -np.sin(theta), 0],
[np.sin(theta), np.cos(theta), 0],
[0, 0, 1]
]), vector)

# unit test cases
vector = np.array([1, 0, 0])
theta = 0
print(rotate_about_z_axis(vector, theta))

vector = np.array([1, 0, 0])
theta = np.pi / 2
print(rotate_about_z_axis(vector, theta))

vector = np.array([1, 2, 3])
theta = np.pi / 4
print(rotate_about_z_axis(vector, theta))

