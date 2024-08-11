import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def euler_and_translation_to_matrix(euler_angles, translation_vector):
    """
    Converts Euler angles and a translation vector into a 4x4 transformation matrix.

    Args:
        euler_angles (list or np.array): Euler angles in degrees for rotation around the x, y, and z axes.
        translation_vector (list or np.array): Translation vector for x, y, and z coordinates.

    Returns:
        np.array: A 4x4 transformation matrix combining both rotation and translation.

    This function constructs a homogeneous transformation matrix that combines both rotation and translation.
    The rotation is specified by Euler angles (roll, pitch, yaw), which are converted into a rotation matrix.
    The translation is represented as a vector. This transformation matrix is useful in applications such as
    robotics, 3D graphics, and computer vision, where transformations in 3D space are common.
    """
    # ----

    # Create a rotation object from the Euler angles
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    # Convert the rotation object to a 3x3 rotation matrix
    rotation_matrix = rotation.as_matrix()

    # Initialize a 4x4 identity matrix
    matrix = np.eye(4)
    # Insert the rotation matrix into the top-left 3x3 part of the identity matrix
    matrix[:3, :3] = rotation_matrix
    # Insert the translation vector into the first three elements of the fourth column
    matrix[:3, 3] = translation_vector

    return matrix



# unit test cases
print(euler_and_translation_to_matrix(np.array([45, 45, 45]), np.array([10, 0, 5])))
print(euler_and_translation_to_matrix([0, 0, 0], [5, 5, 5]))
print(euler_and_translation_to_matrix([-90, 360, 270], [-1, -1, -1]))