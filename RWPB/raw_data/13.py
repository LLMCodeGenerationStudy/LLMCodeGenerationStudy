import numpy as np


def remove_scale_from_camera_pose(pose_matrix):
    '''pose_matrix is a 4x4 numpy array

    Args:
        - pose_matrix: numpy.array
    Return:
        - pose_matrix: numpy.array
    '''
    # ----

    # Ensure the matrix is in floating point format to avoid type casting issues in division
    pose_matrix = pose_matrix.astype(np.float64)

    # For each of the first three columns, normalize to remove scale
    for i in range(3):
        column = pose_matrix[:, i]
        scale_factor = np.linalg.norm(column[:3])  # Compute the norm of the column, excluding the bottom element
        if scale_factor > 0:  # Avoid division by zero
            pose_matrix[:, i] /= scale_factor  # Normalize column to remove scale

    # The last column (translation) and the last row are not modified, as they do not contribute to scale
    return pose_matrix

# unit test cases
print(remove_scale_from_camera_pose(np.array([[3, 0, 0, 0],
          [0, 4, 0, 0],
          [0, 0, 2, 0],
          [0, 0, 0, 1]])
))
print(remove_scale_from_camera_pose(np.array([[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1]])
))
print(remove_scale_from_camera_pose(np.array([[1, 2, 3, 0],
          [4, 5, 6, 0],
          [7, 8, 9, 0],
          [0, 0, 0, 1]])
))