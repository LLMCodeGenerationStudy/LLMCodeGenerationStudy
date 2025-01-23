import numpy as np

def smooth_pose_seq(pose_seq, window_size=5):
    """
    Smooths a sequence of poses using a moving average filter.

    Args:
        pose_seq (np.array): Array of pose data, where each pose is expected to be an array of coordinates.
        window_size (int, optional): The number of frames to include in the moving average window. Default is 5.

    Returns:
        np.array: An array of smoothed poses with the same shape as the input.

    This function applies a simple moving average smoothing technique to a sequence of poses, which is often useful
    in motion analysis or animation to reduce noise or fluctuations between frames. The function handles edge cases by
    adjusting the window size at the start and end of the sequence, ensuring all poses are smoothed.
    """
    # ----

    # Initialize an array of zeros with the same shape as the input sequence to store the smoothed poses
    smoothed_pose_seq = np.zeros_like(pose_seq)

    # Iterate over each pose in the sequence
    for i in range(len(pose_seq)):
        # Calculate the start and end indices of the window, ensuring they are within the bounds of the sequence
        start = max(0, i - window_size // 2)
        end = min(len(pose_seq), i + window_size // 2 + 1)

        # Compute the mean of the poses within the window and assign it to the smoothed sequence
        smoothed_pose_seq[i] = np.mean(pose_seq[start:end], axis=0)

    return smoothed_pose_seq



# unit test cases
print(smooth_pose_seq(np.array([[1, 2], [2, 3], [3, 4]])))
print(smooth_pose_seq(np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]])))
print(smooth_pose_seq(np.array([[1e10, 1e10], [1e-10, 1e-10], [1.5, 2.5], [0.001, 0.002], [5000, 5000]])))