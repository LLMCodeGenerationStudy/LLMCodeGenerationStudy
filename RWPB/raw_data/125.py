import numpy as np

def resample_segments(segments, n=1000):
    """
    Resamples each segment in a list to have a specified number of equidistant points.

    Args:
        segments (list): A list of numpy arrays of shape (n, 2), where each array represents a segment with n points.
        n (int): The number of points to resample each segment to, defaults to 1000.

    Returns:
        list: A list of numpy arrays, each reshaped to (n, 2), representing the resampled segments.
    """
    # ----
    
    for i, s in enumerate(segments):
        # Ensure the segment is closed by appending the first point to the end
        s = np.concatenate((s, s[0:1, :]), axis=0)
        # Create an array of the desired number of points spaced between 0 and the last index of the segment
        x = np.linspace(0, len(s) - 1, n)
        # Create an array of existing indices
        xp = np.arange(len(s))
        # Interpolate new points for both x and y coordinates
        segments[i] = np.stack(
            [np.interp(x, xp, s[:, j]) for j in range(2)], axis=-1
        )  # Stack interpolated x and y coordinates
    return segments

# unit test cases
segment1 = np.array([[0, 0], [2.5, 2.5], [5, 5], [7.5, 7.5], [10, 10]])
segment2 = np.array([[0, 0], [1, 5], [2, 10], [3, 15], [4, 20], [5, 25]])
segments = [segment1, segment2]
resampled_segments = resample_segments(segments, n=100)
print(resampled_segments)

segment = np.array([[0, 0], [1, 1]])
segments = [segment]
resampled_segments = resample_segments(segments, n=3)
print(resampled_segments)

segment = np.array([[0, 0], [1, 1]])
segments = [segment]
resampled_segments = resample_segments(segments, n=1)
print(resampled_segments)

repeated_point_segment = np.array([[1, 1]] * 5)
circle = np.array([[np.cos(theta), np.sin(theta)] for theta in np.linspace(0, 2 * np.pi, 100, endpoint=False)])
segments = [repeated_point_segment, circle]
resampled_segments = resample_segments(segments, n=50)
print(resampled_segments)