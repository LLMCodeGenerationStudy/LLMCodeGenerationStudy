import numpy as np

def moving_average(arr, window_size):
    """
    Calculates the moving average of an array with a specified window size and returns both
    the smoothed values and the residuals.

    Arguments:
    arr : numpy.ndarray
        The input array over which the moving average is computed.
    window_size : int
        The number of elements in the moving average window.

    Returns:
    list of numpy.ndarray
        The first element is an array of the moving averages, and the second element is an array of residuals.
    """
    # ----

    # Pad the input array with zeros at the beginning to handle the boundary effect
    arr_padded = np.pad(arr, (window_size - 1, 0), "constant")

    # Calculate the moving average using a convolution with a uniform kernel (i.e., np.ones(window_size))
    smoothed_arr = np.convolve(arr_padded, np.ones(window_size), "valid") / window_size

    # Calculate residuals by subtracting the smoothed array from the original array
    residuals = arr - smoothed_arr

    return [smoothed_arr, residuals]


# unit test cases
print(moving_average(np.array([1, 2, 3, 4, 5, 6]), 3))
print(moving_average(np.array([1, 2, 3, 4, 5, 6]), 6))
print(moving_average(np.array([1, 2, 3]), 5))