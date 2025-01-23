import math
import numpy as np
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN


def remove_outliers(arr, num_std_devs=2):
    """
    Remove outliers from an array based on a specified number of standard deviations from the mean.
    Args:
        - arr: numpy array, the input array
        - num_std_devs: int, number of standard deviations from the mean to consider as outliers (default is 2)
    Returns:
        filtered_arr: numpy array, the array with outliers removed
    """
    # ----
    
    if len(arr) < 2:
        return arr
    mean = np.mean(arr)
    std_dev = np.std(arr)
    return arr[np.abs(arr - mean) <= num_std_devs * std_dev]


# unit test cases
print(remove_outliers(np.random.normal(50, 10, size=100)))
print(remove_outliers(np.linspace(0, 100, 100)))
print(remove_outliers(np.append(np.random.normal(50, 10, size=95), [150, 200, -50, -100])))