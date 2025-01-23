import math
import numpy as np
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN


def arccos2(vector, value):
    """
    Calculate the inverse cosine of a value with consideration of the vector direction.
    Args:
        - vector: numpy array of shape (2,), the 2D vector to consider for direction
        - value: float, the value for which to calculate the inverse cosine
    Returns:
        angle: float, the inverse cosine with consideration of vector direction
    """
    # ----
    
    if vector[0] == 0:
        if vector[1] >= 0:
            return 0
        else:
            return np.pi
    return -np.sign(vector[0]) * np.arccos(value)


# unit test cases
print(arccos2(np.array([-1, 0]), value = 0.5))
print(arccos2(np.array([1, 0]), value = 0.5))
print(arccos2(np.array([0, 1]), value = -1))
print(arccos2(np.array([0, -1]), value = -1))