import math
import numpy as np


def r_sqrd(x, y):
    """
    Calculate R^2 of best fit simple least squares regression line.
    If denominator is zero, the code should return KeyError
    Args:
        - x: numpy array, shape (n,), x coordinates of (x,y) points.
        - y: numpy array, shape (n,), y coordinates of (x,y) points.

    Return:
        r_squared: float, R^2 of the best fit simple least squares regression line.

    """
    # ----
    
    x_dot_x = x.dot(x)
    x_dot_y = x.dot(y)
    x_mean = x.mean()
    x_sum = x.sum()
    y_mean = y.mean()
    denominator = x_dot_x - x_mean * x_sum
    if denominator == 0:
        return KeyError
    
    m = (x_dot_y - y_mean * x_sum) / denominator
    b = (y_mean * x_dot_x - x_mean * x_dot_y) / denominator
    y_pred = m * x + b
    residuals = y - y_pred
    tot = y - y_mean

    if int(tot.dot(tot)) == 0:
        return KeyError

    return 1.0 - residuals.dot(residuals) / tot.dot(tot)

    

# unit test cases
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
print(r_sqrd(x, y))

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 2, 2, 2, 2])
print(r_sqrd(x, y))

x = np.array([1, 1, 1, 1, 1])
y = np.array([1, 2, 3, 4, 5])
print(r_sqrd(x, y))

x = np.array([1, 1, 3, 1, 5])
y = np.array([1, 0, 3, 1, 5])
print(r_sqrd(x, y))

x = np.array([6, 1, 5, 7, 5])
y = np.array([0, -1, 3, 2, 3])
print(r_sqrd(x, y))