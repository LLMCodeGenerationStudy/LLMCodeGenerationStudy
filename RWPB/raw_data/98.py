import numpy as np


def exp_smooth(vals, alpha):
  """
  Exponential smoothing function.
  Smooths a sequence of values using exponential smoothing.
  If input array is an empty, the code should return KeyError
  Args:
      - vals: list or numpy.ndarray, input sequence of values
      - alpha: float, smoothing factor
  Returns:
      smoothed: numpy.ndarray, smoothed sequence of values
  """
  # ----
  
  if len(vals) == 0:
    return KeyError
  smooth = np.array(vals[0], dtype=float)
  smoothed = [smooth.copy()]

  for i in range(1, len(vals)):
    smooth += -alpha * (smooth - vals[i])
    smoothed.append(smooth.copy())

  return np.array(smoothed)


# unit test cases
vals = [1, 2, 3, 4, 5]
alpha = 0.5
print(exp_smooth(vals, alpha))

vals = [67, 576, 12, 4, 5, 34, 123, 67, 32, 34]
alpha = 0.9
print(exp_smooth(vals, alpha))

vals = [10, 10, 10, 10, 10]
alpha = 0.2
print(exp_smooth(vals, alpha))

vals = []
alpha = 0.3
print(exp_smooth(vals, alpha))