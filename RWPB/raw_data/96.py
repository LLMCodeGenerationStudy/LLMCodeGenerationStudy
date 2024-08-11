import numpy as np

def movmean(arr, n_avg):
  """
  Compute the moving average of an array.
  Args:
      - arr: numpy.ndarray, the input array
      - n_avg: int, the number of elements to include in the moving average
  Returns:
      mov_avg: numpy.ndarray, the moving average of the input array

  If the windows size is bigger than the length of array, the code should return a empty array.
  """
  # ----
  
  sums = np.cumsum(arr, dtype=float)
  sums[n_avg:] -= sums[:-n_avg]
  return sums[n_avg - 1 :] / n_avg


# unit test cases
arr = [1, 2, 3, 4, 5]
n_avg = 3
print(movmean(arr, n_avg))

arr = [1, 2, 3, 4, 5]
n_avg = 5
print(movmean(arr, n_avg))

arr = [1, 2, 3]
n_avg = 4
print(movmean(arr, n_avg))