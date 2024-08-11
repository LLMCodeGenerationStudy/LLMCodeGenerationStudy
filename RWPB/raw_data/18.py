import numpy as np

def last_before(val, arr):
  """
  Find the index of the last element in a sorted array that is less than or equal to a given value.Requires sorted (monotonic) arr.
  Args:
      - val: int, the value to search for
      - arr: numpy.ndarray, the sorted array
  Returns:
      idx: int, the index of the last element in the array that is less than or equal to val
  """
  # ----

  assert np.all(np.diff(arr) >= 0)
  return np.searchsorted(arr, val) - 1


# unit test cases
print(last_before(val = 5, arr = np.array([1, 2, 3, 4, 5])))
print(last_before(val = 3.5, arr = np.array([1, 2, 3, 4, 5])))
print(last_before(val = 0, arr = np.array([1, 2, 3, 4, 5])))