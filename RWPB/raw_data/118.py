import numpy as np

def filter_nan(ts, vs):
  """
  Filter NaN values from a time series.
  Args:
      - ts: numpy.ndarray, the time series
      - vs: numpy.ndarray, the values corresponding to the time series
  Returns:
      filtered_ts: numpy.ndarray, the time series with NaN values removed
      filtered_vs: numpy.ndarray, the corresponding values with NaN values removed
  """
  # ----
  
  good_vs = np.logical_not(np.isnan(vs))
  return ts[good_vs], vs[good_vs]


# unit test cases
ts = np.array([1, 2, 3, 4, 5])
vs = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
print(filter_nan(ts, vs))

ts = np.array([1, 2, 3, 4, 5])
vs = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
print(filter_nan(ts, vs))

ts = np.array([1, 2, 3, 4, 5])
vs = np.array([1.1, np.nan, 3.3, np.nan, 5.5])
print(filter_nan(ts, vs))
