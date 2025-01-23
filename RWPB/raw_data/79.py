import numpy as np

def pr_from_grav(grav):
  """
  Compute pitch and roll angles from a matrix of gravity vectors in body frame.
  Args:
      - grav: numpy.ndarray, matrix of gravity vectors in body frame (shape: (n, 3))
  Returns:
      - pitches: numpy.ndarray, pitch angles (shape: (n,))
      - rolls: numpy.ndarray, roll angles (shape: (n,))
  """
  # ----
  
  pitches = np.arcsin(-grav[:, 0])
  rolls = np.arctan2(grav[:, 1], grav[:, 2])
  return pitches, rolls


# unit test cases

print(pr_from_grav(
  np.array([[0, 0, -1],  # No tilt (pitch=roll=0)
          [0, 0.7071, 0.7071],  # Tilted 45 degrees along Y-axis
          [0.5, 0, 0.866],  # Tilted 30 degrees along X-axis
          [0, -1, 0]])  # Upside down along Y-axis

))
print(pr_from_grav(
  np.array([[-1, 0, 0],    # Pitch -90 degrees (pointing directly downward)
          [1, 0, 0],     # Pitch 90 degrees (pointing directly upward)
          [0, 1, 0],     # Roll 90 degrees
          [0, 0, 1],     # No tilt, gravity perfectly aligned
          [0, -1, 1e-6]]) # Near vertical Y-axis, small Z to avoid zero

))
print(pr_from_grav(
  np.array([[0, 0, 0],  # Zero vector
          [0, 0, 0],  # Repeated zero vector to test consistent handling
          [0, 0, 0]])

))
print(pr_from_grav(
  np.array([[1e-12, 0, -1],   # Extremely small x component
          [0, 1e-12, -1],   # Extremely small y component
          [1e-12, 1e-12, -1]])  # Small x and y components

))
print(pr_from_grav(
  np.array([[1, 0, 0],   # Along X-axis
          [0, 1, 0],   # Along Y-axis
          [0, 0, 1],   # Along Z-axis
          [0, 0, -1],  # Along -Z-axis
          [-1, 0, 0],  # Along -X-axis
          [0, -1, 0]]) # Along -Y-axis

))