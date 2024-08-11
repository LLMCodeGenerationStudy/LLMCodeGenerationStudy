import numpy as np


def mse_to_psnr(mse):
  """
  Compute the Peak Signal-to-Noise Ratio (PSNR) from the Mean Squared Error (MSE).
  If mse is equal zero, the code need to return ValueError

  Parameters:
  mse (float): Mean Squared Error value. This value should be a non-negative float, 
               representing the average squared difference between the reference 
               and the distorted image.

  Returns:
  float: The PSNR value in decibels (dB).

  Process:
  - The function assumes that the maximum possible pixel value (I_max) in the image 
    is 1.
  - The formula to convert MSE to PSNR is given by:
      PSNR = -10 * log10(MSE)
  - Use np.log to calculate.
  """
  # ----
  
  if mse == 0:
    return ValueError
  
  return -10. / np.log(10.) * np.log(mse)


# unit test cases
print(mse_to_psnr(0.01))
print(mse_to_psnr(1e-10))
print(mse_to_psnr(0))
print(mse_to_psnr(1))
print(mse_to_psnr(0.2312))