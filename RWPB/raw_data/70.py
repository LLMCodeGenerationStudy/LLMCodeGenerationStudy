import numpy as np

def psnr_to_mse(psnr):
  """
  Compute the Mean Squared Error (MSE) given the Peak Signal-to-Noise Ratio (PSNR).

  Parameters:
  psnr (float): The Peak Signal-to-Noise Ratio, typically in decibels (dB).

  Returns:
  float: The computed Mean Squared Error (MSE).

  Process:
  - The function assumes the maximum pixel value is 1.
  - PSNR is converted to a linear scale using the formula:
    MSE = exp(-0.1 * ln(10) * PSNR)
  - This conversion is based on the logarithmic relationship between PSNR and MSE.
  - The constant -0.1 * ln(10) transforms the PSNR value into a scale where 
    the exponentiation gives the MSE.
  """
  # ----
  
  return np.exp(-0.1 * np.log(10.) * psnr)


# unit test cases
print(psnr_to_mse(1))
print(psnr_to_mse(0))
print(psnr_to_mse(0.2))
print(psnr_to_mse(0.31423))
print(psnr_to_mse(1657546223))