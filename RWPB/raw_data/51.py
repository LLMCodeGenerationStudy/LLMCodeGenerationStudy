import torch
import torch.nn.functional as F


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Arguments:
    img1, img2 : torch.Tensor
        The two images between which SSIM is to be computed. They should have the same dimensions.
    window : torch.Tensor
        The window tensor used for mean and variance calculation, typically a Gaussian window.
    window_size : int
        The size of the Gaussian window, used here for padding calculations.
    channel : int
        The number of channels in img1 and img2 (e.g., 3 for RGB, 1 for grayscale).
    size_average : bool, optional
        If True (default), returns the mean SSIM over all elements. If False, returns a tensor of SSIM values for each channel.

    Returns:
    torch.Tensor
        SSIM index as a float or as a tensor based on 'size_average' parameter.
    """
    # ----

    # Apply convolution to calculate the mean of img1 and img2 using the specified window
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # Compute the squares of means
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    # Compute the mean of products
    mu1_mu2 = mu1 * mu2

    # Calculate variance of img1 and img2, and the covariance between them
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants for stability (avoid division by zero)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute the SSIM index
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Return the average SSIM over all pixels if size_average is True, else return SSIM map
    if size_average:
        return ssim_map.mean()  # average over all channels and pixels
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # average over each channel separately


# unit test cases
def gaussian_window(channel, window_size):
    """ Creates a Gaussian window tensor for given channel and window size. """
    # Ensure the operation is done with tensors
    _1D_window = torch.exp(-torch.arange(window_size).float()**2 / (2 * window_size**2))
    _1D_window = _1D_window / _1D_window.sum()  # Normalize the window
    _2D_window = _1D_window[:, None] * _1D_window[None, :]  # Create 2D Gaussian window
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


img = torch.rand(1, 3, 256, 256)
print(_ssim(img, img.clone(), gaussian_window(3, 11), 11, 3))

img1 = torch.zeros(1, 1, 512, 512)  # Black image
img2 = torch.ones(1, 1, 512, 512)   # White image
window = gaussian_window(1, 25)
print(_ssim(img1, img2, window, 25, 1, False))

img1 = torch.rand(1, 1, 300, 300)
img2 = torch.rand(1, 1, 300, 300)
window = gaussian_window(1, 25)
print(_ssim(img1, img2, window, 11, 1))