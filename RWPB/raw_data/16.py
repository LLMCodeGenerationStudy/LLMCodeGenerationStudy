import numpy as np

def his_match(src, dst):
    """
    Performs histogram matching between two color images src and dst.

    Parameters
    ----------
    src : numpy.ndarray
        Source image array of shape (H, W, 3) and type float64.
    dst : numpy.ndarray
        Destination image array of shape (H, W, 3) and type float64.

    Returns
    -------
    numpy.ndarray
        The matched image array of the same shape and type as src and dst.

    Notes
    -----
    This function converts the images to uint8 type, calculates the cumulative
    distribution functions (CDF) for each channel of the source and destination
    images, and then finds the best match for each pixel in the destination image
    in the source image's CDF. The matched values are then used to create a new
    image that has a similar color distribution to the source image.
    """
    # ----

    # Scale images to [0, 255] range and convert to uint8 type
    src = src * 255.0
    dst = dst * 255.0
    src = src.astype(np.uint8)
    dst = dst.astype(np.uint8)
    res = np.zeros_like(dst)

    # Initialize CDF arrays for each channel
    cdf_src = np.zeros((3, 256))
    cdf_dst = np.zeros((3, 256))
    cdf_res = np.zeros((3, 256))

    # Keyword arguments for histogram calculation
    kw = dict(bins=256, range=(0, 256), density=True)

    # Calculate histograms and CDFs for each channel
    for ch in range(3):
        his_src, _ = np.histogram(src[:, :, ch], **kw)
        hist_dst, _ = np.histogram(dst[:, :, ch], **kw)
        cdf_src[ch] = np.cumsum(his_src)
        cdf_dst[ch] = np.cumsum(hist_dst)

        # Find the best match index for each pixel in the destination image
        index = np.searchsorted(cdf_src[ch], cdf_dst[ch], side="left")
        np.clip(index, 0, 255, out=index)

        # Assign the matched values to the result image
        res[:, :, ch] = index[dst[:, :, ch]]

        # Update histogram and CDF for the result image
        his_res, _ = np.histogram(res[:, :, ch], **kw)
        cdf_res[ch] = np.cumsum(his_res)

    # Scale the result back to [0, 1] range and return
    return res / 255.0



# unit test cases
print(his_match(src = np.random.rand(100, 100, 3), dst = np.random.rand(100, 100, 3)))
print(his_match(src = np.full((50, 50, 3), 0.5), dst = np.full((50, 50, 3), 0.7)))
print(his_match(src = np.random.rand(64, 64, 3), dst = np.random.rand(128, 128, 3)))

