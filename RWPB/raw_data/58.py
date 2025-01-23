def unpad_image_shape(current_height, current_width, original_size):
    """
    Calculates the new shape of an image after unpading a resized and padded PyTorch tensor.

    Parameters
    ----------
    current_height : int
        The current height of the padded image.
    current_width : int
        The current width of the padded image.
    original_size : tuple
        A tuple representing the original size of the image in the format (original_width, original_height).

    Returns
    -------
    tuple
        The new shape of the image after unpadding, in the format (new_height, new_width).

    Notes
    -----
    This function first calculates the aspect ratios of the original and current images.
    It then determines how to unpad the image based on the aspect ratios to maintain the original aspect ratio.

    If the original aspect ratio is greater than the current aspect ratio, it means the image has been made wider.
    The function calculates the new height by scaling the original height with the same factor that the width has been scaled.
    It then computes the padding needed to center the original aspect ratio within the current dimensions.

    Conversely, if the original aspect ratio is less than the current aspect ratio, the image has been made taller.
    The function calculates the new width and computes the necessary padding to center the image vertically.

    The function returns the new dimensions of the image after unpadding, which can be used to trim the padded image back to its original shape.
    """
    # ----

    original_width, original_height = original_size

    # Calculate aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Check which dimension has been altered more and unpad accordingly
    if original_aspect_ratio > current_aspect_ratio:
        # The image has been made wider, so calculate new height and padding
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        new_shape = (current_height - 2 * padding, current_width)
    else:
        # The image has been made taller, so calculate new width and padding
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        new_shape = (current_height, current_width - 2 * padding)

    return new_shape


# unit test cases
print(unpad_image_shape(current_height=1000, current_width=2000, original_size=(1000, 500)))
print(unpad_image_shape(current_height=2000, current_width=1000, original_size=(500, 1000)))
print(unpad_image_shape(current_height=1200, current_width=1600, original_size=(600, 800)))
print(unpad_image_shape(current_height=100, current_width=160, original_size=(5, 1)))
print(unpad_image_shape(current_height=1231, current_width=4123, original_size=(76, 2)))