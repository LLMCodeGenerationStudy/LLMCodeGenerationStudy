from PIL import Image
import numpy as np

def resize_and_center_crop(image, target_width, target_height):
    """
    Resizes an input image and then crops it to the target dimensions, centered on the resized image.

    Args:
        image (np.array): The input image as a NumPy array.
        target_width (int): The desired width of the output image.
        target_height (int): The desired height of the output image.

    Returns:
        np.array: The processed image as a NumPy array, resized and cropped to the target dimensions.

    This function first converts the input image to a PIL Image object, then calculates the scale factor needed
    to ensure that the resized image will cover the target dimensions. It resizes the image using the LANCZOS filter
    for high-quality downsampling. After resizing, it calculates the coordinates necessary to crop the image to the
    target dimensions centered on the resized image, and finally converts the cropped PIL image back to a NumPy array.
    """
    # ----

    # Convert the input NumPy array to a PIL Image object
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size

    # Calculate the scale factor to resize the image
    scale_factor = max(target_width / original_width, target_height / original_height)

    # Calculate the new dimensions
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))

    # Resize the image using the LANCZOS filter for high quality
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)

    # Calculate the coordinates for cropping the image to center it
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2

    # Crop the image using the calculated coordinates
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Convert the cropped PIL image back to a NumPy array and return
    return np.array(cropped_image)


# unit test cases
print(resize_and_center_crop(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8), 100, 100))
print(resize_and_center_crop(np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8), 150, 100))
print(resize_and_center_crop(np.random.randint(0, 256, (1000, 800, 3), dtype=np.uint8), 200, 150))