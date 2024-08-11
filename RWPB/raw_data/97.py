import numpy as np
import torch

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy image array to a normalized PyTorch tensor and reorders the dimensions.
    If the img is invalid, please return ValueError
    
    Parameters:
    image (np.ndarray): The input image in NumPy array format. 
                        - For color images, the shape should be (H, W, C) where H is height, W is width, and C is the number of channels (typically 3 for RGB).
                        - For grayscale images, the shape should be (H, W).

    Returns:
    torch.Tensor: The converted image as a PyTorch tensor with normalized values in the range [0, 1].
                  - For color images, the shape will be (C, H, W).
                  - For grayscale images, the shape will be (1, H, W).

    Process:
    1. Check the number of dimensions of the input image:
       - If the image is a 3D array (color image), transpose the dimensions from (H, W, C) to (C, H, W).
       - If the image is a 2D array (grayscale image), add a new axis to create a channel dimension, resulting in a shape of (1, H, W).
       - If the image does not match these conditions, raise a ValueError indicating it is not a valid image.
    2. Normalize the pixel values by dividing by 255.0 to convert the range from [0, 255] to [0, 1].
    3. Convert the resulting NumPy array to a PyTorch tensor with dtype=torch.float.
    """
    # ----
    
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        return ValueError
    return torch.tensor(image / 255.0, dtype=torch.float)


# unit test cases


color_image = np.random.randint(0, 256, size=(100, 200, 3), dtype=np.uint8)
print(numpy_image_to_torch(color_image))

grayscale_image = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
print(numpy_image_to_torch(grayscale_image))

invalid_image = np.random.randint(0, 256, size=(100,), dtype=np.uint8)
print(numpy_image_to_torch(invalid_image))
