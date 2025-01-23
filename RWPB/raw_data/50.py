import torch

def unpad_image(tensor: torch.Tensor, original_size: tuple) -> torch.Tensor:
    """
    Removes padding from an image tensor to restore it to its original aspect ratio.
    The function assumes the tensor is in the CxHxW format and has been padded symmetrically.

    Args:
        tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
        original_size (tuple): The original dimensions of the image as (height, width).

    Returns:
        torch.Tensor: The unpadded image tensor resized back to the original aspect ratio.

    The function calculates the aspect ratio of the original image and the current tensor,
    determining the necessary adjustments to either height or width. It then crops the tensor
    to match the original aspect ratio, ensuring that the central part of the image is retained.
    """
    # ----

    # Extract original and current dimensions
    original_height, original_width = original_size
    _, current_height, current_width = tensor.shape

    # Calculate aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine how to adjust tensor to match original aspect ratio
    if original_aspect_ratio > current_aspect_ratio:
        # Original is wider than current: adjust height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        vertical_padding = (current_height - new_height) // 2
        # Crop vertically
        unpadded_tensor = tensor[:, vertical_padding:vertical_padding + new_height, :]
    else:
        # Original is taller than current: adjust width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        horizontal_padding = (current_width - new_width) // 2
        # Crop horizontally
        unpadded_tensor = tensor[:, :, horizontal_padding:horizontal_padding + new_width]

    return unpadded_tensor


# unit test cases
print(unpad_image(torch.rand(3, 10, 20), (5, 10)))
print(unpad_image(torch.rand(3, 20, 10), (10, 5)))
print(unpad_image(torch.rand(3, 15, 30), (10, 20)))