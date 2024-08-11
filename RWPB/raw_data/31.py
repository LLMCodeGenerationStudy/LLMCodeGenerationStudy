import torch
import math

def resize_to_batch_size(tensor, batch_size):
    """
    Adjusts the batch size of a tensor to the specified size, interpolating if necessary.

    Args:
        tensor (torch.Tensor): Input tensor whose batch size needs to be adjusted.
        batch_size (int): The desired batch size after adjustment.

    Returns:
        torch.Tensor: The tensor with its batch size adjusted to the specified size.

    This function handles resizing of the batch dimension of a tensor. If the current batch size matches
    the desired batch size, the tensor is returned unchanged. For reducing the batch size, it uses linear
    interpolation to select indices, ensuring that the resized tensor covers the whole original batch as
    evenly as possible. For increasing the batch size, it applies a different interpolation method to avoid
    extrapolation and to ensure that every element from the original batch is represented.
    """
    # ----
    
    # Retrieve the initial batch size from the tensor
    in_batch_size = tensor.shape[0]

    # Return the tensor unchanged if the current batch size is already the desired size
    if in_batch_size == batch_size:
        return tensor

    # Return only the required portion of the tensor if the desired batch size is less than or equal to 1
    if batch_size <= 1:
        return tensor[:batch_size]

    # Create an empty tensor to store the output, maintaining the original data type and device
    output = torch.empty([batch_size] + list(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)

    # Handle reduction of batch size
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]

    # Handle increase in batch size
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            index = min(math.floor((i + 0.5) * scale), in_batch_size - 1)
            output[i] = tensor[index]

    return output


# unit test cases:
print(resize_to_batch_size(torch.randn(5, 3, 3), 5))
print(resize_to_batch_size(torch.randn(10, 4, 4), 3))
print(resize_to_batch_size(torch.randn(3, 2, 2), 8))