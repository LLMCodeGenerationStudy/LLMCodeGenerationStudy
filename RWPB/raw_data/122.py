import torch

def view_range(x, i, j, shape):
    """
    Reshapes a portion of a tensor from dimension i (inclusive) to dimension j (exclusive) into the specified shape.

    Args:
        x (torch.Tensor): The input tensor.
        i (int): The start dimension for reshaping, inclusive. Negative values count from the end.
        j (int): The end dimension for reshaping, exclusive. If None, extends to the end of dimensions. Negative values count from the end.
        shape (tuple): The target shape to apply between dimensions i and j.

    Returns:
        torch.Tensor: A tensor reshaped from dimension i to j according to the specified shape.

    Example:
        # Assuming x has shape (batch, thw, channels)
        # To reshape 'thw' into (t, h, w):
        reshaped_x = view_range(x, 1, 2, (t, h, w))
        # reshaped_x will have the shape (batch, t, h, w, channels)
    """
    # ----
    
    shape = tuple(shape)  # Ensure the shape is a tuple, necessary for concatenation with other shapes

    n_dims = len(x.shape)  # Total number of dimensions in the input tensor

    # Adjust indices for negative values
    if i < 0:
        i += n_dims
    if j is None:
        j = n_dims
    elif j < 0:
        j += n_dims

    # Ensure the indices are valid
    assert 0 <= i < j <= n_dims, "Indices i and j must define a valid range of dimensions"

    # Get the current shape of the tensor
    x_shape = x.shape
    # Construct the new shape by combining slices of the old shape with the new specified shape
    target_shape = x_shape[:i] + shape + x_shape[j:]

    # Reshape the tensor to the target shape
    return x.view(target_shape)

# unit test cases
x = torch.randn(4, 3, 2, 5)
i = 1
j = 3
shape = (3, 2)
print(view_range(x, i, j, shape))

x = torch.randn(4, 3, 2, 5, 6)
i = -4
j = -2
shape = (6,)
print(view_range(x, i, j, shape))

x = torch.randn(4, 3, 2)
i = 1
j = None
shape = (6,)
print(view_range(x, i, j, shape))