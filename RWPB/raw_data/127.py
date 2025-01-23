import torch

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    """
    Shifts a specified dimension of a tensor to a new position.

    Args:
        x (torch.Tensor): The input tensor.
        src_dim (int): The original position of the dimension to move. Negative values count from the end.
        dest_dim (int): The destination position for the dimension. Negative values count from the end.
        make_contiguous (bool): If True, returns a contiguous tensor. This can be necessary after a permute
                                operation because permute can lead to non-contiguous memory layout, potentially
                                affecting performance.

    Returns:
        torch.Tensor: A tensor with the specified dimension shifted to the new position.

    Example:
        # Assuming x is a tensor with shape (batch, channels, time, height, width)
        # and we want to move the 'channels' dimension to the end:
        shifted_tensor = shift_dim(x, 1, -1)
        # shifted_tensor will have the shape (batch, time, height, width, channels)
    """
    # ----
    
    n_dims = len(x.shape)  # Number of dimensions in the input tensor

    # Adjust src_dim and dest_dim to be positive if they are given as negative
    if src_dim < 0:
        src_dim += n_dims
    if dest_dim < 0:
        dest_dim += n_dims

    # Ensure the source and destination dimensions are within the valid range
    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims, "Source and destination dimensions must be within the valid range"

    # Create a list of dimensions except for the source dimension
    dims = list(range(n_dims))
    del dims[src_dim]  # Remove the source dimension from the list

    # Build a new permutation of dimensions
    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)  # Insert the source dimension at the destination index
        else:
            permutation.append(dims[ctr])  # Fill in the rest of dimensions
            ctr += 1

    # Permute the dimensions of the tensor according to the new order
    x = x.permute(permutation)

    # Make the tensor contiguous if required
    if make_contiguous:
        x = x.contiguous()

    return x


# unit test cases
tensor = torch.randn(2, 3, 4, 5)
shifted_tensor = shift_dim(tensor, src_dim=1, dest_dim=-1)
print(shifted_tensor)

tensor = torch.randn(3, 4, 5)
shifted_tensor = shift_dim(tensor, src_dim=-3, dest_dim=0, make_contiguous=False)
print(shifted_tensor)


tensor = torch.randn(1, 2, 3, 4, 5)
shifted_tensor = shift_dim(tensor, src_dim=0, dest_dim=4, make_contiguous=True)
print(shifted_tensor)