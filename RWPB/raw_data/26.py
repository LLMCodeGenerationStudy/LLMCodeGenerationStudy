import torch

def to_one_hot(tensor, num_cls, dim=1, ignore_index=255):
    """
    Converts a given tensor to a one-hot encoded tensor.

    Args:
        tensor (torch.Tensor): A tensor containing class indices.
        num_cls (int): The total number of classes.
        dim (int, optional): The dimension along which to apply one-hot encoding. Default is 1.
        ignore_index (int, optional): A specific index that should be ignored during the conversion. Default is 255.

    Returns:
        torch.Tensor: A one-hot encoded tensor.

    This function takes a tensor with class indices (usually integers) and converts it into a one-hot
    encoded format. Entries with the 'ignore_index' are treated as background and set to zero before encoding.
    The function uses the scatter_ method to expand the tensor along the specified 'dim' dimension
    and sets the corresponding index to 1, indicating class presence.
    """
    # ----

    # Get dimensions of the input tensor
    b, h, w = tensor.shape

    # Set ignore_index to 0 to treat it as background
    tensor[tensor == ignore_index] = 0

    # Initialize a new tensor of zeros with dimensions for batch, classes, height, and width
    onehot_tensor = torch.zeros(b, num_cls, h, w)

    # Perform the one-hot encoding using scatter_
    onehot_tensor.scatter_(dim, tensor.unsqueeze(dim), 1)

    return onehot_tensor



# unit test cases
print(to_one_hot(torch.tensor([[[1, 2], [0, 1]]]), num_cls = 3))
print(to_one_hot(torch.tensor([[[255, 1], [2, 255]]]), num_cls = 4, ignore_index = 255))
print(to_one_hot(torch.tensor([[[0, 1, 2, 3, 4, 5]]]), num_cls = 1000))