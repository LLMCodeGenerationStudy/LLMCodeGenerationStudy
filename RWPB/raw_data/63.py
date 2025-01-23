import torch
from einops import rearrange



def rotate_half(x):
    """
    Rotates the last dimension of a tensor by 90 degrees counterclockwise,
    assuming the last dimension represents 2D coordinates or complex numbers.

    Arguments:
    x : torch.Tensor
        Input tensor with its last dimension size as a multiple of 2, representing 2D vectors.

    Returns:
    torch.Tensor
        A tensor with the same shape as the input, where each 2D vector has been rotated.
    """
    # ----
    
    # Reshape x assuming the last dimension can be split into 2 parts
    x = rearrange(x, '... (d r) -> ... d r', r=2)

    # Unbind the last dimension into two separate tensors
    x1, x2 = x.unbind(dim=-1)

    # Rotate each vector by 90 degrees counterclockwise
    x = torch.stack((-x2, x1), dim=-1)

    # Rearrange back to the original dimension structure
    return rearrange(x, '... d r -> ... (d r)')

# unit test cases
print(rotate_half(torch.tensor([[1, 0], [0, 1]])))
print(rotate_half(torch.tensor([[[1, 0, 0, 1], [0, 1, -1, 0]]])))
print(rotate_half(torch.tensor([[1, 0]])))