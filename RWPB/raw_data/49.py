import torch

def make_even_first_dim(tensor):
    """
    Ensures that the first dimension of the tensor is even. If it's odd, reduces it by one.

    Args:
    tensor (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Tensor with an even first dimension.
    """
    # ----

    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor

# unit test cases
print(make_even_first_dim(torch.randn(4, 3, 2)))
print(make_even_first_dim(torch.randn(5, 3, 2)))
print(make_even_first_dim(torch.randn(1, 4, 4)))