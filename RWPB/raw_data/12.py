import torch


def normalize_activation(x, eps=1e-10):
    """
    Normalize activation values along the channel dimension.
    Args:
        - x: Tensor, the input activation tensor.
        - eps: float, optional, small value to avoid division by zero.
    Return:
        normalized_activation: Tensor, the normalized activation tensor.
    """
    # ----
    
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)



# unit test cases
print(normalize_activation(torch.rand([1, 3, 5, 5])))
print(normalize_activation(torch.rand([2, 3, 4, 4])))
print(normalize_activation(torch.rand([1, 1, 1, 1])))