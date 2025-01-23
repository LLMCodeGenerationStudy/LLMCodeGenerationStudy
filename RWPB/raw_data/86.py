import torch

def inverse_sigmoid(x):
    """
    Computes the inverse sigmoid of a tensor.

    Args:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Tensor with the inverse sigmoid applied.
    """
    # ----
    
    return torch.log(x / (1 - x))

# unit test cases
x = torch.tensor([0.2, 0.5, 0.8])
print(inverse_sigmoid(x))

x = torch.tensor([0.001, 0.999])
print(inverse_sigmoid(x))

x = torch.tensor([0.0, 1.0])
print(inverse_sigmoid(x))