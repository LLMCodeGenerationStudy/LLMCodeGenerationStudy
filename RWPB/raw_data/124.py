import math
import torch


def make_divisible(x, divisor):
    """
    Adjusts x to the nearest value that is divisible by divisor.

    Arguments:
    x : int or float
        The value to be made divisible.
    divisor : int or torch.Tensor
        The divisor to be used. If a torch.Tensor is provided, uses its maximum value.

    Returns:
    int
        The nearest value to x that is divisible by the specified divisor.
    """
    # ----
    
    # Convert divisor to int if it is a torch.Tensor, taking the maximum value in the tensor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # Convert to int

    # Compute the nearest value to x that is divisible by divisor
    return math.ceil(x / divisor) * divisor  # Use ceil to ensure it rounds up to the nearest divisor

# unit test cases
x = 23
divisor = 5
print(make_divisible(x, divisor))

x = 14.7
divisor = 3
print(make_divisible(x, divisor))

x = 100
divisor = torch.Tensor([2,8,10])
print(make_divisible(x, divisor))