import importlib.util
import numpy as np
import torch

def is_tensor(x):
    """
    Determines if the input `x` is a `torch.Tensor` or a `np.ndarray`.

    Parameters:
    x : any type
        The input variable to be checked.

    Returns:
    bool
        Returns True if `x` is either a `torch.Tensor` (if the PyTorch library is available) 
        or a `np.ndarray`. Otherwise, returns False.

    Process:
    1. Checks if the PyTorch library is available using `importlib.util.find_spec("torch")`.
    2. If PyTorch is available, imports the library and checks if `x` is an instance of `torch.Tensor`.
    3. If `x` is not a `torch.Tensor` or if PyTorch is not available, it then checks if `x` is an instance of `np.ndarray`.
    """
    # ----
    
    if importlib.util.find_spec("torch") is not None:
        import torch

        if isinstance(x, torch.Tensor):
            return True

    return isinstance(x, np.ndarray)

# unit test cases
print(is_tensor(np.array([1, 2, 3])))
print(is_tensor([1, 2, 3]))
print(is_tensor(torch.tensor([1, 2, 3])))
print(is_tensor(0))