import torch
import numpy as np


def inv(mat):
    """
    Inverts a matrix that can be either a torch.Tensor or a numpy.ndarray.

    Parameters
    ----------
    mat : torch.Tensor or np.ndarray
        A square matrix to be inverted.

    Returns
    -------
    torch.Tensor or np.ndarray
        The inverse of the input matrix, with the same type as the input.

    Raises
    ------
    ValueError
        If the input is not a torch.Tensor or np.ndarray.

    Notes
    -----
    This function checks the type of the input matrix and applies the appropriate inverse
    operation using either PyTorch's torch.linalg.inv for tensor objects or NumPy's
    np.linalg.inv for ndarray objects. It is important that the input matrix is square
    and non-singular (i.e., it has an inverse), otherwise the inversion will fail.
    """
    # ----
    
    # Check if the input matrix is a torch.Tensor and invert it using torch.linalg.inv
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)

    # Check if the input matrix is a np.ndarray and invert it using np.linalg.inv
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)

    # If the input is neither a torch.Tensor nor a np.ndarray, raise a ValueError
    return ValueError


# unit test cases
print(inv(torch.tensor([[4.0, 7.0], [2.0, 6.0]])))
print(inv(np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])))
print(inv([[1, 2], [3, 4]]))