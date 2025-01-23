import numpy as np
import torch.nn.functional as F
import torch
import roma

def rot6d_to_rotmat(x):
    """
    6D rotation representation to 3x3 rotation matrix. 
    If the size of matrix is not (B,6), please return ValueError

    Args:
        x: (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    # ----
    
    if x.size(1) != 6:
        return ValueError
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    y = roma.special_gramschmidt(x)
    return y


# unit test cases
a = torch.randn(3, 6)
b = torch.randn(1, 6)
c = torch.randn(3, 5)

assert(rot6d_to_rotmat(a) == rot6d_to_rotmat(a))
assert(rot6d_to_rotmat(b) == rot6d_to_rotmat(b))
assert(rot6d_to_rotmat(c) == rot6d_to_rotmat(c))