import torch

def interleave(tensor1, tensor2):
    """
    Interleaves two tensors along a new dimension, creating two interleaved sequences.
    If the 

    Args:
        tensor1 (torch.Tensor): The first tensor to interleave.
        tensor2 (torch.Tensor): The second tensor to interleave.

    Returns:
        tuple: A tuple containing two tensors. The first tensor is interleaved starting with tensor1,
               and the second tensor is interleaved starting with tensor2.

    This function interleaves two input tensors element-wise along a newly created dimension and then flattens
    the interleaved tensor back into the original dimension. It produces two outputs: one starting with the first tensor
    and the other starting with the second tensor. This can be particularly useful for tasks where data from two different
    sources need to be alternately processed or analyzed.

    Example Usage:
        tensor1 = torch.tensor([1, 3, 5])
        tensor2 = torch.tensor([2, 4, 6])
        res1, res2 = interleave(tensor1, tensor2)
        print(res1)  # Outputs: tensor([1, 2, 3, 4, 5, 6])
        print(res2)  # Outputs: tensor([2, 1, 4, 3, 6, 5])
    """
    # ----

    # Stack the tensors alternately along a new dimension (dim=1)
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)  # Flatten across original and new dimension
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)  # Start with the second tensor

    return res1, res2


# unit test cases
print(interleave(torch.tensor([1, 3, 5]), torch.tensor([2, 4, 6])))
print(interleave(torch.tensor([1, 3, 5, 7]), torch.tensor([2, 4, 5, 0])))
print(interleave(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])))