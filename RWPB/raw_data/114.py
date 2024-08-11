import torch

def reshape_weight_task_tensors(task_tensors, weights):
    """
    Reshapes `weights` to match the shape of `task_tensors` by unsqeezing in the remaining dimensions.

    Args:
        task_tensors (torch.Tensor): The tensors that will be used to reshape `weights`.
        weights (torch.Tensor): The tensor to be reshaped.

    Returns:
        torch.Tensor: The reshaped tensor.
    """
    # ----
    
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    weights = weights.view(new_shape)
    return weights


# unit test cases
task_tensors = torch.rand((2, 3, 4))
weights = torch.rand((2))
print(reshape_weight_task_tensors(task_tensors, weights))

task_tensors = torch.rand((5, 6, 7, 8))
weights = torch.rand((5, 6))
print(reshape_weight_task_tensors(task_tensors, weights))

task_tensors = torch.rand((10, 20))
weights = torch.rand((10))
print(reshape_weight_task_tensors(task_tensors, weights))