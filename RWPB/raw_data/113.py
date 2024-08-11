import torch

def _linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Generate a sequence of evenly spaced values between start and stop tensors.

    Args:
        start (torch.Tensor): The starting values of the sequence.
        stop (torch.Tensor): The stopping values of the sequence.
        num (int): The number of values to generate.

    Returns:
        torch.Tensor: A tensor containing `num` evenly spaced values between `start` and `stop`.
    """
    # ----
    
    if num == 1:
        return start.clone()

    steps = torch.linspace(0, 1, num, dtype=start.dtype, device=start.device)

    steps = steps.view(*([1] * (start.dim() - 1)), num)
    out = start.unsqueeze(-1) + (stop - start).unsqueeze(-1) * steps
    return out


# unit test cases
start = torch.tensor([0.])
stop = torch.tensor([10.])
num = 5
print(_linspace(start, stop, num))

start = torch.tensor([1., 2., 3.])
stop = torch.tensor([1., 2., 3.])
num = 1
print(_linspace(start, stop, num))

start = torch.tensor([0, 10], dtype=torch.int32)
stop = torch.tensor([5, 15], dtype=torch.int32)
num = 3
print(_linspace(start, stop, num))

start = torch.tensor([5., 5., 5.])
stop = torch.tensor([5., 5., 5.])
num = 3
print(_linspace(start, stop, num))

start = torch.tensor([[0., 0.], [10., 10.]])
stop = torch.tensor([[5., 10.], [15., 20.]])
num = 4
print(_linspace(start, stop, num))