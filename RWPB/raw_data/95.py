import torch

def lines_focal(o, d):
    """
    Estimates the focal point of a set of lines in 3D space given their origins and directions.

    Parameters:
    o (torch.Tensor): A tensor of shape (N, 3) representing the origins of N lines in 3D space.
    d (torch.Tensor): A tensor of shape (N, 3) representing the directions of N lines in 3D space.

    Returns:
    torch.Tensor: A tensor of shape (3,) representing the estimated focal point in 3D space.

    Process:
    1. Normalize the direction vectors `d` to unit length along the last dimension.
    2. Create a 3x3 identity matrix `I` with the same dtype and device as the input tensors.
    3. Compute the sum `S` of the outer products of each direction vector with itself, subtracting the identity matrix `I` from each outer product. This results in a tensor of shape (3, 3).
    4. Compute the sum `C` of the matrix products of each adjusted direction outer product with the corresponding origin vector. This results in a tensor of shape (3,).
    5. Compute the pseudo-inverse of `S` using `torch.linalg.pinv`.
    6. Multiply the pseudo-inverse of `S` with `C` to obtain the estimated focal point.

    Returns the estimated focal point as a tensor of shape (3,).
    """
    # ----
    
    d = d / torch.norm(d, dim=-1, keepdim=True)
    I = torch.eye(3, dtype=o.dtype, device=o.device)
    S = torch.sum(d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...], dim=0)
    C = torch.sum((d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...]) @ o[..., None], dim=0).squeeze(1)
    return torch.linalg.pinv(S) @ C


# unit test cases
o = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
d = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
print(lines_focal(o, d))

o = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float32)
d = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
print(lines_focal(o, d))

o = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.float32)
d = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
print(lines_focal(o, d))