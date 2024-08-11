import torch

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, DOT_THRESHOLD: float = 0.9995) -> torch.Tensor:
    """
    Performs spherical linear interpolation between two vectors.

    Args:
        v0 (torch.Tensor): The starting vector of the interpolation.
        v1 (torch.Tensor): The ending vector of the interpolation.
        t (float): The interpolation parameter, where 0 <= t <= 1.
        DOT_THRESHOLD (float): A threshold for the cosine of the angle between v0 and v1 to handle numerical instabilities.

    Returns:
        torch.Tensor: The interpolated vector at parameter t along the great circle joining v0 and v1.
    """
    # ----
    
    # Normalize the input vectors to unit vectors
    u0 = v0 / v0.norm()
    u1 = v1 / v1.norm()
    # Compute the cosine of the angle between the vectors
    dot = (u0 * u1).sum()
    # Check if the vectors are almost parallel
    if dot.abs() > DOT_THRESHOLD:
        # If vectors are nearly parallel, fall back to linear interpolation
        return (1.0 - t) * v0 + t * v1
    # Compute the angle between the vectors using arccos
    omega = dot.acos()
    # Perform spherical linear interpolation
    return (((1.0 - t) * omega).sin() * v0 + (t * omega).sin() * v1) / omega.sin()

# unit test cases
v0 = torch.tensor([1.0, 0.0, 0.0])
v1 = torch.tensor([0.0, 1.0, 0.0])
t = 0.5
print(slerp(v0, v1, t))

v0 = torch.tensor([1.0, 0.0, 0.0])
v1 = torch.tensor([1.0, 0.0, 0.0])
t = 0.3
print(slerp(v0, v1, t))

v0 = torch.tensor([0.9995, 0.01, 0.0])
v1 = torch.tensor([1.0, 0.0, 0.0])
t = 0.7
print(slerp(v0, v1, t))