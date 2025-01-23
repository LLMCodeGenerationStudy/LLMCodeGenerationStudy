import torch

def euler2rot(euler_angle):
    """
    Converts Euler angles to a rotation matrix.

    Args:
    euler_angle (torch.Tensor): Tensor of shape [batch_size, 3] containing Euler angles.

    Returns:
    torch.Tensor: Rotation matrix of shape [batch_size, 3, 3].
    """
    # ----

    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


# unit test cases
a = torch.tensor([[0.0, 0.0, 0.0]])
print(euler2rot(a))

a = torch.tensor([[0.0, 0.0, torch.pi / 4]])
print(euler2rot(a))

a = torch.tensor([[torch.pi / 2, torch.pi / 4, torch.pi / 6]])
print(euler2rot(a))