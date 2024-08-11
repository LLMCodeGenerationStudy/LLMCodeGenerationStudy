import torch

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    # ----
    
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat




# unit test cases
print(quat_to_rotmat(torch.tensor([[1.0, 0.0, 0.0, 0.0]])))
print(quat_to_rotmat(torch.tensor([
    [0.7071, 0.7071, 0.0, 0.0],
    [0.7071, 0.0, 0.7071, 0.0],
    [0.7071, 0.0, 0.0, 0.7071]
])))
print(quat_to_rotmat(torch.tensor([
    [2.0, 0.0, 0.0, 0.0],
    [1.4142, 1.4142, 0.0, 0.0],
    [10.0, 0.0, 0.0, 10.0]
])
))