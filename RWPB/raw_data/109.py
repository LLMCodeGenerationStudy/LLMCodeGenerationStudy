import torch

def rot_trans_geo(geometry, rot, trans):
    """
    Applies rotation and translation to geometry.

    Args:
    geometry (torch.Tensor): Tensor of shape [batch_size, num_points, 3] representing the geometry points.
    rot (torch.Tensor): Rotation matrix of shape [batch_size, 3, 3].
    trans (torch.Tensor): Translation vector of shape [batch_size, 3].

    Returns:
    torch.Tensor: Transformed geometry of shape [batch_size, num_points, 3].
    """
    # ----
    
    rott_geo = torch.bmm(rot, geometry.permute(0, 2, 1)) + trans.view(-1, 3, 1)
    return rott_geo.permute(0, 2, 1)


# unit test cases
geometry = torch.tensor([[[1., 2., 3.],
                          [4., 5., 6.]]])
rot = torch.eye(3).unsqueeze(0)  # Identity rotation for 1 batch
trans = torch.zeros(1, 3)        # Zero translation for 1 batch
print(rot_trans_geo(geometry, rot, trans))

geometry = torch.tensor([[[1., 0., 0.],
                          [0., 1., 0.]],
                         [[0., 0., 1.],
                          [1., 1., 1.]]])
rot = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],  # 90 degrees on Z-axis
                    [[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]]) # 180 degrees on Y-axis
trans = torch.tensor([[1., 2., 3.],
                      [0., -1., 2.]])
print(rot_trans_geo(geometry, rot, trans))

geometry = torch.tensor([[[0., 0., 0.],
                          [999., 999., 999.]]])
rot = torch.tensor([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]])  # 90 degrees on X-axis
trans = torch.tensor([[1000., 2000., 3000.]])
print(rot_trans_geo(geometry, rot, trans))