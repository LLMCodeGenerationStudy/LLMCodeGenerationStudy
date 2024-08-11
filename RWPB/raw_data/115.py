import torch
import torch.nn as nn

def compute_tri_normal(geometry, tris):
    """
    Compute the normal vectors for triangles defined by vertex indices.

    Args:
        geometry (torch.Tensor): A tensor of shape (batch_size, num_points, 3) representing the 3D coordinates of points.
        tris (torch.Tensor): A tensor of shape (num_tris, 3) representing the vertex indices that form triangles.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_tris, 3) representing the normal vectors of the triangles.
    """
    # ----
    
    geometry = geometry.permute(0, 2, 1)
    tri_1 = tris[:, 0]
    tri_2 = tris[:, 1]
    tri_3 = tris[:, 2]

    vert_1 = torch.index_select(geometry, 2, tri_1)
    vert_2 = torch.index_select(geometry, 2, tri_2)
    vert_3 = torch.index_select(geometry, 2, tri_3)

    nnorm = torch.cross(vert_2 - vert_1, vert_3 - vert_1, 1)
    normal = nn.functional.normalize(nnorm).permute(0, 2, 1)
    return normal


# unit test cases
geometry = torch.tensor([[[0., 0., 0.],
                  [1., 0., 0.],
                  [0., 1., 0.]]], dtype=torch.float32)
tris = torch.tensor([[0, 1, 2]], dtype=torch.int64)
print(compute_tri_normal(geometry, tris))

geometry = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [2., 0., 0.], [0., 2., 0.]],
                 [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.], [1., 1., 1.], [1., 0., 1.]]],
                dtype=torch.float32)
tris = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
print(compute_tri_normal(geometry, tris))

geometry = torch.tensor([[[0., 0., 0.],
                  [1., 1., 1.],
                  [2., 2., 2.]]], dtype=torch.float32)
tris = torch.tensor([[0, 1, 2]], dtype=torch.int64)
print(compute_tri_normal(geometry, tris))