import torch

def project_onto_planes(planes: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """
    Projects 3D coordinates onto a batch of 2D planes, returning 2D plane coordinates.

    Args:
        planes (torch.Tensor): Tensor defining plane axes with shape (n_planes, 3, 3).
        coordinates (torch.Tensor): Tensor of 3D coordinates with shape (N, M, 3).

    Returns:
        torch.Tensor: Projected 2D coordinates with shape (N*n_planes, M, 2).
    """
    # ----
    
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


# unit test cases
planes = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
                       [[0, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=torch.float32)
coordinates = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32)
print(project_onto_planes(planes, coordinates))

planes = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
                       [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                       [[1, 0, 0], [0, 0, 1], [0, 1, 0]]], dtype=torch.float32)
coordinates = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                            [[9, 8, 7], [6, 5, 4], [3, 2, 1]]], dtype=torch.float32)
print(project_onto_planes(planes, coordinates))

planes = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
coordinates = torch.tensor([[[0, -1, -2]]], dtype=torch.float32)
print(project_onto_planes(planes, coordinates))