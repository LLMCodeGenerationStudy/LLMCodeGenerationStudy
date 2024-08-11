import torch

def pad_camera_extrinsics_4x4(extrinsics):
    """
    Pads a given extrinsics matrix to a 4x4 matrix if it is not already.

    Args:
        extrinsics (torch.Tensor): Input extrinsics matrix of shape (..., 3, 4).

    Returns:
        torch.Tensor: Padded extrinsics matrix of shape (..., 4, 4).
    """
    # ----
    
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


# unit test cases
extrinsics = torch.rand((2, 4, 4))
print(pad_camera_extrinsics_4x4(extrinsics))

extrinsics = torch.rand((3, 3, 4))
print(pad_camera_extrinsics_4x4(extrinsics))

extrinsics = torch.rand((3, 4))
print(pad_camera_extrinsics_4x4(extrinsics))

extrinsics = torch.rand((2, 1, 4))
print(pad_camera_extrinsics_4x4(extrinsics))