import torch

def build_homog_matrix(R, t=None):
    """
    Create a batch of homogeneous transformation matrices from given rotation matrices and translation vectors.
    If the shape of R do not match the shape of t, please return ValueError

    Args:
    R (torch.Tensor): Rotation matrices of shape (B, Nj, 3, 3), where B is the batch size, and Nj is the number of joints.
    t (torch.Tensor, optional): Translation vectors of shape (B, Nj, 3, 1). If None, a zero translation vector is used.
    
    Returns:
    torch.Tensor: Homogeneous transformation matrices of shape (B, Nj, 4, 4).
    
    Process:
    1. If the translation vector `t` is not provided, initialize it to zero vectors of the appropriate shape.
    2. If the rotation matrix `R` is not provided, initialize it to identity matrices of the appropriate shape.
    3. Ensure that the shapes of `R` and `t` match the expected dimensions.
    4. Concatenate the rotation matrix `R` and translation vector `t` along the last dimension to form a (B, Nj, 3, 4) local transformation matrix.
    5. Create a padding row [0, 0, 0, 1] and expand it to match the batch and joint dimensions, forming a (B, Nj, 1, 4) tensor.
    6. Concatenate the padded row to the local transformation matrix to create a final homogeneous matrix of shape (B, Nj, 4, 4).
    7. Return the resulting homogeneous transformation matrix.
    """
    # ----
    
    
    if t is None:
        B = R.shape[0]
        Nj = R.shape[1]
        t = torch.zeros(B, Nj, 3, 1).to(R.device)
    
    if R is None:
        B = t.shape[0]
        Nj = t.shape[1]
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, Nj, 1, 1).to(t.device)
    
    B = t.shape[0]
    Nj = t.shape[1]
        
    # import ipdb; ipdb.set_trace()
    if R.shape != (B, Nj, 3, 3) or t.shape != (B, Nj, 3, 1):
        return KeyError
    
    G = torch.cat([R, t], dim=-1) # BxJx3x4 local transformation matrix
    pad_row = torch.FloatTensor([0, 0, 0, 1]).to(R.device).view(1, 1, 1, 4).expand(B, Nj, -1, -1) # BxJx1x4
    G = torch.cat([G, pad_row], dim=2) # BxJx4x4 padded to be 4x4 matrix an enable multiplication for the kinematic chain

    return G


# unit test cases
R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1)
t = torch.tensor([[[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]], [[7.0], [8.0], [9.0]]],
                  [[[1.0], [0.0], [1.0]], [[2.0], [2.0], [2.0]], [[3.0], [3.0], [3.0]]]])
print(build_homog_matrix(R=R, t=t))

R = torch.tensor([[[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                   [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]])
print(build_homog_matrix(R=R, t=t))

R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(3, 2, 1, 1)
t = torch.zeros(2, 3, 3, 1)  # Intentional shape mismatch to trigger an assertion error
print(build_homog_matrix(R=R, t=t))