import torch

def lookAt(eye, at, up):
    """
    Creates a view transformation matrix using the "look-at" method, which is commonly used in 3D graphics to create a camera view matrix.

    Parameters:
    eye (torch.Tensor): A 1D tensor with 3 elements representing the position of the camera in world coordinates.
    at (torch.Tensor): A 1D tensor with 3 elements representing the point in world coordinates that the camera is looking at.
    up (torch.Tensor): A 1D tensor with 3 elements representing the up direction of the camera.

    Returns:
    torch.Tensor: A 4x4 tensor representing the view transformation matrix.

    Process:
    1. Compute the forward vector `w` as the normalized vector pointing from `at` to `eye`.
    2. Compute the right vector `u` as the normalized cross product of `up` and `w`.
    3. Compute the true up vector `v` as the cross product of `w` and `u`.
    4. Construct the translation matrix to move the scene so that the camera is at the origin.
    5. Construct the rotation matrix to align the camera's orientation with the computed `u`, `v`, and `w` vectors.
    6. Combine the translation and rotation matrices by matrix multiplication to get the final view transformation matrix.
    """
    # ----
    
    a = eye - at
    w = a / torch.linalg.norm(a)
    u = torch.cross(up, w)
    u = u / torch.linalg.norm(u)
    v = torch.cross(w, u)
    translate = torch.tensor([[1, 0, 0, -eye[0]], 
                              [0, 1, 0, -eye[1]], 
                              [0, 0, 1, -eye[2]], 
                              [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    rotate = torch.tensor([[u[0], u[1], u[2], 0], 
                           [v[0], v[1], v[2], 0], 
                           [w[0], w[1], w[2], 0], 
                           [0, 0, 0, 1]], dtype=eye.dtype, device=eye.device)
    return rotate @ translate


# unit test cases

eye = torch.tensor([1.0, 1, 1])
at = torch.tensor([0.0, 0, 0])
up = torch.tensor([0.0, 1, 0])
print(lookAt(eye, at, up))

eye = torch.tensor([0.0, 0, 1])
at = torch.tensor([0.0, 0, 0])
up = torch.tensor([0.0, 1, 0])
print(lookAt(eye, at, up))

eye = torch.tensor([-3.0, 2, 5])
at = torch.tensor([4.0, -1, 0])
up = torch.tensor([1.0, 1, 1])
print(lookAt(eye, at, up))