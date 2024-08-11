import numpy as np

def get_focalLength_from_fieldOfView(fov=60, img_size=512):
    """
    Compute the focal length of the camera lens by assuming a certain FOV for the entire image
    Args:
        - fov: float, expressed in degree
        - img_size: int
    Return:
        focal: float
    """
    # ----
    
    focal = img_size / (2 * np.tan(np.radians(fov) /2))
    return focal


# unit test cases
assert(get_focalLength_from_fieldOfView(fov = 60, img_size = 512) == get_focalLength_from_fieldOfView(fov = 60, img_size = 512))
assert(get_focalLength_from_fieldOfView(fov = 1, img_size = 512) == get_focalLength_from_fieldOfView(fov = 1, img_size = 512))
assert(get_focalLength_from_fieldOfView(fov = 120, img_size = 2048) == get_focalLength_from_fieldOfView(fov = 120, img_size = 2048))