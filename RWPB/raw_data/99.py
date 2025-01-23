import math

def focal2fov(focal, pixels):
    """
    Converts focal length to field of view (FOV).
    If focal is zero, the code should return KeyError

    Args:
    focal (float): Focal length.
    pixels (int): Number of pixels.

    Returns:
    float: Field of view in radians.
    """
    # ----
    
    if int(focal) == 0:
        return KeyError
    return 2 * math.atan(pixels / (2 * focal))

# unit test cases

focal = 49.1
pixels = 50000
print(focal2fov(focal, pixels))

focal = 50.0
pixels = 1920
print(focal2fov(focal, pixels))

focal = 1.0
pixels = 1920
print(focal2fov(focal, pixels))

focal = 0
pixels = 2048
print(focal2fov(focal, pixels))