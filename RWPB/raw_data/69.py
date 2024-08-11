import numpy as np
import math

def fovx_to_fovy(fovx, aspect):
    """
    Converts the horizontal field of view (FoVx) to the vertical field of view (FoVy) based on the aspect ratio.

    Parameters:
    fovx (float): The horizontal field of view in radians.
    aspect (float): The aspect ratio of the image (width/height).

    Returns:
    float: The vertical field of view in radians.

    Process:
    1. Calculate the tangent of half the horizontal field of view (FoVx/2).
    2. Divide this value by the aspect ratio to get the tangent of half the vertical field of view.
    3. Use the arctangent function to find half the vertical field of view from its tangent.
    4. Multiply by 2 to obtain the full vertical field of view (FoVy).
    """
    # ----
    
    return np.arctan(np.tan(fovx / 2) / aspect) * 2.0


# unit test cases
print(fovx_to_fovy(math.pi/4, 16/9))
print(fovx_to_fovy(math.pi/3, 0.5))
print(fovx_to_fovy(math.pi, 1))