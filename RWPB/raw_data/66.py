import numpy as np

def focal_length_to_fovy(focal_length, sensor_height):
    """
    Calculates the vertical field of view (FoVy) from the focal length of the lens and the sensor height.

    Parameters:
    focal_length (float): The focal length of the lens.
    sensor_height (float): The height of the camera sensor.

    Returns:
    float: The vertical field of view in radians.

    Process:
    1. Compute half the sensor height.
    2. Divide half the sensor height by the focal length to get the tangent of half the vertical field of view.
    3. Use the arctangent function to determine half the vertical field of view.
    4. Multiply by 2 to get the full vertical field of view (FoVy).
    """
    # ----
    
    return 2 * np.arctan(0.5 * sensor_height / focal_length)


# unit test cases
print(focal_length_to_fovy(focal_length=50, sensor_height=24))
print(focal_length_to_fovy(focal_length=1, sensor_height=24))
print(focal_length_to_fovy(focal_length=1000, sensor_height=24))