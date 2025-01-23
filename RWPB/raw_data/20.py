import math


def get_line_angle(x1, y1, x2, y2):
    '''
    Calculates the angle of a line segment relative to the horizontal axis.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of the first point (x1, y1).
    x2, y2 : float
        Coordinates of the second point (x2, y2).

    Return
    ------
    float
        The angle of the line segment from point (x1, y1) to (x2, y2) in degrees.

    The function computes the slope of the line segment and then uses the arctangent
    function to determine the angle in radians. This angle is converted to degrees.
    The function includes handling for vertical lines where the slope would be undefined.
    '''
    # ----

    # Check for a vertical line to avoid division by zero
    if x2 == x1:
        return 90 if y2 > y1 else -90  # Directly return 90 or -90 degrees depending on the direction

    # Calculate the slope of the line
    slope = (y2 - y1) / (x2 - x1)

    # Calculate the angle in radians
    angle_radians = math.atan(slope)

    # Convert the angle to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


# unit test cases
print(get_line_angle(0, 10, 10, 0))
print(get_line_angle(5, 5, 5, 10))
print(get_line_angle(0, 0, 10, 0))
