import math
import numpy as np
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN



def find_intersection(plane_normal,plane_point,ray_direction,ray_point):
    """
    Finds the intersection point of a line and a plane.

    Args:
        - plane_normal: numpy array, representing a normal vector of the plane.
        - plane_point: numpy array, representing a point on the plane.
        - ray_direction: numpy array, representing a direction vector of the line.
        - ray_point: numpy array, representing a point on the line.

    Returns:
        Psi: numpy array or None, the intersection point of the line and the plane.
    """
    # ----
    
    epsilon=1e-6
    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        # no intersection or line is within plane
        return None
    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * ray_direction + plane_point #intersections
    return Psi


# unit test cases
plane_normal = np.array([0, 0, 1])
plane_point = np.array([0, 0, 5])
ray_direction = np.array([1, 1, -1])
ray_point = np.array([1, 1, 10])
print(find_intersection(plane_point, plane_point, ray_direction, ray_point))

plane_normal = np.array([0, 0, 1])
plane_point = np.array([0, 3, 5])
ray_direction = np.array([1, 1, -1])
ray_point = np.array([1, 1, 9])
print(find_intersection(plane_point, plane_point, ray_direction, ray_point))

plane_normal = np.array([0, 1, 0])
plane_point = np.array([0, 2, 0])
ray_direction = np.array([1, 0, 0])
ray_point = np.array([0, 0, 0])
print(find_intersection(plane_point, plane_point, ray_direction, ray_point))

plane_normal = np.array([0, 0, 1])
plane_point = np.array([0, 0, 0])
ray_direction = np.array([1, 1, 0])
ray_point = np.array([1, 1, 0])
print(find_intersection(plane_point, plane_point, ray_direction, ray_point))