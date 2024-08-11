import numpy as np

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    '''Aligns vector a to vector b with axis angle rotation.
    If array a is equal array b, please return (None, None)
    Args:
        - a: numpy.array. the first vector. 
        - b: numpy.array. the second vector.
    Return:
        - axis: numpy.array. the axis of vector
        - angle: numpy.array. the angle of vector
    '''
    # ----

    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

# unit test cases
print(align_vector_to_another(a = [1, 0, 0], b = [1, 0, 0]))
print(align_vector_to_another(a = [0, 0, 1], b = [0, 1, 0]))
print(align_vector_to_another(a = [1, 1, 0], b = [-1, 0, 0]))