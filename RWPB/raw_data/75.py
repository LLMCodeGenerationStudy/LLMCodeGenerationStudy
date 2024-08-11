import numpy as np

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector 'a' to vector 'b' using an axis-angle rotation.

    Parameters:
    a (np.array): The initial vector to be aligned. Default is np.array([0, 0, 1]).
    b (np.array): The target vector to align with. Default is np.array([1, 0, 0]).

    Returns:
    tuple: A tuple containing:
        - axis_ (np.array): The normalized axis of rotation.
        - angle (float): The angle of rotation in radians.

    Process:
    1. Check if vectors 'a' and 'b' are already aligned. If they are, return (None, None).
    2. Compute the cross product of 'a' and 'b' to find the rotation axis.
    3. Normalize the rotation axis to get a unit vector.
    4. Calculate the angle between 'a' and 'b' using the arccosine of their dot product.

    If 'a' and 'b' are not aligned, the function returns the necessary rotation axis and angle to align 'a' with 'b'.
    """
    # ----
    
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


# unit test cases
print(align_vector_to_another(a = np.array([1, 0, 0]), b = np.array([1, 0, 0])))
print(align_vector_to_another(a = np.array([0, 0, 1]), b = np.array([1, 0, 0])))
print(align_vector_to_another(a = np.array([1, 0, 1]), b = np.array([-1, 0, 0])))
print(align_vector_to_another(a = np.array([1, 0, 0]), b = np.array([1, 0, 1])))
print(align_vector_to_another(a = np.array([1, 1, 1]), b = np.array([-1, 1, -1])))
