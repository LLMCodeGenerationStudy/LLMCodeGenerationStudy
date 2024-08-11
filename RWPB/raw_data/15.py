import numpy as np


def cosine_distance(a, b):
    """
    Compute the consine distance between vector a and vector b. Only support `a` is an embedding vector, `b` is a vector or matrix.
    if the length of a is not equal the length of b, please return KeyError
    Input: 
    a: numpy vector
    b: numpy vector
    return:
    the cosine distance between vector a and vector b: int.
    """
    # ----

    if a.shape[-1] != b.shape[-1]:
        return ValueError
    dist = np.dot(a, b.T) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return dist

# unit test cases
print(cosine_distance(a = np.array([1, 1]), b = np.array([0, 1])))
print(cosine_distance(a = np.array([1, 0, 0]), b = np.array([0, 1])))
print(cosine_distance(a = np.array([1, 0, 0]), b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))