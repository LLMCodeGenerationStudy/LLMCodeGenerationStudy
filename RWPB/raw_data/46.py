from typing import List, Union

import numpy as np
def cosine_similarity(X: Union[List[List[float]], List[np.ndarray], np.ndarray], Y: Union[List[List[float]], List[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Compute the row-wise cosine similarity between two matrices of equal width.
    If the length of x is not equal the length of Y, return ValueError
    
    Parameters:
    X (Union[List[List[float]], List[np.ndarray], np.ndarray]): A matrix (or a list of lists) where each row represents a vector.
    Y (Union[List[List[float]], List[np.ndarray], np.ndarray]): Another matrix (or a list of lists) where each row represents a vector. Must have the same number of columns as X.

    Returns:
    np.ndarray: A matrix of cosine similarities. Each element at position (i, j) represents the cosine similarity between the i-th row of X and the j-th row of Y.

    The process involves:
    1. Converting the inputs X and Y to numpy arrays if they are not already in that format.
    2. Checking that X and Y have the same number of columns to ensure they are comparable.
    3. Calculating the L2 norm (Euclidean norm) of each row in X and Y.
    4. Computing the cosine similarity using the dot product of X and Y, normalized by the outer product of the row norms.
    5. Handling cases where division by zero or invalid operations might occur by setting such results to zero.
    """
    # ----

    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        return ValueError
    
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    # Ignore divide by zero errors run time warnings as those are handled below.
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
    similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
    return similarity


# unit test cases
print(cosine_similarity([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
print(cosine_similarity([[0, 0], [1, 1]], [[1, 1], [0, 0]]))
print(cosine_similarity([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4]]))
print(cosine_similarity(np.array([[1, 2], [3, 4]]), [[5, 6], [7, 8]]))
print(cosine_similarity(np.array([[0, 0], [1, 1]]), np.array([[1, 1], [0, 0]])))
