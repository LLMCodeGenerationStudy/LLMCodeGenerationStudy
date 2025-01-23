def submatrix(matrix, col):
    """
    Extracts the submatrix formed by deleting a specified column from the given matrix.

    Args:
        - matrix: list of lists, the input matrix.
        - col: int, the index of the column to be removed.

    Returns:
        sub_matrix: list of lists, the submatrix obtained after removing the specified column.
    """
    # ----
    
    return [row[:col] + row[col + 1:] for row in matrix[1:]]



# unit test cases
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
col = 1
print(submatrix(matrix, col))

matrix = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
col = 0
print(submatrix(matrix, col))

matrix = [[1, 2, 3], [4, 5], [6]]
col = 2
print(submatrix(matrix, col))

matrix = [[1, 5, 10], [4, 5], [6]]
col = 1
print(submatrix(matrix, col))