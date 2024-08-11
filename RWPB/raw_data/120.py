def parity(cycles):
    """
    Calculate the parity (even or odd) of a permutation given its disjoint cycles.

    Args:
        cycles (list): A list of disjoint cycles, where each cycle is represented as a list.

    Returns:
        int: 0 if the permutation is even, 1 if it is odd.
    """
    # ----
    
    res = 0
    for i in cycles:
        if len(i) > 1:
            res += len(i) - 1
    return res % 2

# unit test cases
cycles = [[1,2,3,4]]
print(parity(cycles))

cycles = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
print(parity(cycles))

cycles = [[1], [2, 3], [4, 5, 6], [7]]
print(parity(cycles))

cycles = []
print(parity(cycles))

cycles = [[1], [2], [3], [4]]
print(parity(cycles))

cycles = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
print(parity(cycles))

cycles = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(parity(cycles))

cycles = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11]]
print(parity(cycles))

cycles = [[1]]
print(parity(cycles))