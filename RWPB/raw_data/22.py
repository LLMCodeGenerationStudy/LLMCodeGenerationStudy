def get_stats(ids, counts=None):
    """
    Computes the frequency of consecutive integer pairs in a list.

    Args:
        ids (list of int): List of integers whose consecutive pairs are to be counted.
        counts (dict, optional): Existing dictionary to update with new counts. If None, a new dictionary is created.

    Returns:
        dict: A dictionary where each key is a tuple representing a pair of consecutive integers,
              and the value is the count of how often each pair appears.

    Example:
        >>> get_stats([1, 2, 3, 1, 2])
        {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    This function iterates over consecutive elements in the provided list, creating pairs,
    and either updates an existing dictionary or creates a new one to record the frequency of each pair.
    """
    # ----

    counts = {} if counts is None else counts  # Initialize the dictionary if not provided
    for pair in zip(ids, ids[1:]):  # Create pairs from consecutive elements
        counts[pair] = counts.get(pair, 0) + 1  # Increment the count for each pair in the dictionary
    return counts


# unit test cases
print(get_stats([]))
print(get_stats([1, 2, 1, 2, 1, 2, 3, 4, 3, 4]))
print(get_stats([5, 6, 7, 5, 6], counts={(5, 6): 1, (6, 7): 2}))