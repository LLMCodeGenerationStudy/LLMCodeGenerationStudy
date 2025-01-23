def merge(ids, pair, idx):
    """
    Merges consecutive occurrences of a pair of integers in a list with a new integer.

    Parameters
    ----------
    ids : list[int]
        The list of integers where the merging will occur.
    pair : tuple[int, int]
        The pair of integers to be merged.
    idx : int
        The new integer that will replace the consecutive occurrences of 'pair'.

    Returns
    -------
    list[int]
        A new list of integers with all occurrences of 'pair' merged into 'idx'.

    Examples
    --------
    >>> merge([1, 2, 3, 1, 2], (1, 2), 4)
    [4, 3, 4]

    The function iterates through the list 'ids' and checks for consecutive occurrences
    of the integers in 'pair'. When such a pair is found, it is replaced by 'idx' in the
    new list 'newids'. If the pair is not found, the current integer is simply appended to
    'newids'. The function returns the modified list with the pairs merged.
    """
    # ----

    newids = []
    i = 0
    while i < len(ids):
        # Check if the current and next item form the pair and are not the last element
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            # Append the idx to the new list and skip the next item
            newids.append(idx)
            i += 2
        else:
            # Append the current item to the new list
            newids.append(ids[i])
            i += 1
    return newids


# unit test cases
print(merge(ids = [1, 2, 1, 2, 1, 2, 3], pair = (1, 2), idx = 4))
print(merge(ids = [1, 3, 4, 5, 6], pair = (1, 2), idx = 4))
print(merge(ids = [1, 2, 3, 4, 1, 2], pair = (1, 2), idx = 4))