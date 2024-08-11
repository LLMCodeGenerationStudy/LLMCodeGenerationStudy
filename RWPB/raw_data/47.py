def disjoint_cycles(perm):
    """
    Find the disjoint cycles of a permutation.

    Args:
        perm (list): The permutation as a list.

    Returns:
        list: A list of lists, where each inner list represents a disjoint cycle.
    """
    # ----

    used = set()
    cycles = []
    minimum = 1
    while len(used) < len(perm):
        cycle = [minimum]
        used.add(minimum)
        elt = perm[minimum - 1]
        while elt not in cycle:
            cycle.append(elt)
            used.add(elt)
            elt = perm[elt - 1]
        if len(set(perm).symmetric_difference(used)) != 0:
            minimum = min(set(perm).symmetric_difference(used))
        cycles.append(cycle)
    return cycles

# unit test cases
print(disjoint_cycles([2, 3, 4, 5, 1]))
print(disjoint_cycles([2, 1, 4, 3, 6, 5]))
print(disjoint_cycles([1, 3, 2, 5, 4]))