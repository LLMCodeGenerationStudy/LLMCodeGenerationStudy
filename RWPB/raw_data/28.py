import numpy as np


def mean(l, ignore_nan=False, empty=0):
    """
    Calculates the mean of a collection of numbers, handling NaN values and empty collections.

    Parameters
    ----------
    l : iterable
        An iterable of numbers (e.g., list, tuple, generator).
    ignore_nan : bool, optional
        If True, NaN values are ignored in the calculation. Default is False.
    empty : int, str, optional
        Defines the behavior when the input iterable is empty:
            - If 0, the function returns 0.
            - If 'raise', the function return a ValueError.
            - For other values, the function returns the value of `empty`.

    Returns
    -------
    float
        The mean of the input numbers, excluding NaN values if `ignore_nan` is True.

    or
    ------
    ValueError
        If the input iterable is empty and `empty` is set to 'raise'.

    Notes
    -----
    This function is similar to the built-in `mean` function in numpy, but it is designed to work
    with generators and other iterables that do not support indexing.
    """
    # ----

    # Create an iterator from the input iterable
    l = iter(l)

    # Initialize the accumulator and the count of elements
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        # If the iterable is empty, handle it based on the value of `empty`
        if empty == 'raise':
            return ValueError
        return empty

    # Sum the elements of the iterable and count them
    for n, v in enumerate(l, 2):
        acc += v

    # Calculate the mean, taking care of the case when there's only one element
    return acc / n if n > 1 else acc


# unit test cases
print(mean([np.nan, 2.5, np.nan, 5.0, 10.0], True, 0))
print(mean([], False, 'raise'))
print(mean([1, 2.5, 3, 4.5], False, 0))