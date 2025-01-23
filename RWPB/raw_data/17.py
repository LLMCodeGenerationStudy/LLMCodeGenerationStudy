import numpy as np

def generate_linear_parameters(start, stop, num):
    """
    Generates a list of linearly spaced parameters between start and stop.

    Parameters
    ----------
    start : float
        The starting value of the sequence.
    stop : float
        The ending value of the sequence.
    num : int
        The number of values to generate.

    Returns
    -------
    List[float]
        A list containing 'num' values, linearly spaced between 'start' and 'stop'.

    Notes
    -----
    This function uses numpy's linspace function to create an array of 'num' values evenly
    distributed over the interval [start, stop]. The resulting array is then converted to a list.
    """
    # ----

    # Generate linearly spaced values between 'start' and 'stop'
    parames = list(
        np.linspace(
            start=start,
            stop=stop,
            num=num,
        )
    )
    return parames


# unit test cases
print(generate_linear_parameters(start=0, stop=10, num=5))
print(generate_linear_parameters(start=0, stop=10, num=1))
print(generate_linear_parameters(start=0, stop=10, num=0))