import inspect
from typing import Callable, List

def extract_code_from_funct(funct: Callable) -> List[str]:
    """
    Extracts the body of a Python function, removing the function definition and return statement, and adjusting indentation.

    Args:
        funct (Callable): The function from which code is extracted.

    Returns:
        List[str]: A list of strings, where each string is a line of code from the function's body, adjusted for indentation.

    This function utilizes Python's `inspect` module to retrieve the source code of a given function. It then processes
    this code to remove the function's signature and any return statement at the end, adjusting the indentation of the
    remaining lines to align with the leftmost edge.

    Example:
        def example_function(x):
            result = x + 1
            return result

        >>> extract_code_from_funct(example_function)
        ['result = x + 1']
    """
    # ----

    # Retrieve the source code of the function
    source_code = inspect.getsource(funct)

    # Split the source code into individual lines and remove the first line (function definition)
    source_code_lines = source_code.splitlines()[1:]

    # Calculate the number of characters of indentation in the first line of the function body
    nident = len(source_code_lines[0]) - len(source_code_lines[0].lstrip())

    # Adjust each line to remove the initial indentation and exclude the last line (usually a return statement)
    return [line[nident:] for line in source_code_lines[:-1]]


# unit test cases
def sample_function(x):
    y = x + 10
    z = y * 2
    return z
print(extract_code_from_funct(sample_function))
def empty_function():
    pass
print(extract_code_from_funct(empty_function))
def complex_function(x):
    if x > 0:
        for i in range(x):
            print(i)
    else:
        while x < 0:
            x += 1
            print(x)
    return x
print(extract_code_from_funct(complex_function))