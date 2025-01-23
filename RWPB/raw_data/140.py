from typing import Callable, Dict, Tuple, Any
import inspect
from functools import partial

def get_args_for_func(func: Callable, params: Dict) -> Tuple[Callable, Dict[str, Any]]:
    """
    Filters keyword arguments from a dictionary based on a function's parameter list and creates a partially
    applied function that already includes the matched keyword arguments.

    Parameters:
        func (Callable): The function for which the keyword arguments need to be matched.
        params (Dict): A dictionary containing potential keyword arguments to pass to the function.

    Returns:
        Tuple[Callable, Dict[str, Any]]:
            A tuple containing:
            - A partial function with pre-applied keyword arguments that are valid for the specified function.
            - A dictionary of the keyword arguments that were applicable to the function.

    Example:
        input_dict = {"a": "foo", "b": "bar"}
        def example_func(a: str):
            return a
        func, kwargs = get_args_for_func(example_func, input_dict)
        # func can now be called as func() and it would execute example_func(a="foo")
        # kwargs will be {'a': 'foo'}
    """
    # ----
    
    # Extracts function argument names using inspect
    _kwargs = {k: v for k, v in params.items() if k in inspect.getfullargspec(func).args}
    # Creates a partial function that pre-applies these keyword arguments
    return (
        partial(func, **_kwargs),
        _kwargs,
    )


# unit test cases
def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

def greet(name, msg="Hello"):
    return f"{msg}, {name}"
func, kwargs = get_args_for_func(add, {'x': 10, 'y': 5})
assert(func() == 15)
assert(kwargs == {'x': 10, 'y': 5})


func, kwargs = get_args_for_func(multiply, {'x': 2, 'y': 8, 'z': 15})
assert(func() == 16)
assert(kwargs == {'x': 2, 'y': 8})


func, kwargs = get_args_for_func(greet, {'name': 'Alice'})
assert(func() == "Hello, Alice")
assert(kwargs == {'name': 'Alice'})