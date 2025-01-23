import itertools as it
import numpy as np


def merge_dicts_recursively(*dicts):
    """
    Create a dictionary whose keys are the union of all input dictionaries. 
    The value for each key is based on the first dictionary in the list with that key.
    Dictionaries later in the list have higher priority.
    When values are dictionaries, it is applied recursively.
    Args:
        - *dicts: tuple of dicts, the dictionaries to merge
    Returns:
        result: dict, the merged dictionary
    """
    # ----
    
    result = dict()
    all_items = it.chain(*[d.items() for d in dicts])
    for key, value in all_items:
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_recursively(result[key], value)
        else:
            result[key] = value
    return result


# unit test cases
print(merge_dicts_recursively({'a': 1, 'b': 2}, {'b': 3, 'c': 4}))
print(merge_dicts_recursively({'a': 1, 'nested': {'x': 10, 'y': 20}}, {'b': 2, 'nested': {'y': 30, 'z': 40}}))
print(merge_dicts_recursively({'a': [1, 2, 3], 'b': 'hello'}, {'a': 'not a list', 'b': {'nested': 'dict'}}))