def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """
    Flattens a nested dictionary by collapsing its nested structure into a single layer dictionary with keys
    representing paths, separated by a specified separator.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base path for keys in the nested dictionary. Initially left empty.
        sep (str): The separator to use when concatenating nested keys.

    Returns:
        dict: A flattened version of the dictionary where all nested keys are transformed into single-layered keys.

    Example:
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
        >>> print(flatten_dict(dct))
        {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    # ----

    items = []
    for k, v in d.items():
        # Generate the new key by concatenating the current key with the parent key using the separator
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursively flatten the dictionary and extend the list items
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            # Add the key-value pair to the items list
            items.append((new_key, v))
    return dict(items)


# unit test cases
print(flatten_dict({"a": {"b": {"c": {"d": 1}}}, "e": {"f": 2, "g": {"h": 3}}, "i": 4}))
print(flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3}, sep='-'))
print(flatten_dict({"a": {}, "b": {"c": {}}, "d": 5}))