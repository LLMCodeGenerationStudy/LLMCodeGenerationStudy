from typing import List

def pad_to_max_seq_length(ls: List[int], max_seq_length: int, pad_idx: int=0, pad_right: bool=True, check: bool=True):
    """Apply padding to an input sequence.

    Args:
        ls: List[int]: sequence to pad.
        max_seq_length: int: max length up to which to apply padding.
        pad_idx :int: element to use for padding.
        pad_right: bool: True if padding is applied to right side of sequence, False to pad on left side.
        check: bool: True if result length should be checked as under the max sequence length.
        if result length is not equal the max sequence length, please return ValueError

    Returns:
        Sequence with specified padding applied.

    """
    # ----

    padding = [pad_idx] * (max_seq_length - len(ls))
    if pad_right:
        result = ls + padding
    else:
        result = padding + ls

    if check:
        if len(result) != max_seq_length:
            return ValueError
    return result



# unit test cases
print(pad_to_max_seq_length([1, 2, 3], 5))          # [1, 2, 3, 0, 0]
print(pad_to_max_seq_length([1, 2, 3], 5, pad_idx=-1))  # [1, 2, 3, -1, -1]
print(pad_to_max_seq_length([1, 2, 3], 5, pad_right=False))  # [0, 0, 1, 2, 3]
print(pad_to_max_seq_length([1, 2, 3], 3))          # [1, 2, 3]
print(pad_to_max_seq_length([1, 2, 3], 2))          # ValueError
