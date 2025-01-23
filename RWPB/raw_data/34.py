import numpy as np

def pad_tokens(tokens: np.ndarray, context_window: int, pad_token: int) -> np.ndarray:
    """
    Pads or truncates a sequence of tokens to a specified length determined by the context window.

    Args:
        tokens (np.ndarray): An array of tokenized data, possibly with multiple sequences.
                             The shape should typically be (batch_size, sequence_length) or (sequence_length,).
        context_window (int): The target number of tokens in a sequence after padding or truncation.
                              The final sequence length will be `context_window + 1`.
        pad_token (int): The token used to pad shorter sequences.

    Returns:
        np.ndarray: An array where each sequence is exactly `context_window + 1` tokens long.

    This function modifies the length of sequences in a batch of tokenized data:
    - If a sequence is longer than `context_window + 1`, it gets truncated to this length.
    - If a sequence is shorter, it gets padded with `pad_token` until it reaches the length of `context_window + 1`.
    """
    # ----

    # Determine the target length for each sequence
    target_length = context_window + 1
    current_length = tokens.shape[-1]

    # Truncate if the current length is greater than the target length
    if current_length > target_length:
        tokens = tokens[..., :target_length]
    # Pad if the current length is less than the target length
    elif current_length < target_length:
        # Calculate the required number of padding elements
        padding_size = target_length - current_length
        # Create a padding array filled with the pad_token
        padding = np.full(tokens.shape[:-1] + (padding_size,), pad_token)
        # Concatenate the original tokens with the padding array
        tokens = np.concatenate([tokens, padding], axis=-1)

    # Ensure the final tokens array is of the correct shape
    assert tokens.shape[-1] == target_length, "Final token sequence length does not match the target length."

    return tokens


# unit test cases
print(pad_tokens(np.array([1, 2, 3, 4, 5, 6, 7]), 4, 0))
print(pad_tokens(np.array([1, 2, 3]), 5, 0))
print(pad_tokens(np.array([1, 2, 3, 4, 5]), 4, 0))