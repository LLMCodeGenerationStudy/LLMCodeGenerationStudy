import torch


def top_p_sample(prob_dist: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Applies top-p filtering to a probability distribution to focus sampling on a subset of likely outputs.

    Args:
        prob_dist (torch.Tensor): The original probability distribution from which to sample, shape (batch_size, vocab_size).
        top_p (float): The cumulative probability threshold used for filtering. Probabilities that add up to `top_p` are kept.

    Returns:
        torch.Tensor: The modified probability distribution after applying top-p filtering.

    This method is used to reduce the sample space to the most probable tokens whose cumulative probability
    exceeds the threshold `top_p`. It improves sampling outcomes by eliminating highly improbable options.
    """
    # ----

    # Sort the probabilities and their corresponding indices in descending order
    sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True, dim=-1)

    # Compute the cumulative sum of the sorted probabilities
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)  # Shape: (batch_size, vocab_size)

    # Create a boolean mask where probabilities that exceed the threshold are set to True
    sorted_indices_to_remove = cum_sum_probs > top_p

    # Shift the indices to the right to include the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0  # Ensure the first token is always included

    # Convert the mask to boolean and apply it to sorted_probs by setting removed probs to 0
    sorted_probs[sorted_indices_to_remove] = 0

    # Reverse the sorting process to reconstruct the original order of probabilities
    reversed_indices = torch.argsort(sorted_indices)
    prob_dist = torch.gather(sorted_probs, -1, reversed_indices)

    # Normalize the probabilities to ensure they sum to 1
    prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)

    return prob_dist


# unit test cases
print(top_p_sample(torch.tensor([[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]]), 0.3))
print(top_p_sample(torch.tensor([[0.05, 0.15, 0.2, 0.3, 0.25, 0.05],
                          [0.4, 0.1, 0.1, 0.1, 0.1, 0.2],
                          [0.25, 0.25, 0.25, 0.15, 0.05, 0.05]]), 0.3))
print(top_p_sample(torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]), 0.5))