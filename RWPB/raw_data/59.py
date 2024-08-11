import torch

def modify_logit_for_repetition_penalty(logits, prev_output_tokens, repetition_penalty=1.0):
    """
    Adjusts logits to penalize or favor the repetition of tokens that have already appeared,
    based on the provided repetition penalty.

    Args:
        logits (torch.Tensor): The logits from a language model, of shape (batch_size, vocab_size).
        prev_output_tokens (torch.Tensor): The indices of previously generated tokens for each sequence
                                           in the batch, of shape (batch_size, seq_len).
        repetition_penalty (float): The penalty factor for repetition. Values >1 penalize, values <1 favor
                                    repetition. The default value of 1.0 means no penalty.

    Returns:
        torch.Tensor: The adjusted logits.

    This function applies a repetition penalty as described in the paper:
    "CTRL: A Conditional Transformer Language Model for Controllable Generation".
    It decreases the probability of tokens that have appeared previously if the penalty is >1,
    and increases if <1 but not equal to zero.
    """
    # ----

    if repetition_penalty == 1.0:
        return logits  # No modification needed if penalty is 1

    # Gather scores of previously generated tokens from logits
    score = torch.gather(logits, 1, prev_output_tokens)

    # Apply repetition penalty:
    # If score < 0 (unlikely tokens), multiplying by a penalty > 1 will make them even less likely
    # If score > 0 (likely tokens), dividing by a penalty > 1 will make them less likely
    adjusted_score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

    # Scatter the adjusted scores back into the logits tensor at the positions of the previous tokens
    logits.scatter_(1, prev_output_tokens, adjusted_score)

    return logits


# unit test cases
print(modify_logit_for_repetition_penalty(torch.rand(1, 5), torch.tensor([[0, 1]])))
print(modify_logit_for_repetition_penalty(torch.tensor([[10.0, -5.0, 0.0, 3.0, -1.0]]), torch.tensor([[0, 3]]), 2.0))
print(modify_logit_for_repetition_penalty(torch.tensor([[2.0, -2.0, 4.0, -1.0, 0.0]]), torch.tensor([[0, 1, 3]]), 0.5))