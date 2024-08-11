import torch

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens the scores and labels from a batch of predictions, removing entries with a specified ignore label.

    Parameters
    ----------
    scores : torch.Tensor
        The tensor containing the prediction scores for each item in the batch.
    labels : torch.Tensor
        The tensor containing the true labels for each item in the batch.
    ignore : int or float, optional
        The label to ignore during the flattening process. If None, all labels are kept.

    Returns
    -------
    vscores : torch.Tensor
        A flattened tensor containing the prediction scores without the ignored labels.
    vlabels : torch.Tensor
        A flattened tensor containing the true labels without the ignored labels.

    Notes
    -----
    This function is typically used in the binary classification case where 'scores' and 'labels'
    are expected to be tensors of shape (batch_size, ...) and the function flattens them to a 1D
    tensor (-1,). If the 'ignore' parameter is provided, it filters out the corresponding labels
    and scores from the result.
    """
    # ----

    # Flatten the scores and labels to 1D tensors
    scores = scores.view(-1)
    labels = labels.view(-1)

    # If there's no ignore label, return the flattened scores and labels
    if ignore is None:
        return scores, labels

    # Create a boolean mask for entries that do not have the ignore label
    valid = (labels != ignore)

    # Apply the mask to filter out the ignored labels and return the valid scores and labels
    vscores = scores[valid]
    vlabels = labels[valid]

    return vscores, vlabels



# unit test cases
print(flatten_binary_scores(torch.tensor([[0.5, 0.2], [0.7, 0.4]]), torch.tensor([[1, 1], [0, 0]]), 2))
print(flatten_binary_scores(torch.tensor([[0.1, 0.3], [0.5, 0.7]]), torch.tensor([[1, 2], [1, 2]]), 2))
print(flatten_binary_scores(torch.tensor([[0.2, 0.4], [0.6, 0.8]]), torch.tensor([[0, 1], [0, 1]]), None))