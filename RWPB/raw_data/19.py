def rank_accuracy(preds, references):
    '''
    Computes the accuracy of ranking predictions compared to reference rankings.

    Parameters
    ----------
    preds : list
        A list of numeric predictions, typically confidence scores, where higher
        values represent stronger predictions.
    references : list
        A list of actual values or labels, with the same ordering principle as
        'preds', used as a ground truth for accuracy calculation.

    Return
    ------
    float
        The proportion of correct ranking relations in 'preds' relative to 'references'.

    This function first generates all possible pairs of different predictions and checks if
    the ranking (greater or lesser) of the first element relative to the second in each pair
    is correct based on the corresponding reference values.
    '''
    # ----

    # Initialize an empty list to store the pairs and their comparison results
    pairs = []

    # Generate all pairwise comparisons from preds except self-comparisons
    for i, pred in enumerate(preds):
        for j, pred2 in enumerate(preds):
            if i == j:
                continue  # Skip comparing the same elements
            pairs.append((i, j, pred > pred2))  # Store index and comparison result

    # Count how many of these predicted comparisons are correct according to the references
    correct = 0
    for i, ref in enumerate(references):
        for j, ref2 in enumerate(references):
            if i == j:
                continue  # Skip self-comparisons
            if (i, j, ref > ref2) in pairs:
                correct += 1  # Increment if the prediction comparison is correct

    # Calculate the accuracy as the ratio of correct predictions to total comparisons
    return correct / len(pairs) if pairs else 0  # Handle the case where pairs might be empty


# unit test cases
print(rank_accuracy(preds = [0.9, 0.1, 0.5], references = [3, 1, 2]))
print(rank_accuracy(preds = [0.5, 0.9, 0.1], references = [3, 1, 2]))
print(rank_accuracy(preds = [0.5, 0.5, 0.1], references = [2, 2, 1]))