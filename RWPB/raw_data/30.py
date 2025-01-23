import numpy as np

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calculates the confusion matrix from the ground truth and predicted labels.
    If the length of gt_label is not equal to the length of pred_label, please return ValueError 

    Args:
        gt_label (np.array): An array of ground truth labels.
        pred_label (np.array): An array of predicted labels from the model.
        num_classes (int): The number of classes in the dataset.

    Returns:
        np.array: A 2D array (matrix) of shape (num_classes, num_classes) representing the confusion matrix,
                  where each row corresponds to the true classes and each column to the predicted classes.

    The function calculates a confusion matrix which helps in evaluating the accuracy of a classification model.
    It does so by cross-tabulating each class label with its corresponding prediction, thus providing insight
    into misclassifications.
    """
    # ----

    if gt_label.shape[-1] != pred_label.shape[-1]:
        return ValueError
    
    # Compute the linear index for each element of the confusion matrix
    index = (gt_label * num_classes + pred_label).astype('int32')

    # Count the occurrence of each index
    label_count = np.bincount(index, minlength=num_classes*num_classes)

    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Fill the confusion matrix with counts
    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


# unit test cases
print(get_confusion_matrix(np.array([0, 1, 2, 1, 0]), np.array([0, 2, 2, 1, 0]), 3))
print(get_confusion_matrix(np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 1))
print(get_confusion_matrix(np.array([0, 1, 2, 3]), np.array([0, 1, 2]), 4))