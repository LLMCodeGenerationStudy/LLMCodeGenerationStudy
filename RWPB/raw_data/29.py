import torch

def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    Computes the Intersection over Union (IoU) for the foreground class in binary segmentation.

    Args:
        preds (Tensor): Predictions from the model, where 1 indicates foreground and 0 indicates background.
        labels (Tensor): Ground truth binary masks corresponding to the predictions.
        EMPTY (float, optional): The value to return if there is no union between predicted and ground truth pixels.
                                 Default is 1.0.
        ignore (int, optional): Label value to ignore during IoU calculation. This is useful for ignoring certain
                                pixels in evaluation, such as boundary or undefined areas.
        per_image (bool, optional): If True, calculates IoU independently for each image and then averages them.
                                    If False, calculates IoU across the entire batch as a whole. Default is True.

    Returns:
        float: The IoU score multiplied by 100 to convert it to a percentage.

    This function calculates the IoU score for the foreground class, accounting for any ignore labels
    and handling both individual image evaluations and batch-wide evaluations. The IoU is calculated
    by determining the intersection and union of the predicted and actual foreground areas, and handling
    cases where the union might be zero.
    """
    # ----

    # Helper function to calculate mean
    def mean(l):
        return sum(l) / len(l) if l else EMPTY

    # Handle batch-wide calculation by wrapping predictions and labels in tuples if per_image is False
    if not per_image:
        preds, labels = (preds,), (labels,)

    ious = []
    for pred, label in zip(preds, labels):
        # Calculate intersection and union for each image
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()

        # Calculate IoU, handling cases with no union
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)

        ious.append(iou)

    # Calculate the mean IoU across images if per_image, or for the batch
    iou = mean(ious)
    return 100 * iou  # Return IoU as a percentage


# unit test cases
print(iou_binary(torch.tensor([[1,0,1,0],[0,1,0,1]]), torch.tensor([[1,0,1,0],[0,1,0,1]]), 1.0, None, True))
print(iou_binary(torch.tensor([[0,0,0,0],[1,1,1,1]]), torch.tensor([[0,0,0,0],[0,0,0,0]]), 0.5, 1, True))
print(iou_binary(torch.tensor([[1,1,0,0],[0,0,1,1]]), torch.tensor([[1,0,0,1],[1,1,0,0]]), 1.0, 0, False))