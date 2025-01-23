import torch

def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    # ----

    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


# unit test cases
mask1 = torch.tensor([
    [1, 0, 1, 0],  # Object 1
    [0, 1, 0, 1]   # Object 2
])
mask2 = torch.tensor([
    [1, 0, 0, 1],  # Prediction 1
    [0, 1, 1, 0],  # Prediction 2
    [1, 1, 1, 1]   # Prediction 3
])
print(mask_iou(mask1, mask2))

mask1 = torch.tensor([
    [1, 0, 0, 0],  # Object 1
    [0, 1, 0, 0]   # Object 2
])
mask2 = torch.tensor([
    [0, 0, 1, 0],  # Prediction 1
    [0, 0, 0, 1]   # Prediction 2
])
print(mask_iou(mask1, mask2))


mask1 = torch.tensor([
    [1, 1, 0, 0, 1, 1],  # Object 1
    [0, 1, 1, 1, 0, 0],  # Object 2
    [1, 0, 1, 0, 1, 0]   # Object 3
])
mask2 = torch.tensor([
    [1, 0, 0, 1, 1, 1],  # Prediction 1
    [0, 1, 1, 0, 0, 1]   # Prediction 2
])
print(mask_iou(mask1, mask2))