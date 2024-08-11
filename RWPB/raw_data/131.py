import torch
import numpy as np

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # ----

    # NOTE: need float32 to get accurate iou values
    box1 = torch.as_tensor(box1, dtype=torch.float32)
    box2 = torch.as_tensor(box2, dtype=torch.float32)
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

# unit test cases
box1 = np.array([[0, 0, 2, 2]])
box2 = np.array([[1, 1, 3, 3]])
print(box_iou(box1, box2))

box1 = np.array([[0, 0, 1, 1]])
box2 = np.array([[2, 2, 3, 3]])
print(box_iou(box1, box2))

box1 = np.array([[0, 0, 3, 3], [4, 4, 5, 5]])
box2 = np.array([[1, 1, 4, 4], [0, 0, 2, 2]])
print(box_iou(box1, box2))

box1 = np.array([[0, 0, 3, 3], [4, 4, 5, 5]])
box2 = np.array([[1, 1, 4, 4], [0, 0, 2, 2]])
print(box_iou(box1, box2, 2e-5))

box1 = np.array([[1, 3, 3, 6], [7, 7, 2, 6]])
box2 = np.array([[9, 3, 5, 4], [0, 1, 3, 8]])
print(box_iou(box1, box2))
