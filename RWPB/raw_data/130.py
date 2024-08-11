import numpy as np

def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
    """
    # ----

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


# unit test cases
box1 = np.array([[0, 0, 2, 2]])
box2 = np.array([[1, 1, 3, 3]])
iou = False
print(bbox_ioa(box1, box2, iou))

box1 = np.array([[0, 0, 1, 1]])
box2 = np.array([[2, 2, 3, 3]])
iou = False
print(bbox_ioa(box1, box2, iou))

box1 = np.array([[0, 0, 3, 3], [4, 4, 5, 5]])
box2 = np.array([[1, 1, 4, 4], [0, 0, 2, 2]])
iou = True
print(bbox_ioa(box1, box2, iou))

box1 = np.array([[0, 0, 3, 3], [4, 4, 5, 5]])
box2 = np.array([[1, 1, 4, 4], [0, 0, 2, 2]])
iou = True
print(bbox_ioa(box1, box2, iou, 2e-5))

box1 = np.array([[1, 3, 3, 6], [7, 7, 2, 6]])
box2 = np.array([[9, 3, 5, 4], [0, 1, 3, 8]])
iou = True
print(bbox_ioa(box1, box2, iou))