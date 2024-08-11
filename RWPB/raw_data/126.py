import torch

def crop_mask(masks, boxes):
    """
    Crops each mask in a batch of masks to the specified bounding boxes.

    Args:
        masks (torch.Tensor): A tensor of shape [n, h, w] where 'n' is the number of masks, and 'h' and 'w' are the height and width of the masks.
        boxes (torch.Tensor): A tensor of shape [n, 4] representing the bounding boxes for each mask. The boxes are in the format [x1, y1, x2, y2].

    Returns:
        torch.Tensor: A tensor of the same shape as 'masks' but with each mask cropped to its corresponding bounding box.
    """
    # ----
    
    _, h, w = masks.shape  # Get the dimensions of the masks
    # Split the bounding box tensor into x1, y1, x2, y2 components. Each is expanded to have shape [n, 1, 1].
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, dim=1)
    # Generate tensors representing the indices of each column and each row in the image, and broadcast them to match dimensions
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # row indices, shape [1, 1, w]
    c = torch.arange(h, device=masks.device, dtype=y1.dtype)[None, :, None]  # column indices, shape [1, h, 1]

    # Generate a mask by comparing indices with the bounding box coordinates
    # The comparison results in a binary mask for each bounding box, which when multiplied with the original mask, crops it.
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

# unit test cases
masks = torch.ones((3, 5, 5))
boxes = torch.tensor([[1, 1, 4, 4], [0, 0, 3, 3], [2, 2, 5, 5]])
result = crop_mask(masks, boxes)
print(result)

masks = torch.ones((2, 5, 5))
boxes = torch.tensor([[0, 0, 5, 5], [-1, -1, 6, 6]])
result = crop_mask(masks, boxes)
print(result)

masks = torch.tensor([
    [[1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1]],
    [[0, 1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 1],
     [1, 0, 1, 0, 1, 0]]
])
boxes = torch.tensor([[1, 1, 3, 3], [3, 3, 6, 6]])
result = crop_mask(masks, boxes)
print(result)