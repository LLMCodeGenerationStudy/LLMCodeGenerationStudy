import torch
import torch.nn.functional as F


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    # ----
    
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


# unit test cases
masks = torch.rand(2, 3, 100, 200)
target_shape = (50, 100)
print(scale_masks(masks, target_shape))

masks = torch.rand(1, 1, 150, 300)
target_shape = (100, 100)
print(scale_masks(masks, target_shape))

masks = torch.rand(3, 3, 200, 200)
target_shape = (210, 210)
print(scale_masks(masks, target_shape))