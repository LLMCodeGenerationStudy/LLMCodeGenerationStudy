from typing import Optional, Sequence

import torch

def window_partition(x: torch.Tensor, window_size: Sequence[int]):
    """
    Partition an image feature map into smaller windows in an ANE (Apple Neural Engine) friendly manner,
    avoiding the use of 6D tensors.

    :param x: A 4D tensor representing the image feature map with shape (batch_size, H, W, C), where:
              - batch_size (B) is the number of images in the batch,
              - H is the height of the feature map,
              - W is the width of the feature map,
              - C is the number of channels.
    :param window_size: A sequence of two integers (win_h, win_w) specifying the height and width of the windows.
    :returns: A 4D tensor of partitioned feature map windows with shape (batch_size * num_windows, win_h, win_w, C), where:
              - num_windows is the total number of windows per image (calculated as (H / win_h) * (W / win_w)),
              - win_h is the window height,
              - win_w is the window width,
              - C is the number of channels.
    :rtype: torch.Tensor

    The process of partitioning is as follows:
    1. Reshape the input tensor to split the height dimension (H) into smaller chunks of size win_h,
       resulting in shape (B, H // win_h, win_h, W, C).
    2. Flatten the batch and height dimensions together to shape (B * H // win_h, win_h, W, C).
    3. Reshape again to split the width dimension (W) into smaller chunks of size win_w,
       resulting in shape (B * H // win_h, win_h, W // win_w, win_w, C).
    4. Permute the dimensions to arrange the tensor into shape (B * H // win_h, W // win_w, win_h, win_w, C).
    5. Flatten the batch and width dimensions together to produce the final shape (batch_size * num_windows, win_h, win_w, C).
    """
    # ----
    
    B, H, W, C = x.shape
    # example partition process: 1, 12, 16, 160 -> 1, 2, 6, 16, 160 -> 2, 6, 16, 160 -> 2, 6, 2, 8, 160 -> ...
    x = x.reshape(
        (B, H // window_size[0], window_size[0], W, C)
    )  # B, H//w_size, w_size, W, C
    x = x.reshape(
        (B * H // window_size[0], window_size[0], W, C)
    )  # B * H // w_size, w_size, W, C
    x = x.reshape(
        (
            B * H // window_size[0],
            window_size[0],
            W // window_size[1],
            window_size[1],
            -1,
        )
    )
    x = x.permute((0, 2, 1, 3, 4))
    windows = x.reshape((-1, window_size[0], window_size[1], C))
    return windows


# unit test cases
a = torch.rand((2, 8, 8, 3))
ws = (4, 4)
print(a, ws)

a = torch.rand((1, 10, 10, 5))
ws = (5, 2)
print(a, ws)

a = torch.rand((1, 8, 8, 3))
ws = (3, 3)
print(a, ws)