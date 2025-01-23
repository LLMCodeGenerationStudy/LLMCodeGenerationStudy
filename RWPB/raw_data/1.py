from typing import Optional

import torch
from matplotlib import cm
from torchtyping import TensorType

def apply_colormap(
    image: TensorType["bs":..., 1],
    cmap="viridis",
) -> TensorType["bs":..., "rgb":3]:
    """Convert single channel to a color image. 
    Args:
        image: Single channel image.
        cmap: Colormap for image.
    Returns:
        TensorType: Colored image
    """
    # ----

    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]


# unit test cases
a = torch.rand(2, 10, 10, 1)
b = torch.rand(1, 100, 100, 1)
c = torch.tensor([[[[0.0]], [[1.0]]]])

assert(apply_colormap(a) == apply_colormap(a))
assert(apply_colormap(b, 'plasma') == apply_colormap(b, 'plasma'))
assert(apply_colormap(c) == apply_colormap(c))