import math
import cv2
import numpy as np

def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """
    Resize the images in the sample to ensure they meet the minimum specified size while maintaining the aspect ratio.

    Args:
        sample (dict): A dictionary containing the images to be resized. The keys are:
            - "image": The main image to be resized.
            - "disparity": The disparity map associated with the image.
            - "mask": A binary mask associated with the image.
        size (tuple): The minimum desired size (height, width) for the images.
        image_interpolation_method (int, optional): Interpolation method to be used for resizing the main image. 
                                                    Defaults to cv2.INTER_AREA.

    Returns:
        tuple: The new size (height, width) of the resized images.
    """
    # ----
    
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)


# unit test cases
sample = {
    "image": np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
    "disparity": np.random.randint(0, 256, (100, 100), dtype=np.uint8),
    "mask": np.random.randint(0, 2, (100, 100), dtype=np.bool_)
}
size = (200, 300)
print(apply_min_size(sample, size))

sample = {
    "image": np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8),
    "disparity": np.random.randint(0, 256, (300, 300), dtype=np.uint8),
    "mask": np.random.randint(0, 2, (300, 300), dtype=np.bool_)
}
size = (200, 200)
print(apply_min_size(sample, size))

sample = {
    "image": np.random.randint(0, 256, (150, 100, 3), dtype=np.uint8),
    "disparity": np.random.randint(0, 256, (150, 100), dtype=np.uint8),
    "mask": np.random.randint(0, 2, (150, 100), dtype=np.bool_)
}
size = (300, 450)
print(apply_min_size(sample, size))