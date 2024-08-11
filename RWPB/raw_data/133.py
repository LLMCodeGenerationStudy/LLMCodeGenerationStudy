from PIL import Image

def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    # ----
    
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


# unit test cases
img = Image.new('RGB', (256, 256))
patch_size = 64
result = divide_to_patches(img, patch_size)
print(result)
assert(len(result) == 16)
assert(all(patch.size == (64, 64) for patch in result))

img = Image.new('RGB', (300, 150))
patch_size = 50
result = divide_to_patches(img, patch_size)
print(result)
assert(len(result) == 18)
assert(all(patch.size == (50, 50) for patch in result))

img = Image.new('RGB', (30, 30))
patch_size = 50
result = divide_to_patches(img, patch_size)
print(result)
assert(len(result) == 1)
assert(result[0].size == (50, 50))
