import numpy as np
import cv2

def hole_fill(img):
    """
    Performs hole filling on an image using morphological operations.

    Args:
        img (np.array): A grayscale image array where holes are to be filled.

    Returns:
        np.array: An image array with holes filled.

    This function applies the flood fill algorithm to identify and fill holes in the input image.
    The process involves padding the image, copying it for result calculation, and applying
    bitwise operations to fill the holes based on the flood filled mask.
    """
    # ----

    # Pad the image to avoid boundary issues during flood fill
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
    img_copy = img.copy()  # Create a copy of the image for later use

    # Create a mask for flood fill operation, size needs to be 2 pixels bigger than the image
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    # Perform flood fill from the top-left corner (0,0) with the new value 255
    cv2.floodFill(img, mask, (0, 0), 255)

    # Create inverse of the flood filled image
    img_inverse = cv2.bitwise_not(img)

    # Combine the original copy and the inverse to fill the holes
    dst = cv2.bitwise_or(img_copy, img_inverse)

    return dst



# unit test cases
print(hole_fill(np.zeros((10, 10), dtype=np.uint8)))

img_large_hole = np.ones((10, 10), dtype=np.uint8) * 255
img_large_hole[3:7, 3:7] = 0
print(hole_fill(img_large_hole))

# Create an image with both small holes on the edges and small holes scattered in the middle of the image
img_multiple_holes = np.ones((10, 10), dtype=np.uint8) * 255
img_multiple_holes[1, 1] = 0 # small holes in corners
img_multiple_holes[1, 8] = 0 # small holes in the other corner
img_multiple_holes[5, 5] = 0 # small holes in the centre
img_multiple_holes[8, 1] = 0 # edge holes
img_multiple_holes[8, 8] = 0 # holes at the edges

print(hole_fill(img_multiple_holes))
