def get_crop_box(box, expand):
    """
    Expands a given bounding box by a specified factor, keeping the center of the box the same.

    Arguments:
    box : list or tuple
        The original bounding box specified as (x, y, x1, y1) where (x, y) is the top-left corner and
        (x1, y1) is the bottom-right corner.
    expand : float
        The factor by which to expand the bounding box. For example, a factor of 1.5 will increase
        the size of the box by 50%.

    Returns:
    tuple
        A tuple containing the new expanded bounding box as a list [new_x, new_y, new_x1, new_y1] and
        the half-size of the new box `s`.
    """
    # ----
    
    # Extract the coordinates from the original bounding box
    x, y, x1, y1 = box
    # Calculate the center of the original box
    x_c, y_c = (x + x1) // 2, (y + y1) // 2
    # Calculate width and height of the original box
    w, h = x1 - x, y1 - y
    # Calculate half of the size of the expanded box
    s = int(max(w, h) // 2 * expand)
    # Define the new expanded bounding box based on the center and size
    crop_box = [x_c - s, y_c - s, x_c + s, y_c + s]

    return crop_box, s


# unit test cases
print(get_crop_box(box = (100, 100, 200, 200), expand = 1.5))
print(get_crop_box(box = (50, 50, 150, 150), expand = 1.0))
print(get_crop_box(box = (-100, -100, 100, 100), expand = 2.0))