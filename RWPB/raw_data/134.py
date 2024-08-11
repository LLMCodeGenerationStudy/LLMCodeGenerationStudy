def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    # ----
    
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


# unit test cases
original_size = (1920, 1080)
possible_resolutions = [(800, 600), (1920, 1080), (2560, 1440)]
print(select_best_resolution(original_size, possible_resolutions))

original_size = (1024, 768)
possible_resolutions = [(800, 600), (1600, 1200), (1280, 960)]
print(select_best_resolution(original_size, possible_resolutions))


original_size = (1280, 720)
possible_resolutions = [(1024, 768), (1920, 1080), (1600, 900)]
print(select_best_resolution(original_size, possible_resolutions))


