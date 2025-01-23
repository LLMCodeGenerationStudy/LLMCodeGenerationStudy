import math
import numpy as np
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN



def round_nearest_multiple(number, a, direction='standard'):
    """
    Rounds a number to the nearest multiple of a. The returned number will have the
    same precision as a.

    Args:
        - number: float, the number to be rounded.
        - a: float, the multiple to round to.
        - direction: str, optional, specifying the rounding direction ('standard', 'down', or 'up').
                     Default is 'standard'.

    Returns:
        rounded_number: float, the rounded number to the nearest multiple of a.
    """
    # ----
    
    if direction == 'down':
        return round(math.floor(number / a) * a, -int(math.floor(math.log10(a))))
    elif direction == 'up':
        return round(math.ceil(number / a) * a, -int(math.floor(math.log10(a))))
    elif direction == 'standard':
        return round(number, -int(math.floor(math.log10(a))))



# unit test cases
number = 14.7
a = 5
print(round_nearest_multiple(number, a))

number = 9.8
a = 2
direction = 'down'
print(round_nearest_multiple(number, a, direction))

number = 7.1
a = 0.5
direction = 'up'
print(round_nearest_multiple(number, a, direction))