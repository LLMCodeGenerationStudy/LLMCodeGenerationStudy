import math
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN


def round(number, num_decimal_places, leave_int=False):
    '''
    Purpose: round a number to the specified number of decimal places. Existing 
        round functions may not round correctly so that's why I built my own.
    Args:
        - number: float, number to round
        - num_decimal_places: int, number of decimal places to round number to
        - leave_int: bool, whether to leave integers unchanged or convert them to floats

    Returns:
        - float: rounded number
    '''
    # ----
    
    if leave_int and int(number) == number:
        return number
    decimal_str = '1.'
    for decimal_place in range(num_decimal_places):
        decimal_str += '1'
    return float(Decimal(str(number)).quantize(Decimal(decimal_str), rounding=ROUND_HALF_UP))


# unit test cases
print(round(3.14159265, 4))
print(round(200, 2, leave_int=True))
print(round(2.789, 0))
