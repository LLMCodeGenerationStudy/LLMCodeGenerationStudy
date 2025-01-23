MAX_VAL = (1 << 256) - 1

# does not revert on overflow
def unsafeMul(x, y):
    """
    Multiply two integers, ignoring overflow.
    Args:
        - x: int, first operand
        - y: int, second operand
    Returns:
        result: int, the product of x and y, with overflow ignored
    """
    # ----
    
    return (x * y) & MAX_VAL


# unit test cases
print(unsafeMul(15, 10))
print(unsafeMul((1 << 256) - 1, 1))
print(unsafeMul(pow(2, 128), pow(2, 129)))