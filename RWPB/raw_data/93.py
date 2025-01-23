MAX_VAL = (1 << 256) - 1

# does not revert on overflow
def unsafeSub(x, y):
    """
    Subtract two integers, ignoring overflow.
    Args:
        - x: int, the minuend
        - y: int, the subtrahend
    Returns:
        result: int, the difference of x and y, with overflow ignored
    """
    # ----
    
    return (x - y) & MAX_VAL


# unit test cases

x = 100
y = 50
print(unsafeSub(x, y))

x = 0
y = 100
print(unsafeSub(x, y))

x = (1 << 256) - 1
y = 1
print(unsafeSub(x, y))