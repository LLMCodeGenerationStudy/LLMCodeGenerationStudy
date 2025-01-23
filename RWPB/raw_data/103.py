MAX_VAL = (1 << 256) - 1



# does not revert on overflow
def unsafeAdd(x, y):
    """
    Add two integers, ignoring overflow.
    Args:
        - x: int, first operand
        - y: int, second operand
    Returns:
        result: int, the sum of x and y, with overflow ignored
    """
    # ----
    
    return (x + y) & MAX_VAL


# unit test cases
print(unsafeAdd(10, 20))
print(unsafeAdd(MAX_VAL - 5, 10))
print(unsafeAdd(MAX_VAL // 2, (MAX_VAL // 2) + 1))