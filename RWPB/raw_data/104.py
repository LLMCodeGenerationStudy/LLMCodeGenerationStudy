MAX_VAL = (1 << 256) - 1


# does not overflow
def mulMod(x, y, z):
    """
    Compute (x * y) % z without overflow.
    Args:
        - x: int, multiplicand
        - y: int, multiplier
        - z: int, modulus
    Returns:
        result: int, the result of (x * y) % z
    """
    # ----
    
    return (x * y) % z


# unit test cases
print(mulMod(10, 20, 7))
print(mulMod(3214, 34, 423))

large_x = (1 << 255)
large_y = (1 << 255)
large_z = (1 << 256) - 1
print(mulMod(large_x, large_y, large_z))

print(mulMod(0, 10, 3))
print(mulMod(10, 0, 3))
print(mulMod(10, 20, 1))