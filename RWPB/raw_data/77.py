def is_power_of_two(n):
    """
    chat.openai.com/chat
    Return True if n is a power of 2, otherwise return False.

    The function is_power_of_two takes an integer n as input and returns True if n is a power of 2, otherwise it returns False.
    The function works by first checking if n is less than or equal to 0. If n is less than or equal to 0, it can't be a power of 2, so the function returns False.
    If n is greater than 0, the function checks whether n is a power of 2 by using a bitwise AND operation between n and n-1. If n is a power of 2, then it will have only one bit set to 1 in its binary representation. When we subtract 1 from a power of 2, all the bits to the right of that bit become 1, and the bit itself becomes 0. So, when we perform a bitwise AND between n and n-1, we get 0 if n is a power of 2, and a non-zero value otherwise.
    Thus, if the result of the bitwise AND operation is 0, then n is a power of 2 and the function returns True. Otherwise, the function returns False.

    """
    # ----
    
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


# unit test cases
print(is_power_of_two(0))
print(is_power_of_two(-16))
print(is_power_of_two(16))
print(is_power_of_two(1024))
print(is_power_of_two(18))
print(is_power_of_two(999))