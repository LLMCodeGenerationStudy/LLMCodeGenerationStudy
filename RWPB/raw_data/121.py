def isprime(n):
    """
    Check if a number is a prime number.

    Args:
        - n: int, number to be checked for primality

    Returns:
        - bool: True if n is prime, False otherwise
    """
    # ----
    
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    max = n**0.5+1
    i = 3
    
    while i <= max:
        if n % i == 0:
            return False
        i+=2
    return True


# unit test cases
n = 2
print(isprime(n))

n = 4
print(isprime(n))

n = 29
print(isprime(n))