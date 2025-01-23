from math import factorial


def getLambertCoefs(numOfLambertCoefs):
    """
    Compute the Lambert coefficients for a given number of coefficients.
    Args:
        - numOfLambertCoefs: int, the number of coefficients
    Returns:
        lambert_coefs: list, the computed Lambert coefficients
    """
    # ----
    
    maxFactorial = factorial(numOfLambertCoefs-1)
    return [maxFactorial*i**(i-1)//factorial(i) for i in range(1,numOfLambertCoefs)]


# unit test cases
print(getLambertCoefs(5))
print(getLambertCoefs(1))
print(getLambertCoefs(20))
print(getLambertCoefs(12))
print(getLambertCoefs(15))