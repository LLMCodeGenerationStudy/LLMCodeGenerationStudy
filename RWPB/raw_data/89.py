import math


def format_ratio(ratio):
    """
    Formats a ratio into a percentage string

    Args:
        - ratio: float, ratio to be formatted
    Returns:
        formatted_ratio: str, formatted ratio in percentage

    Examples:
        - input: 3.1245133412
        - output: percentage:3.124
    """
    # ----
    
    percentage = ratio * 100
    return f"{percentage:3.3f}%"


# unit test cases
print(format(3.14))
print(format(6.29))
print(format(0))