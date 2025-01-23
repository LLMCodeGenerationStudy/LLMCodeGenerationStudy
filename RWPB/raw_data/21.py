import re

def validate_user_id(user_id):
    """
    Validates a user ID against a specific pattern.

    Parameters
    ----------
    user_id : str
        The user ID to validate.

    Returns
    -------
    bool
        True if the user ID is valid, False otherwise.

    Notes
    -----
    The function checks if the user ID is a non-empty string that:
    - Starts with a letter (uppercase or lowercase).
    - Followed by any number of alphanumeric characters or underscores.

    The validation is performed using a regular expression (regex) pattern.
    """
    # ----
    
    # Define the regex pattern for a valid user ID
    pattern = r'^[A-Za-z][A-Za-z0-9_]*$'

    # Check if the user_id matches the pattern and is an instance of str
    if isinstance(user_id, str) and re.match(pattern, user_id):
        return True
    else:
        return False
    

# unit test cases
print(validate_user_id("A1_bcd"))
print(validate_user_id("Z"))
print(validate_user_id("123abc"))