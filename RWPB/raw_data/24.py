import re
import hashlib


def check(x, checksum_token):
    """
    Verifies if the provided checksum in a given string matches the SHA-256 hash of the string with a replaced checksum.

    Parameters
    ----------
    x : str
        The input string that contains a checksum to be verified.
    checksum_token : str
        A placeholder token that will replace the current checksum in the string for hashing.

    Returns
    -------
    bool
        True if the checksum is correct, False otherwise.

    Notes
    -----
    This function uses a regular expression to find the checksum in the input string `x`.
    It then replaces the checksum with `checksum_token`, hashes the resulting string using SHA-256,
    and compares the hash with the original checksum. If they match, the function returns True,
    indicating that the checksum is valid. If they do not match or if no checksum is found,
    the function returns False.
    """
    # ----


    # Pre-compiled regular expression for matching and replacing checksums
    re_checksum = re.compile(r'"Checksum": "([0-9a-fA-F]{64})"')

    # Search for the checksum in the input string
    m = re.search(re_checksum, x)
    if not m:
        return False

    # Replace the checksum with the provided token
    replaced = re.sub(re_checksum, f'"Checksum": "{checksum_token}"', x)

    # Calculate the SHA-256 hash of the modified string
    h = hashlib.sha256(replaced.encode("utf8"))

    # Compare the calculated hash with the original checksum
    return h.hexdigest() == m.group(1)


# unit test cases
print(check('{"Data": "Example data", "Checksum": "0b3ad42a156a99abcce760eee811fcfcbb806126697b77c5eb9638e20c455439"}', checksum_token = "TOKEN"))
print(check('{"Data": "Example data", "Checksum": "f03a9dcfea7e161acf569d267c75034617de5779b416ea8e4aea18d52318dd4b"}', checksum_token = "Apple"))
print(check('{"Data": "Example data", "Checksum": "f03a9dcfea7e161acf569d267c75034617de5779b416ea8e4aea18d52318dd4b"}', checksum_token = "LLMs"))