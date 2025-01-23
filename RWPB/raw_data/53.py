import re

def general_detokenize(string):
    """
    Detokenizes a string by fixing spacing around punctuation and contractions.

    Arguments:
    string : str
        The input string that needs to be detokenized.

    Returns:
    str
        The detokenized string with corrected spacing.
    """
    # ----

    # Fix the space before contractions like "n't" (e.g., " ca n't" -> "can't")
    string = string.replace(" n't", "n't")

    # Remove space before closing parentheses and after opening parentheses
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")

    # Correct spacing issues with quotation marks
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')

    # Use regular expressions to remove space before commas and periods, and other single character punctuations
    string = re.sub(r" (['.,])", r"\1", string)

    return string

# unit test cases
print(general_detokenize('She said , " Hello world ! " I can not believe it ; can you ? Yes , I ca n\'t .'))
print(general_detokenize('We need ( in an ideal world ) a kind of setup that works ( efficiently and effectively ).'))
print(general_detokenize('This is a test . What do you think ? Isn\'t it interesting ?'))
