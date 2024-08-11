from typing import List

def extract_imports_from_lines(lines: List[str]) -> str:
    """
    Extracts import statements from a list of Python code lines.

    Args:
        lines (List[str]): A list of strings, each representing a line of Python code.

    Returns:
        str: A single string containing all extracted import and from-import lines, separated by new lines.

    This function filters through a list of Python code lines, identifying and extracting lines that are import statements.
    It supports both 'import' and 'from ... import ...' formats. This is useful for analyzing or consolidating imports
    from a set of Python code lines, such as when preparing a script or checking for consistency and redundancy in imports.
    """
    # ----

    # Use list comprehension to filter out lines that start with 'import' or 'from', indicating import statements
    return "\n".join(
        [line for line in lines if line.startswith("from") or line.startswith("import")]
    )



# unit test cases
a = [
    "import os",
    "x = 10",                 # Non-import line
    "from sys import argv",
    "print('Hello World!')",  # Non-import line
    "import math"
]
print(extract_imports_from_lines(a))

b = [
    "Import os",                  # Incorrect case
    "# from datetime import datetime",  # Commented import
    "from collections import deque",    # Correct import
    "IMPORT sys",                 # Incorrect case
    "from math import pi"         # Correct import
]
print(extract_imports_from_lines(b))

c = [
    "from numpy import array, random",
    "import scipy.stats as stats",  # Aliased import
    "from os.path import join, exists",  # Multiple imports from same module
    "x = 'import should not be detected'",  # Import within a string
    "import pandas as pd"  # Aliased import
]
print(extract_imports_from_lines(c))