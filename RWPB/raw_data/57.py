def clean_folder_name(folder_name: str) -> str:
    """
    Sanitizes a folder name by replacing characters that are invalid in file paths with underscores.

    Arguments:
    folder_name : str
        The original folder name that may contain invalid characters.

    Returns:
    str
        A sanitized version of the folder name where all invalid characters have been replaced with underscores.
    """
    # ----

    cleaned_name = folder_name  # Start with the original folder name
    invalid_chars = '<>:"/\\|?*.'  # Define a string of characters that are invalid in file names

    # Replace each invalid character with an underscore
    for char in invalid_chars:
        cleaned_name = cleaned_name.replace(char, "_")

    return cleaned_name

# unit test cases
print(clean_folder_name("Project<Name>:Version/1.2\3|4*5?6"))
print(clean_folder_name("Regular_Folder_Name"))
print(clean_folder_name("<>>:|?**...."))