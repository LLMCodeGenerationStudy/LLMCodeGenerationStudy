import os.path

def path_spliter(path: str):
    """
    Splits the given path into its components.

    Args:
        path (str): The path to split.

    Returns:
        list: A list of path components.
    """
    # ----
    
    res = []
    if path == '':
        return []

    while path != '' and path != '/':
        res.append(os.path.split(path)[1])
        path = os.path.split(path)[0]
    if path != '':
        res.append(path)
    

    return res[::-1]


# unit test cases
str = '/usr/bin/python'
print(path_spliter(str))

str = '../folder/subfolder/file.txt'
print(path_spliter(str))

str = './../xx/yy/./../yty/file.txt'
print(path_spliter(str))

str = ''
print(path_spliter(str))