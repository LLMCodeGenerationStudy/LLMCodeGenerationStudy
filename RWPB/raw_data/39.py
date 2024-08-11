import re
from typing import List


def clean_html(
        html_to_clean: str,
        tags_to_remove: List[str] = ["style", "svg", "script"],
        attributes_to_keep: List[str] = ["id", "href"],
) -> str:
    """
    Cleans HTML content by removing specified tags and their contents, and strips out all attributes except those specified.

    Args:
        html_to_clean (str): The HTML content to clean.
        tags_to_remove (List[str]): List of HTML tags to remove from the content. Default tags are 'style', 'svg', and 'script'.
        attributes_to_keep (List[str]): List of HTML attributes to retain in the remaining HTML tags. Default is ['id', 'href'].

    Returns:
        str: The cleaned HTML content, free of specified tags and with only desired attributes retained.

    This function first removes all specified tags and their contents using regular expressions. It then eliminates all attributes
    from the remaining tags except for those explicitly listed to be kept. This is useful for preparing HTML content for environments
    where only basic styling or functionality is needed, or for security purposes to remove potentially malicious content.

    Example:
        >>> clean_html('<div id="main" style="color:red">Hello <script>alert("World")</script></div>',
                       tags_to_remove=['script'], attributes_to_keep=['id'])
        '<div id="main">Hello </div>'
    """
    # ----

    # Remove specified tags and their contents completely
    for tag in tags_to_remove:
        html_to_clean = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", "", html_to_clean, flags=re.DOTALL
        )

    # Compile a pattern that matches any attribute not in the list of attributes to keep
    attributes_to_keep = "|".join(attributes_to_keep)
    pattern = rf'\b(?!({attributes_to_keep})\b)\w+(?:-\w+)?\s*=\s*["\'][^"\']*["\']'

    # Remove all attributes that are not explicitly listed to be kept
    cleaned_html = re.sub(pattern, "", html_to_clean)

    cleaned_html = re.sub(rf'\s*>', ">", cleaned_html)

    return cleaned_html


# unit test cases
print(clean_html('<div><style> .text { font-size: 12px; } </style><p id="info">Visit our <a href="http://example.com" style="color: red;">website</a></p></div>'))
print(clean_html('<div id="container"><script>alert(\'Hello\');</script><svg><circle cx="50" cy="50" r="40" stroke="black" fill="red"></circle></svg><a href="#nothing" onclick="javascript:void(0);" style="color: black;">Click here</a></div>'))
print(clean_html('<header style="background: black;"><nav id="top-nav" class="navigation" data-toggle="collapse"><a href="/home" title="Home">Home</a></nav></header>'))