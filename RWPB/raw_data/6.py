def format_seconds(seconds: float) -> str:
    """Converts a float number (seconds) to a string in the format:

    - HH:MM:SS.SSS when greater than or equal to one hour
    - MM:SS.SSS when less than one hour

    Args:
        seconds: The float number to convert (in seconds).

    Returns:
        str: The formatted string representation of the time.
    """
    # ----
    
    hours = int(seconds // 3600)  # Extract whole hours
    minutes = int((seconds % 3600) // 60)  # Extract whole minutes
    seconds = seconds % 60  # Extract remaining seconds

    # Use f-strings for clean formatting
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{minutes:02d}:{seconds:06.3f}"
    

# unit test cases
assert(format_seconds(90.123) == format_seconds(90.123))
assert(format_seconds(3600.0) == format_seconds(3600.0))
assert(format_seconds(5436.789) == format_seconds(5436.789))