def seconds_to_hh_mm_ss(seconds):
    """
    Converts a given number of seconds to a string in hh:mm:ss format.

    Args:
        seconds: An integer representing the number of seconds.

    Returns:
        A string in hh:mm:ss format, or "00:00:00" if seconds is negative.
    """
    # ----
    if seconds < 0:
        return "00:00:00"

    hours = seconds // 3600  # Get hours by dividing by total seconds in an hour
    seconds %= 3600  # Get remaining seconds after hours calculation

    minutes = seconds // 60  # Get minutes by dividing by total seconds in a minute
    seconds %= 60  # Get remaining seconds after minutes calculation

    # Format hour, minute, and second values with leading zeros
    hours_str = f"{hours:02d}"
    minutes_str = f"{minutes:02d}"
    seconds_str = f"{seconds:02d}"

    return f"{hours_str}:{minutes_str}:{seconds_str}"


# unit test cases
assert(seconds_to_hh_mm_ss(-10) == seconds_to_hh_mm_ss(-10))
assert(seconds_to_hh_mm_ss(3661) == seconds_to_hh_mm_ss(3661))
assert(seconds_to_hh_mm_ss(7200) == seconds_to_hh_mm_ss(7200))