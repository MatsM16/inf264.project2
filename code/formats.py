def format_bytes(byte_count:int):
    """
    Formats a byte-count into a human readable format.
    """
    INCREMENT = 1024
    byte_count *= 1.0 # Convert to float
    
    for name in ['B', 'KB', 'MB', 'GB', 'TB']:
        if byte_count < INCREMENT:
            return f"{byte_count:.0f}{name}"
        byte_count /= INCREMENT

    return f"{byte_count:.0f}PB"

def format_duration(duration:float):
    """
    Formats a duration (in nanoseconds) to a human readable format.
    """
    if duration < 1_000_000:
        # I dont really care about decimal places for nanoseconds.
        return f"{duration:.0f}ns"
    
    duration /= 1_000_000
    if duration < 1_000:
        return f"{duration:.2f}ms"

    duration /= 1_000
    if duration < 60:
        return f"{duration:.2f}s"

    duration /= 60
    return f"{duration:.2f}min"

def format_label(label:int):
    if label > 15: return "_"
    return f"{int(label):x}".upper()