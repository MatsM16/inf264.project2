from datetime import datetime

start_timestamp = datetime.now()
"""
Date and time the program was started.
"""

dump_file_prefix = f"dump/{start_timestamp.strftime('%Y-%m-%d-%H%M')}_"
"""
Prefix given to dump-files.
"""

def get_dump_file(name:str):
    """
    Gets a file-path to a dump-file with the given name.
    """
    return dump_file_prefix + name