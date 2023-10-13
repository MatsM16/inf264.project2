from datetime import datetime
from os import mkdir

start_timestamp = datetime.now()
"""
Date and time the program was started.
"""

dump_file_prefix = f"dump/{start_timestamp.strftime('%Y-%m-%d-%H%M')}/"
"""
Prefix given to dump-files.
"""

# Create the dump directory.
mkdir(dump_file_prefix)

def get_dump_file(name:str):
    """
    Gets a file-path to a dump-file with the given name.
    """
    return dump_file_prefix + name