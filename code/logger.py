from datetime import datetime

log_level = "debug"
"""
One of "debug", "critical"
"""

log_file_path = f"dump/{datetime.now().strftime('%Y-%m-%d-%H%M')}.txt"
"""
File location for log file: dump/log_YEAR-MONTH-DATE-HOURMINUTE.txt
"""

def log(message:str):
    """
    Writes a message to the log.
    """
    print(message)
    with open(log_file_path, mode="a+") as log_file:
        log_file.write(f"{message}\n")

def log_debug(message:str):
    """
    Writes a debug-message to the log.
    """
    if log_level == "debug":
        log(f"[dbug] {message}")

def log_critical(message:str):
    """
    Writes a critical-message to the log.
    """
    log(f"[crit] {message}")