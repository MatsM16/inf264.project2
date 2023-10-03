from dump import get_dump_file

log_level = "debug"
"""
One of "debug", "critical"
"""

def log(message:str):
    """
    Writes a message to the log.
    """
    print(message)
    with open(get_dump_file("log.txt"), mode="a+") as log_file:
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