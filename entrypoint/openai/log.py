import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_logger_initialized = False


def setup_logging():
    """
    Configures the application's logging.
    Logs will be written to a timestamped file in the 'logs' directory,
    with rotation based on file size (10MB) and daily new files on app restart.
    """
    global _logger_initialized
    if _logger_initialized:
        return logging.getLogger()

    # Create a logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Generate a timestamped log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = os.path.join(log_dir, f"api_server_{timestamp}.log")

    # Configure the root logger
    logger = logging.getLogger()  # Get the root logger
    logger.setLevel(logging.INFO)  # Set the minimum logging level

    # Clear existing handlers to prevent duplicate logs if called multiple times
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Create a file handler for rotating logs by size
    # MaxBytes = 10MB, backupCount = 5 (keep 5 old log files)
    file_handler = RotatingFileHandler(
        log_file_name, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10 MB
    )
    # Define the format for the log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Optionally, also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Ensure Uvicorn loggers propagate to the root logger
    logging.getLogger("uvicorn").propagate = True
    logging.getLogger("uvicorn.access").propagate = True
    logging.getLogger("uvicorn.error").propagate = True

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.ERROR)

    logger.info("Logging configured successfully.")
    _logger_initialized = True
    return logger


# Define a custom stream to redirect stdout/stderr to the logger
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass
