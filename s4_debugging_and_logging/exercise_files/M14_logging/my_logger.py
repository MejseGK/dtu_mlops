from loguru import logger
import sys

logger.remove()  # Remove the default logger / sink / where the messages are sent
logger.add(sys.stdout, level="WARNING")  # Add a new logger with >=WARNING level (only WARNING and above will be shown to user / in terminal)
logger.add("my_log_file.log", level="DEBUG", rotation="100 MB")  # Add a file logger with >=DEBUG level (all messages will be logged to the file). Log file will rotate when it reaches 100 MB.

logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")