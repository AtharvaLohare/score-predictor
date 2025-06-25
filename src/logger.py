import os
import logging
from datetime import datetime

# Creating the name for the log file.
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Creating the path for the directory that will store the log files.
LOGS_PATH = os.path.join(os.getcwd(),"logs")
os.makedir(LOGS_PATH, exist_ok=True)

# Create the path for the current log file
LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    # Logs only above the level INFO are logged.
    level=logging.INFO,
)