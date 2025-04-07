import logging
from datetime import datetime
import os

log_filename = f"opencda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(current_dir, log_filename)
logger = logging.getLogger("opencda_log")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

