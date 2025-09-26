import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, name="carla_ns3_co_simulation"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "simulation.log")
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def get_log_file(self):
        return self.log_file

logger = Logger() 