import json
import os
from datetime import datetime
from typing import List, Dict

class VehicleDataLogger:
    def __init__(self):
        self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.temp_dir, f'vehicle_data.json')
        self.data = {
            "simulation_start": self.timestamp,
            "frames": []
        }
        
    def log_frame(self, vehicle_data: List[Dict]):
        """Log a single frame of vehicle data
        
        Args:
            vehicle_data: List of dictionaries containing vehicle data
        """
        frame_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vehicles": vehicle_data
        }
        self.data["frames"].append(frame_data)
        
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)

vehicle_data_logger = VehicleDataLogger() 