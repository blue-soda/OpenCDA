# from opencda.core.common.v2x_manager import V2XManager
from collections import defaultdict
import threading
from typing import Tuple
import opencda.customize.core.v2x.utils as utils
import numpy as np
from collections import defaultdict
import math
import numpy as np
import time
from opencda.log.logger_config import logger
from .ns3_co_simulation.carla.vehicle_data import collect_vehicle_data
from .ns3_co_simulation.bridge.carla_ns3_bridge import CarlaNs3Bridge

class NetworkManager:
    """
    Enhanced network manager with comprehensive communication statistics tracking.
    
    Maintains:
    - Per-time-slot allocation records
    - Subchannel interference levels
    - Detailed communication metrics
    - Historical performance data
    """

    def __init__(self, cav_world, config):
        self.cav_world = cav_world
        self.subchannel_num = config.get("subchannel_num", 10)
        self.subchannel_bandwidth = config.get("subchannel_bandwidth", 0.180) * 1e6  #Hz
        # self.max_interference = config.get("max_interference", 0.2)
        self.min_sinr_threshold = config.get("min_sinr_threshold", 3) #dB
        self.time_slot = config.get("time_slot", 0.05)
        self.use_ns3 = config.get("use_ns3", False)
        self.current_time_slot = 0

        # Allocation state
        self.active_allocations = defaultdict(set)  # {subchannel: {(src_id, tgt_id, end_time_slot)}}
        
        # Enhanced statistics tracking
        self.current_slot = {
            'total_volume': 0.0,
            'intra_cluster': {'upload': 0.0, 'download': 0.0},
            'inter_cluster': 0.0,
            'control_overhead': 0.0,
            'collisions': 0,
            't_latency': [],
            'p_latency': [],
            'utilization': 0.0  # Will be calculated when slot ends
        }
        
        # History stores complete snapshots of each time slot
        self.history = []  # List of slot records

        self.bridge = None
        self.communication_requests = []
        self.receiver_thread = None
        self.sender_thread = None
        self.all_vehicles = []
        self.max_vehicle_num = 30
        self.max_packet_size = 10000 # 64000 # bytes, UDP packet size
        if self.use_ns3:
            self.init_ns3()
        self.data_event = None

    def set_data_event(self, event):
        self.data_event = event

    def add_vehicles(self, vehicles):
        self.all_vehicles.extend(vehicles)

    def send_msg_to_ns3(self):
        """Send messages to ns-3 if needed."""
        try:
            self.bridge.send_vehicles_num(self.max_vehicle_num)  # Initial vehicle count
            while self.bridge.is_simulation_running():
                while len(self.communication_requests) == 0:
                    time.sleep(self.time_slot)
                # self.bridge.send_vehicles_num(len(self.all_vehicles))
                vehicle_data = collect_vehicle_data(self.all_vehicles)
                self.bridge.send_vehicles_position(vehicle_data)
                self.bridge.send_transfer_requests(self.communication_requests[:])
                self.communication_requests = []
                time.sleep(self.time_slot)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")

        finally:
            try:
                self.bridge.stop()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        logger.info("Simulation ended")

    def init_ns3(self):
        """Initialize ns-3 module if needed."""
        if not self.use_ns3:
            return
        
        self.bridge = CarlaNs3Bridge()
        self.bridge.start()

        if not self.sender_thread:
            self.sender_thread = threading.Thread(target=self.send_msg_to_ns3)
            self.sender_thread.daemon = True
            self.sender_thread.start()

    def communicate(self, source, target, volume: float, subchannel_start: int = -1, subchannel_num: int = 0) -> bool:
        """
        Wrapper for resource allocation and communication handling.
        """
        if(self.use_ns3):
            return self.communicate_through_ns3(source, target, volume, subchannel_start, subchannel_num)
        elif(subchannel_start >= 0 and subchannel_start + subchannel_num <= self.subchannel_num):
            return self.allocate_resource(source, target, volume, subchannel_start)
        else:
            raise ValueError("Invalid subchannel index.")

    def communicate_through_ns3(self, source, target, volume: float, subchannel_start: int = -1, subchannel_num: int = 0) -> bool:
        """
        Handle communication via ns-3 bridge.
        """
        if self.bridge is None:
            raise RuntimeError("ns-3 bridge not initialized.")
        
        while volume > self.max_packet_size:
            self.send_cams_via_ns3(source.vehicle_id, target.vehicle_id, self.max_packet_size, subchannel_start, subchannel_num)
            volume -= self.max_packet_size

        self.send_cams_via_ns3(source.vehicle_id, target.vehicle_id, volume, subchannel_start, subchannel_num)

        return True
    
    def send_cams_via_ns3(self, src_id, tgt_id, volume: float, subchannel_start: int = -1, subchannel_num: int = 0):
        use_default_subchannel = subchannel_start == -1 or subchannel_num == 0
        if use_default_subchannel:
            self.communication_requests.append({
                "source": src_id,
                "target": tgt_id,
                "size": volume, # bytes
            })
        else:
            self.communication_requests.append({
                "source": src_id,
                "target": tgt_id,
                "size": volume, # bytes
                "sc_start": subchannel_start,
                "sc_num": subchannel_num
            })
            
    def get_all_received_cams(self):
        return self.bridge.received_cams.copy()
    
    def set_all_received_cams(self, cams):
        self.bridge.received_cams = cams

    def clear_all_received_cams(self):
        self.bridge.received_cams = {}

    def get_received_cams(self, receiver_id):
        return self.bridge.received_cams.get(receiver_id, {}).copy()

    def pop_received_cams(self, receiver_id, sender_id=None):
        if sender_id is None:
            cams = self.bridge.received_cams.pop(receiver_id, None)
            if cams:
                self.analyze_ns3_results(cams)
            return
        cams = self.bridge.received_cams.get(receiver_id, None)
        if cams:
            cam = cams.pop(sender_id, None)
            if cam:
                self.analyze_ns3_result(cam)
            self.bridge.received_cams[receiver_id] = cams

    def analyze_ns3_result(self, cam):
        delay = cam.get('receive_timestamp', -1) - cam.get('send_timestamp', -1)
        self._record_transmission_latency(delay)  #ms
        # print(f"CAM from {cam.get('sender_id')} to {cam.get('receiver_id')} delay: {delay} ms")
        logger.info(f"CAM from {cam.get('sender_id')} to {cam.get('receiver_id')} delay: {delay} ms")

    def analyze_ns3_results(self, cams):
        for cam in cams.values():
            self.analyze_ns3_result(cam)

    def allocate_resource(self, source, target, volume: float,
                        subchannel_start: int, subchannel_num: int):
        """
        Allocate resources for a communication request and calculate the required number of time slots.
        
        Args:
            source (V2XManager): Source V2XManager.
            target (V2XManager): Target V2XManager.
            volume (float): Data volume to transmit (in bytes).
            subchannel (int): Subchannel to allocate.

        Returns:
            bool: Whether the communication was successful.

        Raises:
            ValueError: If the maximum interference threshold is exceeded.
        """
        # 1. Calculate interference at receiver from OTHER transmitters
        interference = self.calculate_interference(subchannel_start, target)
        
        # 2. Calculate our signal's contribution to receiver
        our_signal = utils.get_interference_contribution(source, target)
        
        # 3. Calculate SINR
        sinr = utils.calculate_sinr(our_signal, interference, target.noise_level)
        
        # 4. Verify interference threshold
        logger.debug(f"signal power: {our_signal}, {interference}, {target.noise_level} in subchannel {subchannel_start}-{subchannel_start + subchannel_num -1}")
        logger.info(f"sinr: {sinr}")
        if sinr < self.min_sinr_threshold: 
            # raise ResourceConflictError("SINR too low for reliable communication.")
            self._record_collision()
            return False
        
        # 5. Determine data rate and time slots needed
        data_rate = utils.calculate_available_data_rate(
            self.subchannel_bandwidth * subchannel_num,
            sinr,
        ) / 8 #(bit to byte)
        logger.info(f"data rate: {data_rate}")
        
        transmission_delay = volume / data_rate
        time_slots = math.ceil(transmission_delay / self.time_slot)
        
        # Record allocation
        end_time_slot = self.current_time_slot + time_slots
        for sc in range(subchannel_start, subchannel_start + subchannel_num):
            self.active_allocations[sc].add((source.vehicle_id, target.vehicle_id, end_time_slot))

        self._record_transmission_latency(transmission_delay)
        # # Update communication stats (assume 'upload' type for now)
        # self._update_communication_stats(volume, "upload")

        # return time_slots
        return True
    
    def calculate_interference(self, subchannel: int, target_vehicle) -> float:
        """
        Calculate the total interference experienced by a target vehicle on a subchannel.
        
        Args:
            subchannel: Subchannel index
            target_vehicle: The receiving vehicle experiencing interference, a V2XManager instance
            
        Returns:
            Total interference power at the receiver from all other transmitters
        """
        interference = 0.0
        
        # Sum interference from all active transmissions on this subchannel
        for src_id, tgt_id, _ in self.active_allocations[subchannel]:
                
            source_vm = self.cav_world.get_vehicle_manager(src_id).v2x_manager
            if source_vm:
                interference += utils.get_interference_contribution(
                    source_vm, 
                    target_vehicle
                )
        
        return interference
    
    def _update_communication_stats(self, volume: float, comm_type: str = "upload"):
        """
        Update real-time communication metrics for current time slot
        
        Args:
            volume: Data volume in Bytes
            comm_type: Type of communication, one of:
                      'upload' - intra-cluster upstream (child->leader)
                      'download' - intra-cluster downstream (leader->child)
                      'inter' - inter-cluster communication
                      'control' - control signaling overhead
        """
        self.current_slot['total_volume'] += volume
        
        if comm_type == "upload":
            self.current_slot['intra_cluster']['upload'] += volume
        elif comm_type == "download":
            self.current_slot['intra_cluster']['download'] += volume
        elif comm_type == "inter":
            self.current_slot['inter_cluster'] += volume
        elif comm_type == "control":
            self.current_slot['control_overhead'] += volume

    def _calculate_utilization(self):
        """
        Calculate network utilization percentage for current slot
        
        Returns:
            Utilization percentage (0-100)
        """
        max_capacity = self.subchannel_bandwidth * self.subchannel_num / 8 * 0.9
        if max_capacity <= 0:
            return 0.0
        return min(100.0, (self.current_slot['total_volume'] / max_capacity) * 100)
    
    def finalize_slot(self):
        """
        Finalize current time slot statistics and archive to history
        
        Args:
            max_capacity: Used for utilization calculation
        """
        # Calculate final utilization before archiving
        self.current_slot['utilization'] = self._calculate_utilization()
        
        # Deep copy current slot to history
        self.history.append({
            'slot_index': len(self.history),
            **{k: v.copy() if isinstance(v, dict) else v 
               for k, v in self.current_slot.items()}
        })
        
        # self.analyze_ns3_results()
        # Reset current slot counters
        self._reset_current_slot()

    def _reset_current_slot(self):
        """Reset all counters for new time slot"""
        self.current_slot = {
            'total_volume': 0.0,
            'intra_cluster': {'upload': 0.0, 'download': 0.0},
            'inter_cluster': 0.0,
            'control_overhead': 0.0,
            'collisions': 0,
            't_latency': [],
            'p_latency': [],
            'utilization': 0.0
        }

    def _record_collision(self):
        """Handle collision events in statistics."""
        self.current_slot['collisions'] += 1

    def _record_transmission_latency(self, latency: float):
        self.current_slot['t_latency'].append(latency)

    def _record_cp_latency(self, latency: float):
        self.current_slot['p_latency'].append(latency)

    def get_communication_report(self) -> dict:
        """
        Generate comprehensive communication performance report
        
        Returns:
            Dictionary containing:
            - current: Latest slot metrics
            - historical: Aggregated statistics over all slots
            - traffic_distribution: Percentage breakdown by type
        """
        self.current_slot['utilization'] = self._calculate_utilization()

        if not self.history:
            return {'current': self.current_slot, 'historical': None}
        
        # Convert history lists to numpy arrays for vector operations
        hist_arrays = {
            'throughput': np.array([s['total_volume'] for s in self.history]),
            'intra_upload': np.array([s['intra_cluster']['upload'] for s in self.history]),
            'intra_download': np.array([s['intra_cluster']['download'] for s in self.history]),
            'inter_cluster': np.array([s['inter_cluster'] for s in self.history]),
            'control': np.array([s['control_overhead'] for s in self.history]),
            't_latency': np.array([latency_value for s in self.history for latency_value in s['t_latency']]),
            'p_latency': np.array([latency_value for s in self.history for latency_value in s['p_latency']]),
            'utilization': np.array([s['utilization'] for s in self.history])
        }
        
        # Calculate traffic distribution percentages
        total_vol = hist_arrays['throughput'].sum()
        dist = {}
        if total_vol > 0:
            dist = {
                'total_vol(Bytes)': total_vol,
                'intra_upload_pct(%)': 100 * hist_arrays['intra_upload'].sum() / total_vol,
                'intra_download_pct(%)': 100 * hist_arrays['intra_download'].sum() / total_vol,
                'inter_cluster_pct(%)': 100 * hist_arrays['inter_cluster'].sum() / total_vol,
                'control_pct(%)': 100 * hist_arrays['control'].sum() / total_vol
            }
        else:
            dist = {k: 0.0 for k in ['intra_upload_pct(%)', 'intra_download_pct(%)', 
                                    'inter_cluster_pct(%)', 'control_pct(%)']}
        
        return {
            'current': self.current_slot,
            'traffic_distribution': dist,
            'historical': {
                'total_slots': len(self.history),
                'avg_throughput': float(np.mean(hist_arrays['throughput'])),
                'avg_t_latency': float(np.mean(hist_arrays['t_latency'])),
                'avg_p_latency': float(np.mean(hist_arrays['p_latency'])),
                'avg_utilization': float(np.mean(hist_arrays['utilization'])),
                'total_volume_bytes': float(hist_arrays['throughput'].sum()),
                'max_throughput': float(np.max(hist_arrays['throughput'])),
                # 'throughput_trend': hist_arrays['throughput'].tolist()  # Full history
            },
        }

    def advance_time_slot(self):
        """Progress network state while preserving statistics."""
        # Clean up expired allocations
        for subchannel in list(self.active_allocations.keys()):
            self.active_allocations[subchannel] = {
                allocation for allocation in self.active_allocations[subchannel]
                if allocation[2] > self.current_time_slot
            }
            if not self.active_allocations[subchannel]:
                del self.active_allocations[subchannel]

        # Update current time slot
        self.current_time_slot += 1

        # Finalize and reset statistics for the current slot
        self.finalize_slot()


class ResourceConflictError(Exception):
    """
    Raised when a resource allocation conflict occurs.
    """
    pass
