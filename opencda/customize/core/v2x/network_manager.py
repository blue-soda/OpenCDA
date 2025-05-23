# from opencda.core.common.v2x_manager import V2XManager
from collections import defaultdict
import opencda.customize.core.v2x.utils as utils
import numpy as np
from collections import defaultdict
import math
import numpy as np
from opencda.log.logger_config import logger

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
        self.subchannel_num = config.get("subchannel_num", 25)
        self.subchannel_bandwidth = config.get("subchannel_bandwidth", 0.180) * 1e6  #Hz
        # self.max_interference = config.get("max_interference", 0.2)
        self.min_sinr_threshold = config.get("min_sinr_threshold", 3) #dB
        self.time_slot = config.get("time_slot", 0.05)
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
            # Skip our own transmission (we want OTHER transmitters' interference)
            # if tgt_id == target_vehicle.vehicle_id:
            #     continue
                
            source_vm = self.cav_world.get_vehicle_manager(src_id).v2x_manager
            if source_vm:
                interference += utils.get_interference_contribution(
                    source_vm, 
                    target_vehicle
                )
        
        return interference

    def allocate_resource(self, source, target, volume: float,
                        subchannel: int):
        """
        Allocate resources for a communication request and calculate the required number of time slots.
        
        Args:
            source (V2XManager): Source V2XManager.
            target (V2XManager): Target V2XManager.
            volume (float): Data volume to transmit (in bytes).
            subchannel (int): Subchannel to allocate.

        Returns:
            Tuple: subchannel, current_time_slot, end_time_slot

        Raises:
            ValueError: If the maximum interference threshold is exceeded.
        """
        # 1. Calculate interference at receiver from OTHER transmitters
        interference = self.calculate_interference(subchannel, target)
        
        # 2. Calculate our signal's contribution to receiver
        our_signal = utils.get_interference_contribution(source, target)
        
        # 3. Total interference = other transmitters + noise floor
        # total_interference = interference + target.noise_level
        
        # 4. Calculate actual SINR
        # snr = utils.calculate_snr(
        #     tx_power=source.tx_power,
        #     noise_level=total_interference,  # Includes other transmitters + noise
        #     distance=utils.calculate_distance(source, target)
        # )
        sinr = utils.calculate_sinr(our_signal, interference, target.noise_level)
        
        # 5. Verify interference threshold
        logger.debug(f"signal power: {our_signal}, {interference}, {target.noise_level} in subchannel {subchannel}")
        logger.info(f"sinr: {sinr}")
        if sinr < self.min_sinr_threshold: 
            # raise ResourceConflictError("SINR too low for reliable communication.")
            self._record_collision()
            return -1, -1, -1
        
        # 6. Determine data rate and time slots needed
        data_rate = utils.calculate_available_data_rate(
            self.subchannel_bandwidth,
            sinr,
        ) / 8 #(bit to byte)
        logger.info(f"data rate: {data_rate}")
        
        transmission_delay = volume / data_rate
        time_slots = math.ceil(transmission_delay / self.time_slot)
        
        # Record allocation
        end_time_slot = self.current_time_slot + time_slots
        self.active_allocations[subchannel].add((source.vehicle_id, target.vehicle_id, end_time_slot))

        self._record_transmission_latency(transmission_delay)
        # # Update communication stats (assume 'upload' type for now)
        # self._update_communication_stats(volume, "upload")

        # return time_slots
        return subchannel, self.current_time_slot, end_time_slot
    

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

    def _record_packet_latency(self, latency: float):
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
