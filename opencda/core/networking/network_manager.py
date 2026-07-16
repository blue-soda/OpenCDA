# from opencda.core.common.v2x_manager import V2XManager
from collections import defaultdict
import threading
from typing import Tuple
import opencda.core.networking.utils as utils
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
        self.config = config  # Store full config for later use
        self.subchannel_num = config.get("subchannel_num", 10)
        self.subchannel_bandwidth = config.get("subchannel_bandwidth", 0.180) * 1e6  #Hz
        # self.max_interference = config.get("max_interference", 0.2)
        self.min_sinr_threshold = config.get("min_sinr_threshold", 3) #dB
        # One NetworkManager slot is one CARLA synchronous tick. CavWorld
        # injects world.fixed_delta_seconds into config["time_slot"], so do
        # not rescale it here; otherwise CARLA, local scheduling, and NS3 sync
        # drift into different time bases.
        self.time_slot = config.get("time_slot", 0.05)
        self.use_ns3 = config.get("use_ns3", False)
        self.current_time_slot = 0
        self.world_tick = False
        self.pkt_id = 1

        # Time synchronization state
        self.fixed_delta_seconds = cav_world.fixed_delta_seconds if hasattr(cav_world, 'fixed_delta_seconds') else 0.05
        self.current_sim_time = 0.0  # Current CARLA simulation time in seconds
        self.tick_count = 0  # Number of ticks processed

        # Allocation state
        self.active_allocations = defaultdict(set)  # {subchannel: {(src_id, tgt_id, end_time_slot)}}

        # Enhanced statistics tracking
        self.current_slot = {
            'try_volume': 0.0,
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
        self.vehicle_registration_complete = False
        self.max_vehicle_num = 30
        self.max_packet_size = 10000 # 64000 # bytes, UDP packet size
        self._ns3_initialized = False
        if self.use_ns3:
            self.init_ns3()
        self.data_event = None

    def set_data_event(self, event):
        self.data_event = event

    def add_vehicles(self, vehicles):
        self.all_vehicles.extend(vehicles)

    def mark_vehicle_registration_complete(self):
        """Allow the NS3 sender to initialize after scenario vehicles exist."""
        self.vehicle_registration_complete = True
        logger.info(
            "NS3 vehicle registration marked complete with %s vehicles",
            len(self.all_vehicles))

    def _wait_for_ns3_vehicle_initialization(self):
        """Send the first vehicle frame before time synchronization starts."""
        while self.bridge.is_simulation_running():
            if not self.vehicle_registration_complete:
                time.sleep(min(self.time_slot, 0.01))
                continue
            if self.all_vehicles:
                vehicle_data = collect_vehicle_data(
                    self.all_vehicles, self.cav_world)
                if any(v.get("carla_id") is None for v in vehicle_data):
                    logger.info(
                        "Waiting for CARLA-to-OpenCDA id mapping before NS3 "
                        "initialization")
                    time.sleep(min(self.time_slot, 0.01))
                    continue
                self.bridge.send_vehicles_num(len(vehicle_data))
                self.bridge.send_vehicles_position(vehicle_data)
                self._ns3_initialized = True
                logger.info(
                    "NS3 initialized with %s vehicles before first sync",
                    len(vehicle_data))
                return True
            time.sleep(min(self.time_slot, 0.01))
        return False

    def tick(self):
        """Called when CARLA world ticks. Updates simulation time and signals sender."""
        self.tick_count += 1
        self.current_sim_time = self.tick_count * self.fixed_delta_seconds
        self.world_tick = True

    def send_msg_to_ns3(self):
        """Send messages to ns-3 if needed.

        This method runs in a separate thread and communicates with NS3.
        It uses time synchronization to ensure NS3 and CARLA stay in sync.
        """
        try:
            if not self._wait_for_ns3_vehicle_initialization():
                logger.warning("NS3 sender stopped before vehicles were registered")
                return
            logger.info("NS3 sender thread started, waiting for world ticks")
            while self.bridge.is_simulation_running():
                if self.world_tick:
                    self.world_tick = False

                    # Synchronize with NS3 before sending data
                    sync_result = self.bridge.sync_with_ns3(self.current_sim_time)
                    if not sync_result:
                        # NS3 is not responding - don't send on a dead socket.
                        # Setting connected=False triggers reconnection on next tick.
                        # Sending on a dead socket would cause TCP errors that could
                        # propagate to bridge.stop() and close the CARLA-NS3 connection,
                        # which terminates the entire co-simulation prematurely.
                        logger.warning(f"Sync failed at time {self.current_sim_time:.4f}s, not sending data (NS3 may be terminated)")
                        self.bridge.connected = False
                        # Trigger reconnection attempt so the next sync can succeed
                        self.bridge.ensure_connection()
                        time.sleep(min(self.time_slot, 0.01))
                        continue

                    vehicle_data = collect_vehicle_data(self.all_vehicles, self.cav_world)
                    self.bridge.send_vehicles_position(vehicle_data)

                    if len(self.communication_requests) == 0:
                        time.sleep(min(self.time_slot, 0.01))
                        continue
                    self.bridge.send_transfer_requests(self.communication_requests[:])
                    self.communication_requests = []
                time.sleep(min(self.time_slot, 0.01))

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")

        finally:
            try:
                self.bridge.stop()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        logger.info("Simulation ended")

    def get_current_sim_time(self):
        """Get the current CARLA simulation time in seconds."""
        return self.current_sim_time

    def init_ns3(self):
        """Initialize ns-3 module if needed."""
        if not self.use_ns3:
            return

        self.bridge = CarlaNs3Bridge()
        self.bridge.start()

        # Configure time synchronization from config
        enable_sync = self.config.get('enable_time_sync', True)
        # Very short sync timeouts misclassify normal NS3 startup / burst-delivery
        # latency as bridge failure and can tear down the co-simulation early.
        sync_timeout = max(self.config.get('sync_timeout', 10.0), 8.0)
        self.bridge.enable_time_sync(enable_sync)
        self.bridge.sync_timeout = sync_timeout
        logger.info(f"NS3 time sync enabled={enable_sync}, timeout={sync_timeout}s")

        if not self.sender_thread:
            self.sender_thread = threading.Thread(target=self.send_msg_to_ns3)
            self.sender_thread.daemon = True
            self.sender_thread.start()

    def enable_time_sync(self, enable: bool = True):
        """Enable or disable time synchronization with NS3.

        Args:
            enable: True to enable sync (default), False to disable
        """
        if self.bridge:
            self.bridge.enable_time_sync(enable)
        logger.info(f"Time synchronization {'enabled' if enable else 'disabled'}")

    def communicate(self, source, target, volume: float, subchannel_start: int = -1, subchannel_num: int = 0) -> bool:
        """
        Wrapper for resource allocation and communication handling.
        """
        logger.info(f"[DEBUG] communicate: {source.vehicle_id} -> {target.vehicle_id}, volume={volume}, subchannel_start={subchannel_start}, subchannel_num={subchannel_num}, use_ns3={self.use_ns3}")
        # print(f"Communicate from {source.vehicle_id} to {target.vehicle_id} with volume {volume} on subchannel {subchannel_start} for {subchannel_num} subchannels, use ns-3: {self.use_ns3}.")
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
        logger.info(f"[DEBUG] communicate_through_ns3: {source.vehicle_id} -> {target.vehicle_id}, volume={volume}, subchannel_start={subchannel_start}, subchannel_num={subchannel_num}, use_ns3={self.use_ns3}")
        if self.bridge is None:
            raise RuntimeError("ns-3 bridge not initialized.")

        while volume > self.max_packet_size:
            self.send_cams_via_ns3(source.vehicle_id, target.vehicle_id, self.max_packet_size, subchannel_start, subchannel_num)
            volume -= self.max_packet_size

        self.send_cams_via_ns3(source.vehicle_id, target.vehicle_id, volume, subchannel_start, subchannel_num)

        return True
    
    def send_cams_via_ns3(self, src_id, tgt_id, volume: float, subchannel_start: int = -1, subchannel_num: int = 0):
        logger.info(f"[DEBUG] send_cams_via_ns3: src={src_id}, tgt={tgt_id}, volume={volume}")
        use_default_subchannel = subchannel_start < 0 or subchannel_num <= 0
        if use_default_subchannel:
            self.communication_requests.append({
                "source": src_id,
                "target": tgt_id,
                "size": volume, # bytes
                "pkt_id": self.pkt_id,
            })
        else:
            self.communication_requests.append({
                "source": src_id,
                "target": tgt_id,
                "size": volume, # bytes
                "sc_start": subchannel_start,
                "sc_num": subchannel_num,
                "pkt_id": self.pkt_id,
            })
        self.pkt_id += 1
        self._update_communication_stats(volume, "try")
        logger.info(f"[DEBUG] send_cams_via_ns3: communication_requests now has {len(self.communication_requests)} items")
            
    def get_all_received_cams(self):
        return self.bridge.received_cams.copy()
    
    def set_all_received_cams(self, cams):
        self.bridge.received_cams = cams

    def clear_all_received_cams(self):
        self.bridge.received_cams = {}

    def get_received_cams(self, receiver_id):
        logger.debug(f"get_received_cams, {self.bridge.received_cams}")
        logger.debug(f"get_received_cams, {self.bridge.received_cams.keys()}")
        return self.bridge.received_cams.get(receiver_id, {}).copy()

    def pop_received_cams(self, receiver_id, sender_id=None):
        if sender_id is None:
            cams = self.bridge.received_cams.pop(receiver_id, None)
            if cams:
                return self.analyze_ns3_results(cams)
            else:
                return {}
        cams = self.bridge.received_cams.get(receiver_id, None)
        if cams:
            cam = cams.pop(sender_id, None)
            self.bridge.received_cams[receiver_id] = cams
            if cam:
                return self.analyze_ns3_result(cam)
            else:
                return {}

    def peek_received_cams(self, receiver_id, sender_id):
        """Peek at received cam without removing it. For NS3 fragmented reception."""
        cams = self.bridge.received_cams.get(receiver_id, None)
        if cams:
            return cams.get(sender_id, None)
        return None

    def analyze_ns3_result(self, cam):
        delay = cam.get('receive_timestamp', -1) - cam.get('send_timestamp', -1)
        self._record_transmission_latency(delay)  #ms
        volume = cam.get('packet_size', -1)
        self._update_communication_stats(volume, "upload")
        # print(f"CAM from {cam.get('sender_id')} to {cam.get('receiver_id')} delay: {delay} ms")
        logger.info(f"CAM from {cam.get('sender_id')} to {cam.get('receiver_id')} delay: {delay} ms, volume: {volume} bytes")
        return { (cam.get('sender_id'), cam.get('receiver_id')): delay}

    def analyze_ns3_results(self, cams):
        ret = {}
        for cam in cams.values():
            ret.update(self.analyze_ns3_result(cam))
        return ret

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
        sinr = utils.calculate_sinr(our_signal, interference, target.noise_power)
        
        # 4. Verify interference threshold
        logger.debug(f"signal power: {our_signal}, {interference}, {target.noise_power} in subchannel {subchannel_start}-{subchannel_start + subchannel_num -1}")
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
        if comm_type == "try":
            self.current_slot['try_volume'] += volume
            return
        
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
            'try_volume': 0.0,
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

        print(f"{self.pkt_id} pkts sent.")
        
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
            'utilization': np.array([s['utilization'] for s in self.history]),
            'try_volume': np.array([s['try_volume'] for s in self.history])
        }

        print("history_try_volume: ", hist_arrays['try_volume'])
        
        # Calculate traffic distribution percentages
        total_vol = hist_arrays['throughput'].sum()
        dist = {}
        if total_vol > 0:
            dist = {
                'total_vol(Bytes)': total_vol,
                'intra_upload_pct(%)': 100 * hist_arrays['intra_upload'].sum() / total_vol,
                'intra_download_pct(%)': 100 * hist_arrays['intra_download'].sum() / total_vol,
                'inter_cluster_pct(%)': 100 * hist_arrays['inter_cluster'].sum() / total_vol,
                'control_pct(%)': 100 * hist_arrays['control'].sum() / total_vol,
                'try_volume': hist_arrays['try_volume'].sum()
            }
        else:
            dist = {k: 0.0 for k in ['intra_upload_pct(%)', 'intra_download_pct(%)', 
                                    'inter_cluster_pct(%)', 'control_pct(%)', 'try_volume']}
        
        return {
            # 'current': self.current_slot,
            'traffic_distribution': dist,
            'historical': {
                'total_slots': len(self.history),
                'avg_throughput': float(np.mean(hist_arrays['throughput'])),
                'avg_t_latency': float(np.mean(hist_arrays['t_latency'])),
                'avg_p_latency': float(np.mean(hist_arrays['p_latency'])),
                'avg_utilization': float(np.mean(hist_arrays['utilization'])),
                'total_volume_bytes': float(hist_arrays['throughput'].sum()),
                'max_throughput': float(np.max(hist_arrays['throughput'])),
                'avg_try_volume': float(np.mean(hist_arrays['try_volume'])),
                # 'throughput_trend': hist_arrays['throughput'].tolist()  # Full history
            },
        }

    def advance_time_slot(self):
        """Progress network state while preserving statistics."""
        # Archive the slot that just completed before advancing the logical
        # slot index. This keeps history.slot_index aligned with the CARLA
        # tick that produced the statistics.
        self.finalize_slot()

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
        self.tick()


class ResourceConflictError(Exception):
    """
    Raised when a resource allocation conflict occurs.
    """
    pass
