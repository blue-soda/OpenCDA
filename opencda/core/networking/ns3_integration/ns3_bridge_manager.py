"""NS3 bridge manager for network simulation."""

import threading
import time
from opencda.log.logger_config import logger


class NS3BridgeManager:
    """Manages NS3 co-simulation bridge."""

    def __init__(self, bridge, time_slot, max_packet_size=10000):
        self.bridge = bridge
        self.time_slot = time_slot
        self.max_packet_size = max_packet_size
        self.communication_requests = []
        self.sender_thread = None
        self.world_tick = False
        self.pkt_id = 1

    def start(self, cav_world, all_vehicles, max_vehicle_num=30):
        """Start NS3 bridge and sender thread."""
        self.cav_world = cav_world
        self.all_vehicles = all_vehicles
        self.max_vehicle_num = max_vehicle_num

        if not self.sender_thread:
            self.sender_thread = threading.Thread(target=self._send_loop)
            self.sender_thread.daemon = True
            self.sender_thread.start()

    def _send_loop(self):
        """Send messages to NS3."""
        from opencda.core.networking.ns3_co_simulation.carla.vehicle_data import collect_vehicle_data

        try:
            self.bridge.send_vehicles_num(self.max_vehicle_num)
            while self.bridge.is_simulation_running():
                if self.world_tick:
                    self.world_tick = False
                    vehicle_data = collect_vehicle_data(self.all_vehicles, self.cav_world)
                    self.bridge.send_vehicles_position(vehicle_data)
                    if len(self.communication_requests) > 0:
                        self.bridge.send_transfer_requests(self.communication_requests[:])
                        self.communication_requests = []
                time.sleep(self.time_slot / 5.0)
        except KeyboardInterrupt:
            logger.info("NS3 simulation interrupted")
        finally:
            self.bridge.stop()

    def send_packet(self, src_id, tgt_id, volume, subchannel_start=-1, subchannel_num=0):
        """Queue packet for NS3 transmission."""
        request = {
            "source": src_id,
            "target": tgt_id,
            "size": volume,
            "pkt_id": self.pkt_id,
        }
        if subchannel_start >= 0 and subchannel_num > 0:
            request["sc_start"] = subchannel_start
            request["sc_num"] = subchannel_num

        self.communication_requests.append(request)
        self.pkt_id += 1

    def tick(self):
        """Signal world tick."""
        self.world_tick = True

    def get_received_cams(self, receiver_id):
        """Get received CAMs for a receiver."""
        return self.bridge.received_cams.get(receiver_id, {}).copy()
