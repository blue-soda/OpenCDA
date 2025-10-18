import json
import socket
import threading
import time
from opencda.log.logger_config import logger
from ..config.settings import NS3_HOST, NS3_SEND_PORT, NS3_RECV_PORT
from typing import *

class CarlaNs3Bridge:
    """Bridge for communication between CARLA and ns-3 using standard sockets"""
    
    def __init__(self, ns3_host: str = NS3_HOST, ns3_send_port: int = NS3_SEND_PORT, ns3_recv_port: int = NS3_RECV_PORT):
        self.ns3_host = ns3_host
        self.ns3_send_port = ns3_send_port
        self.ns3_recv_port = ns3_recv_port
        self.socket = None
        self.receiver_socket = None
        self.connected = False
        self.running = True
        self.reconnect_thread = None
        self.receiver_thread = None
        self.received_cams = []

    def _connect(self) -> bool:
        """Connect to ns-3 server"""
        if self.socket:
            self.socket.close()
            
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ns3_host, self.ns3_send_port))
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Error connecting to NS-3 bridge: {e}")
            self.connected = False
            return False

    def _reconnect_loop(self):
        """Try to reconnect periodically"""
        while self.running and not self.connected:
            if self._connect():
                break
            time.sleep(5)

    def ensure_connection(self):
        """Ensure there's a connection to ns-3, try to reconnect if not"""
        if not self.connected and not self.reconnect_thread:
            self.reconnect_thread = threading.Thread(target=self._reconnect_loop)
            self.reconnect_thread.daemon = True
            self.reconnect_thread.start()

    def _listen_for_messages(self):
        """Listen for messages from ns-3"""
        try:
            self.receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.receiver_socket.bind((self.ns3_host, self.ns3_recv_port))
            self.receiver_socket.listen(1)
            client_socket, addr = self.receiver_socket.accept()
            while self.running:
                try:
                    data = client_socket.recv(1024)
                    if data:
                        try:
                            message = json.loads(data.decode('utf-8'))
                            if message.get("type") == "simulation_end":
                                logger.info("Received simulation end signal from NS-3")
                                self.running = False
                                break
                            elif message.get("type") == "cam_received":
                                receiver_id = message.get("receiver_id")
                                sender_id = message.get("sender_id")
                                logger.info(f"Info from NS-3: Vehicle {receiver_id} received msg from Vehicle {sender_id}, " +
                                            f"msg sent at {message.get('send_timestamp')}, received at {message.get('receive_timestamp')}")
                                self.received_cams.append(message)
                        except json.JSONDecodeError:
                            pass
                    # client_socket.close()
                except socket.error:
                    break
        except Exception as e:
            logger.error(f"Error in end signal listener: {e}")
        finally:
            client_socket.close()
            if self.receiver_socket:
                self.receiver_socket.close()

    def _start_receiver(self):
        """Start threads to receive messages from ns-3"""
            
        if not self.receiver_thread:
            self.receiver_thread = threading.Thread(target=self._listen_for_messages)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

    def send_something_to_ns3(self, msg_type: str, data):
        """Send something to ns-3"""
        if not self.running:
            logger.info("Simulation ended, not sending more vehicle states")
            return False
            
        if not self.connected:
            logger.warning("Not connected, attempting to reconnect...")
            self.ensure_connection()
            if not self.connected:
                logger.error("Failed to reconnect")
                return False
        
        try:
            message_obj = {"type": msg_type, msg_type: data}
            message = json.dumps(message_obj)
            self.socket.sendall((message + "\n\r").encode('utf-8'))
            # logger.info(f"Sent {len(message)} bytes to NS-3 successfully")
            return True
        except Exception as e:
            logger.error(f"Error sending vehicle states: {e}")
            self.connected = False
            return False

    def stop(self):
        """Stop the bridge"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.receiver_socket:
            self.receiver_socket.close()
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=1.0)
        if self.receiver_thread:
            self.receiver_thread.join(timeout=1.0)

    def _start_sender(self):
        """Start the sender"""
        self._connect()

    def start(self):
        """Start the bridge"""
        self._start_receiver()
        self._start_sender()
        logger.info("Bridge started")

    def is_simulation_running(self) -> bool:
        """Check if the simulation is running"""
        return self.running

    def send_vehicles_num(self, vehicles_num: int):
        self.send_something_to_ns3(msg_type = "vehicles_num", data = vehicles_num)

    def send_transfer_requests(self, requests: List[Dict[str, int]]):
        """
        requests: [{"source":s, "target":t, "size":n}, {...}, ...]
        """
        self.send_something_to_ns3(msg_type = "transfer_requests", data = requests)

    def send_vehicles_position(self, vehicles: List[Dict[str, int]]):
        """
        vehicles: [{
            "id": index,
            "carla_id": vehicle.id,
            "position": position,
            "velocity": velocity_data,
            "heading": heading,
            "speed": speed
        }, ...]
        """
        self.send_something_to_ns3(msg_type = "vehicles_position", data = vehicles)