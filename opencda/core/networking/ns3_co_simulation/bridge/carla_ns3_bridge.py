import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from opencda.log.logger_config import logger

from ..config.settings import NS3_HOST, NS3_RECV_PORT, NS3_SEND_PORT


class CarlaNs3Bridge:
    """Bridge for communication between CARLA and ns-3 using standard sockets."""

    def __init__(
        self,
        ns3_host: str = NS3_HOST,
        ns3_send_port: int = NS3_SEND_PORT,
        ns3_recv_port: int = NS3_RECV_PORT,
    ):
        self.ns3_host = ns3_host
        self.ns3_send_port = ns3_send_port
        self.ns3_recv_port = ns3_recv_port
        self.socket = None
        self.receiver_socket = None
        self.client_socket = None
        self.connected = False
        self.running = True
        self.reconnect_thread = None
        self.receiver_thread = None
        self.received_cams = {}
        self.lock = threading.Lock()

        # Large point-cloud uploads are fragmented; their fragments can arrive
        # far more than 10 ms apart in synchronized NS3 runs.
        self.combine_threshold_ms = 1000

        self.sync_event = threading.Event()
        self.sync_event.set()
        self.last_sync_time = 0.0
        self.ns3_current_time = 0.0
        self.sync_timeout = 1.0
        self.enable_sync = True
        self.connect_timeout = 15.0
        self.connect_retry_interval = 0.2
        self.expected_sync_time = None
        self.sync_time_tolerance = 1e-3

    def _connect(self, quiet: bool = False) -> bool:
        """Connect to ns-3 server."""
        if self.socket:
            self.socket.close()

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ns3_host, self.ns3_send_port))
            self.connected = True
            return True
        except Exception as e:
            if not quiet:
                logger.error(f"Error connecting to NS-3 bridge: {e}")
            self.connected = False
            return False

    def _reconnect_loop(self):
        """Try to reconnect periodically."""
        try:
            while self.running and not self.connected:
                if self._connect():
                    break
                time.sleep(5)
        finally:
            self.reconnect_thread = None

    def _wait_for_connection(self, timeout: Optional[float] = None) -> bool:
        """Block for a short window until the outbound NS3 connection is ready."""
        timeout = self.connect_timeout if timeout is None else timeout
        deadline = time.time() + timeout
        logged_wait = False

        while self.running and time.time() < deadline:
            if self.connected:
                return True
            if not logged_wait:
                logger.info(
                    f"Waiting for NS-3 bridge on {self.ns3_host}:{self.ns3_send_port}..."
                )
                logged_wait = True
            if self._connect(quiet=True):
                logger.info(
                    f"Connected to NS-3 bridge on {self.ns3_host}:{self.ns3_send_port}"
                )
                return True
            time.sleep(self.connect_retry_interval)

        return self.connected

    def ensure_connection(self):
        """Ensure there is a connection to ns-3, try to reconnect if not."""
        if not self.connected and not self.reconnect_thread:
            self.reconnect_thread = threading.Thread(target=self._reconnect_loop)
            self.reconnect_thread.daemon = True
            self.reconnect_thread.start()

    def _close_client_socket(self):
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None

    def _listen_for_messages(self):
        """Listen for messages from ns-3 and tolerate reconnects."""
        self.running = True
        try:
            self.receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self.receiver_socket.bind(("", self.ns3_recv_port))
            self.receiver_socket.listen(1)
            self.receiver_socket.settimeout(10.0)
            logger.info(f"Listening for NS-3 connections on {self.ns3_host}:{self.ns3_recv_port}")

            while self.running:
                if self.client_socket is None:
                    try:
                        self.client_socket, addr = self.receiver_socket.accept()
                        logger.info(f"Connected to NS-3 at {addr}")
                        print(f"Connected to NS-3 at {addr}")
                        self.client_socket.settimeout(5.0)
                    except socket.timeout:
                        logger.info("Still waiting for NS-3 connection...")
                        continue

                incomplete_data = b""
                while self.running and self.client_socket is not None:
                    try:
                        chunk = self.client_socket.recv(131072)
                        if not chunk:
                            logger.warning("NS-3 closed the connection; waiting for reconnection")
                            print("NS-3 closed the connection; waiting for reconnection")
                            self._close_client_socket()
                            break

                        logger.info(f"[DEBUG] _listen_for_messages: received chunk of {len(chunk)} bytes")
                        incomplete_data += chunk
                        while b"\r\n" in incomplete_data:
                            msg_bytes, incomplete_data = incomplete_data.split(b"\r\n", 1)
                            if not msg_bytes:
                                continue
                            try:
                                message = json.loads(msg_bytes.decode("utf-8"))
                                logger.info(
                                    f"[DEBUG] _listen_for_messages: parsed JSON with type={message.get('type')}"
                                )
                                self._process_message(message)
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON from NS-3: {e}, raw data: {msg_bytes}")
                                print(f"Invalid JSON from NS-3: {e}")
                                continue
                    except socket.timeout:
                        continue
                    except socket.error as e:
                        logger.error(f"Socket error: {e}")
                        print(f"Socket error: {e}")
                        self._close_client_socket()
                        break

        except Exception as e:
            logger.error(f"Fatal error in listener: {e}")
            print(f"Fatal error in listener: {e}")
        finally:
            self._close_client_socket()
            if self.receiver_socket:
                try:
                    self.receiver_socket.close()
                except Exception:
                    pass
                self.receiver_socket = None
            logger.info("Listener stopped")
            print("Listener stopped")

    def _process_message(self, message: Dict[str, Any]):
        """Process a single NS-3 message."""
        with self.lock:
            msg_type = message.get("type")
            logger.info(f"[DEBUG] _process_message: received type={msg_type}")
            if msg_type == "simulation_end":
                logger.info("Received simulation end signal from NS-3")
                print("Received simulation end signal from NS-3")
                self.running = False
            elif msg_type == "cam_received":
                logger.info("[DEBUG] _process_message: processing cam_received")
                self._process_cam_message(message)
            elif msg_type == "sync_ack":
                self._process_sync_ack(message)

    def _process_cam_message(self, message: Dict[str, Any]):
        """Process CAM received message."""
        receiver_id = message.get("receiver_id")
        sender_id = message.get("sender_id")
        packet_size = message.get("packet_size", 0)
        receive_timestamp = message.get("receive_timestamp", 0)
        send_timestamp = message.get("send_timestamp", 0)

        if not all([receiver_id, sender_id, receive_timestamp, send_timestamp]):
            logger.warning("Invalid CAM message: missing fields")
            return

        delay = receive_timestamp - send_timestamp
        logger.info(
            f"NS-3 Info: Vehicle {receiver_id} received msg from {sender_id} with {packet_size} bytes, "
            f"delay: {delay}ms (send_timestamp: {send_timestamp}, receive_timestamp: {receive_timestamp})"
        )

        if receiver_id not in self.received_cams:
            self.received_cams[receiver_id] = {sender_id: message.copy()}
            return

        sender_dict = self.received_cams[receiver_id]
        if sender_id not in sender_dict:
            sender_dict[sender_id] = message.copy()
            return

        existing_msg = sender_dict[sender_id]
        existing_send_ts = existing_msg.get("send_timestamp", 0)
        if abs(send_timestamp - existing_send_ts) > self.combine_threshold_ms:
            sender_dict[sender_id] = message.copy()
            logger.info(f"New message for {sender_id}->{receiver_id} (time gap)")
            print(f"New message for {sender_id}->{receiver_id} (time gap)")
            return

        existing_msg["packet_size"] += packet_size
        existing_msg["receive_timestamp"] = receive_timestamp
        existing_msg["is_last_packet"] = existing_msg.get("is_last_packet", 0) or message.get(
            "is_last_packet", 0
        )
        sender_dict[sender_id] = existing_msg
        logger.info(
            f"Combined message for {sender_id}->{receiver_id}, total size: {existing_msg['packet_size']} bytes, "
            f"is_last_packet: {existing_msg['is_last_packet']}, total delay: {receive_timestamp - existing_send_ts}ms"
        )

    def _process_sync_ack(self, message: Dict[str, Any]):
        """Process sync acknowledgment from NS3."""
        ns3_time = message.get("ns3_time", 0.0)
        carla_time = message.get("carla_time", 0.0)

        self.ns3_current_time = ns3_time
        logger.info(
            f"[DEBUG] _process_sync_ack: received sync_ack, NS3 time={ns3_time:.4f}s, CARLA time={carla_time:.4f}s"
        )

        if self.sync_event.is_set():
            logger.warning("[DEBUG] _process_sync_ack: sync_ack received but not waiting for sync")
            return

        expected = self.expected_sync_time
        if expected is not None and abs(carla_time - expected) > self.sync_time_tolerance:
            logger.warning(
                f"[DEBUG] _process_sync_ack: stale sync_ack ignored, "
                f"expected carla_time={expected:.4f}, got={carla_time:.4f}"
            )
            return

        logger.info("[DEBUG] _process_sync_ack: setting sync_event")
        self.sync_event.set()

    def sync_with_ns3(self, carla_time: float) -> bool:
        """Synchronize with NS3 by sending current CARLA time and waiting for acknowledgment."""
        logger.info(
            f"[DEBUG] sync_with_ns3: starting sync for carla_time={carla_time:.4f}, "
            f"enable_sync={self.enable_sync}, running={self.running}"
        )
        if not self.enable_sync:
            return True

        if not self.running:
            return False

        self.sync_event.clear()
        self.expected_sync_time = carla_time
        sync_data = {
            "carla_time": carla_time,
            "request_time": time.time(),
        }

        if not self.send_something_to_ns3(msg_type="sync_request", data=sync_data):
            logger.error("Failed to send sync_request to NS3")
            self.expected_sync_time = None
            self.sync_event.set()
            return False
        logger.info("[DEBUG] sync_with_ns3: sync_request sent, waiting for sync_ack")

        wait_success = self.sync_event.wait(timeout=self.sync_timeout)
        if not wait_success:
            logger.warning(f"Sync timeout after {self.sync_timeout}s for CARLA time {carla_time:.4f}s")
            self.expected_sync_time = None
            self.sync_event.set()
            return False

        self.last_sync_time = carla_time
        self.expected_sync_time = None
        logger.info(
            f"[DEBUG] sync_with_ns3: sync successful, carla_time={carla_time:.4f}, ns3_time={self.ns3_current_time:.4f}"
        )
        return True

    def enable_time_sync(self, enable: bool = True):
        self.enable_sync = enable
        logger.info(f"Time synchronization {'enabled' if enable else 'disabled'}")

    def _start_receiver(self):
        if not self.receiver_thread:
            self.receiver_thread = threading.Thread(target=self._listen_for_messages)
            self.receiver_thread.daemon = True
            self.receiver_thread.start()

    def send_something_to_ns3(self, msg_type: str, data):
        """Send something to ns-3."""
        if not self.running:
            logger.info("Simulation ended, not sending more vehicle states")
            return False

        if not self.connected:
            logger.warning("Not connected, attempting to reconnect...")
            self.ensure_connection()
            if not self._wait_for_connection(timeout=min(self.sync_timeout, 5.0)):
                logger.error("Failed to reconnect")
                return False
            if not self.connected:
                logger.error("Failed to reconnect")
                return False

        try:
            message_obj = {"type": msg_type, msg_type: data}
            message = json.dumps(message_obj)
            logger.info(
                f"[DEBUG] send_something_to_ns3: sending type={msg_type}, msg_len={len(message)}, connected={self.connected}"
            )
            self.socket.sendall((message + "\n\r").encode("utf-8"))
            return True
        except Exception as e:
            logger.error(f"Error sending vehicle states: {e}")
            self.connected = False
            return False

    def stop(self):
        """Stop the bridge."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
        if self.receiver_socket:
            try:
                self.receiver_socket.close()
            except Exception:
                pass
            self.receiver_socket = None
        self._close_client_socket()
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=1.0)
        if self.receiver_thread:
            self.receiver_thread.join(timeout=1.0)

    def _start_sender(self):
        if not self._wait_for_connection():
            logger.error(
                f"Timed out waiting for NS-3 bridge at {self.ns3_host}:{self.ns3_send_port}"
            )

    def start(self):
        self._start_receiver()
        self._start_sender()
        logger.info("Bridge started")

    def is_simulation_running(self) -> bool:
        return self.running

    def send_vehicles_num(self, vehicles_num: int):
        self.send_something_to_ns3(msg_type="vehicles_num", data=vehicles_num)

    def send_transfer_requests(self, requests: List[Dict[str, int]]):
        logger.info(f"[DEBUG] send_transfer_requests: sending {len(requests)} requests")
        self.send_something_to_ns3(msg_type="transfer_requests", data=requests)

    def send_vehicles_position(self, vehicles: List[Dict[str, int]]):
        logger.info(f"[DEBUG] send_vehicles_position: sending {len(vehicles)} vehicles")
        self.send_something_to_ns3(msg_type="vehicles_position", data=vehicles)
