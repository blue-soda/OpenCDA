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
        self.client_socket = None
        self.connected = False
        self.running = True
        self.reconnect_thread = None
        self.receiver_thread = None
        self.received_cams = {}
        self.lock = threading.Lock() # Lock for thread-safe operations
        self.combine_threshold_ms = 10  # Time threshold to combine messages from the same sender (in milliseconds)

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
            """Listen for messages from ns-3 (fixed:粘包/拆包、资源泄漏、超时处理)"""
            self.running = True
            incomplete_data = b""  # 缓存不完整的消息数据
            try:
                # 创建套接字并配置
                self.receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                self.receiver_socket.bind((self.ns3_host, self.ns3_recv_port))
                self.receiver_socket.listen(1)
                self.receiver_socket.settimeout(10.0)  # 监听超时，避免永久阻塞
                logger.info(f"Listening for NS-3 connections on {self.ns3_host}:{self.ns3_recv_port}")

                # 等待NS-3连接
                try:
                    self.client_socket, addr = self.receiver_socket.accept()
                    logger.info(f"Connected to NS-3 at {addr}")
                    print(f"Connected to NS-3 at {addr}")
                    self.client_socket.settimeout(5.0)  # 接收超时
                except socket.timeout:
                    logger.error("Timeout waiting for NS-3 connection")
                    print("Timeout waiting for NS-3 connection")
                    return

                while self.running:
                    try:
                        # 读取数据（处理粘包/拆包）
                        chunk = self.client_socket.recv(4096)  # 增大缓冲区，减少调用次数
                        if not chunk:  # 连接断开
                            logger.warning("NS-3 closed the connection")
                            print("NS-3 closed the connection")
                            break

                        incomplete_data += chunk
                        # 按换行符分割消息（NS-3需在每条JSON后加\n）
                        while b"\n\r" in incomplete_data:
                            msg_bytes, incomplete_data = incomplete_data.split(b"\n", 1)
                            if not msg_bytes:
                                continue
                            # 解析JSON
                            try:
                                message = json.loads(msg_bytes.decode('utf-8'))
                                self._process_message(message)
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON from NS-3: {e}, raw data: {msg_bytes}")
                                print(f"Invalid JSON from NS-3: {e}")
                                continue

                    except socket.timeout:
                        continue  # 超时无数据，继续循环
                    except socket.error as e:
                        logger.error(f"Socket error: {e}")
                        print(f"Socket error: {e}")
                        break

            except Exception as e:
                logger.error(f"Fatal error in listener: {e}")
                print(f"Fatal error in listener: {e}")
            finally:
                # 安全关闭套接字
                if self.client_socket:
                    try:
                        self.client_socket.shutdown(socket.SHUT_RDWR)
                        self.client_socket.close()
                    except:
                        pass
                if self.receiver_socket:
                    self.receiver_socket.close()
                self.running = False
                logger.info("Listener stopped")
                print("Listener stopped")

    def _process_message(self, message: Dict[str, Any]):
        """Process a single NS-3 message (thread-safe)"""
        with self.lock:  # 保护共享数据
            msg_type = message.get("type")
            if msg_type == "simulation_end":
                logger.info("Received simulation end signal from NS-3")
                print("Received simulation end signal from NS-3")
                self.running = False
            elif msg_type == "cam_received":
                self._process_cam_message(message)

    def _process_cam_message(self, message: Dict[str, Any]):
        """Process CAM received message (thread-safe)"""
        receiver_id = message.get("receiver_id")
        sender_id = message.get("sender_id")
        packet_size = message.get("packet_size", 0)
        receive_timestamp = message.get("receive_timestamp", 0)
        send_timestamp = message.get("send_timestamp", 0)

        if not all([receiver_id, sender_id, receive_timestamp, send_timestamp]):
            logger.warning("Invalid CAM message: missing fields")
            return

        # 打印日志
        delay = receive_timestamp - send_timestamp
        logger.info(
            f"NS-3 Info: Vehicle {receiver_id} received {packet_size} bytes from {sender_id}, "
            f"delay: {delay}ms (send_timestamp: {send_timestamp}, receive_timestamp: {receive_timestamp})"
        )
        # print(
        #     f"NS-3 Info: Vehicle {receiver_id} received {packet_size} bytes from {sender_id}, "
        #     f"delay: {delay}ms (send: {send_timestamp}, receive: {receive_timestamp})"
        # )

        # 消息合并逻辑
        if receiver_id not in self.received_cams:
            self.received_cams[receiver_id] = {sender_id: message.copy()}
        else:
            sender_dict = self.received_cams[receiver_id]
            if sender_id not in sender_dict:
                self.received_cams[receiver_id][sender_id] = message.copy()
            else:
                existing_msg = sender_dict[sender_id]
                existing_send_ts = existing_msg.get("send_timestamp", 0)
                if abs(send_timestamp - existing_send_ts) > self.combine_threshold_ms:
                    # 新消息：时间差超过阈值
                    self.received_cams[receiver_id][sender_id] = message.copy()
                    logger.info(f"New message for {sender_id}->{receiver_id} (time gap)")
                    print(f"New message for {sender_id}->{receiver_id} (time gap)")
                else:
                    # 合并消息
                    existing_msg["packet_size"] += packet_size
                    existing_msg["receive_timestamp"] = receive_timestamp
                    self.received_cams[receiver_id][sender_id] = existing_msg
                    logger.info(
                        f"Combined message for {sender_id}->{receiver_id}, total size: {existing_msg['packet_size']} bytes"
                    )
                    # print(
                    #     f"Combined message for {receiver_id}->{sender_id}, total size: {existing_msg['packet_size']} bytes"
                    # )


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
        if self.client_socket:
            self.client_socket.close()
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
        requests: [{"source":s_id, "target":t_id, "size":nbytes, ("sc_start": subchannel_start, "sc_num": subchannel_num) }, {...}, ...]
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