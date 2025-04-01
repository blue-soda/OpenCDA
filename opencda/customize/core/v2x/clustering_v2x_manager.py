from opencda.core.common.v2x_manager import V2XManager
import math
from opencda.core.common.misc import compute_distance
import random
import sys

STANDARD_CAPABILITY= 100
def calculate_cos(direction1, direction2):
    dot_product = (
        direction1[0] * direction2[0] +
        direction1[1] * direction2[1] +
        direction1[2] * direction2[2]
    )

    magnitude1 = (direction1[0] ** 2 + direction1[1] ** 2 + direction1[2] ** 2) ** 0.5
    magnitude2 = (direction2[0] ** 2 + direction2[1] ** 2 + direction2[2] ** 2) ** 0.5

    cos_theta = dot_product / (magnitude1 * magnitude2 + 0.0001) 
    return cos_theta

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class ClusteringV2XManager(V2XManager):
    Communication_Volume = 0.0
    Instance_Nums = 0
    Communication_Volume_Inside_Cluster_Collect = 0.0
    Communication_Volume_Inside_Cluster_Broadcast = 0.0
    Communication_Volume_Outside_Cluster = 0.0
    UpdateSucceed = True

    def __init__(self, cav_world, config_yaml, vid, vehicle_id):
        super(ClusteringV2XManager, self).__init__(cav_world, config_yaml, vid)
        self.vehicle_id = vehicle_id
        
        self.computing_capability = STANDARD_CAPABILITY
        self.communication_quality = STANDARD_CAPABILITY
        self.cp_model = 'default_model'

        self.tick = 0

        self._unread_buffer = False
        # used for cooperative perception.
        # self._recieved_buffer = {} defined in v2xmanager

        self.beacon_frequency = 2 #(Hz)

        ClusteringV2XManager.Instance_Nums += 1
        print(f'{ClusteringV2XManager.Instance_Nums} ExtendedV2XManager initialized')

        # 分簇协议相关参数
        self.cluster_params = {
            'd0': 50.0,           # 距离归一化参数 (单位: m)
            's0': 5.0,            # 速度归一化参数 (单位: m/s)
            'N_th': 5,            # 邻居数阈值
            'kappa': 0.1,         # 权重调节系数
            'eta_join': 0.7,      # 加入簇的阈值
            'eta_create': 0.5,    # 创建簇的阈值
            'RSSI_max': 100,      # 最大接收信号强度
            'epsilon': 0.1,       # 协同感知模型差异阈值
            'sigma': 0.2,         # 创建簇的概率调节参数
            'w1': 0.4,  # 通信质量权重
            'w2': 0.3,  # 计算能力权重
            'w3': 0.3,  # 速度一致性权重
            'T_timeout': 1.0,  # 簇头超时时间 (单位: s)
            'shadow_timeout': 0.5,  # 影子簇头超时时间 (单位: s)
            'eta_join': 0.7,  # 加入阈值
            'eta_leave': 0.3,  # 离开阈值 (eta_join - delta_eta)
        }

        # 分簇协议状态
        self.cluster_state = {
            'cluster_head': None,          # 当前簇头 (None表示无簇头)
            'members': {},                 # 成员信息
            'neighbors': {},               # 邻居信息
            'similarity_scores': {},       # 邻居相似度分数
            'beacons_received': 0,         # 接收到的信标帧数
            'RSSI_avg': 0,                 # 平均接收信号强度
            'shadow_head': None,  # 影子簇头ID
            'priority_score': 0.0,  # 本地优先级得分
        }


    def is_cluster_head(self):
        # print('vehicle_id', self.vehicle_id, self.cluster_state['cluster_head'], self.vehicle_id == self.cluster_state['cluster_head'])
        return self.vehicle_id == self.cluster_state['cluster_head']

    def set_buffer(self, source=None, objects=None, results=None):
        if source:
            self._recieved_buffer['source'] = source
        if objects:
            self._recieved_buffer['objects'] = objects
        if results:
            self._recieved_buffer['results'] = results
        self._unread_buffer = True
        ClusteringV2XManager.Communication_Volume_Inside_Cluster_Broadcast += sys.getsizeof(objects)

    def read_buffer(self):
        if self._unread_buffer:
            self._unread_buffer = False
            return self._recieved_buffer
        else:
            return {'source': "0",
                    'objects':
                    {
                        'vehicles': [],
                        'traffic_lights': []
                    },
                    'results':
                    {
                        'psm': None,
                        'rm': None,
                    }
                }

    def beacon(self):
        """
        Broadcast a beacon frame to nearby vehicles.
        The beacon contains:
            - Position and confidence
            - Motion state (speed, direction, acceleration)
            - Computing capability and communication quality
            - Cooperative perception model information
        """
        beacon = {
            'vehicle_id': self.vehicle_id,
            'position': self.get_ego_pos(),
            'speed': self.get_ego_speed(),
            'direction': self.get_ego_dir(),
            'computing_capability': self.computing_capability,  # Placeholder for computing capability
            'communication_quality': 1.0, # Placeholder for communication quality
            'perception_model': self.cp_model,  # Placeholder for model info
            'cluster_head': self.cluster_state['cluster_head'],  # Placeholder for cluster head info
            'priority_score': self.compute_priority_score()
        }
        return beacon

    def compute_similarity(self, neighbor_beacon):
        """
        Compute the similarity score with a neighbor based on their beacon.

        Parameters
        ----------
        neighbor_beacon : dict
            The beacon frame received from the neighbor.

        Returns
        -------
        similarity_score : float
            The computed similarity score.
        """
        # Extract information from the neighbor beacon
        neighbor_pos = neighbor_beacon['position'].location
        neighbor_speed = neighbor_beacon['speed']
        neighbor_direction = neighbor_beacon['direction']
        neighbor_model = neighbor_beacon['perception_model']

        # Compute distance and speed difference
        distance = compute_distance(self.ego_pos[-1].location, neighbor_pos)
        speed_diff = abs(self.get_ego_speed() - neighbor_speed)
        cosTheta = calculate_cos(self.get_ego_dir(), neighbor_direction)
        # Compute alpha (position weight) based on neighbor count
        N_neigh = len(self.cluster_state['neighbors'])
        alpha = 1 / (1 + math.exp(-self.cluster_params['kappa'] * (N_neigh - self.cluster_params['N_th'])))
        beta = 1 - alpha

        # Compute similarity score
        distance_term = alpha * math.exp(-distance / self.cluster_params['d0'])
        speed_term = beta * math.exp(-speed_diff * (1-cosTheta) / self.cluster_params['s0'])
        similarity_score = distance_term + speed_term

        # Penalize if perception models differ
        if neighbor_model != self.cp_model:
            similarity_score *= 0.01

        return similarity_score

    def compute_average_cluster_speed(self):
        #print(self.cluster_state['members'])
        speed = [member_data['speed'] for member_id, member_data in self.cluster_state['members'].items()]
        return sum(speed) / len(speed) if len(speed) > 0 else 0
    
    def compute_priority_score(self):
        """
        Compute the priority score for cluster head election.

        Returns
        -------
        priority_score : float
            The computed priority score.
        """
        # Extract parameters
        w1 = self.cluster_params['w1']
        w2 = self.cluster_params['w2']
        w3 = self.cluster_params['w3']

        # Extract local information
        communication_quality = self.communication_quality
        computing_capability = self.computing_capability
        speed = self.get_ego_speed()

        # Compute average cluster speed (if part of a cluster)
        avg_speed = self.compute_average_cluster_speed()

        # Compute priority score
        speed_consistency = 1 - abs(speed - avg_speed) if avg_speed is not None else 1.0
        priority_score = (w1 * communication_quality +
                        w2 * computing_capability +
                        w3 * speed_consistency)
        
        self.cluster_state['priority_score'] = priority_score
        return priority_score

    def update_cluster(self):
        """
        Update cluster membership based on similarity scores and network conditions.
        """
        # Check if the vehicle is already part of a cluster
        if self.cluster_state['cluster_head'] is None:
            self.look_for_existed_clusters()
            self.try_to_create_cluster()

        self.update_cluster_membership()
        self.elect_cluster_head()

    def look_for_existed_clusters(self):
        if self.cluster_state['cluster_head'] is not None:
            return
        # Find the best cluster head candidate
        self.cluster_state['members'] = {}
        self.cluster_state['members'][self.vehicle_id] = self.beacon()
        for vehicle_id, neighbor_data in self.cluster_state['neighbors'].items():
            similarity_score = self.cluster_state['similarity_scores'].get(vehicle_id, 0)
            RSSI = neighbor_data['RSSI'] if 'RSSI' in neighbor_data else 0

            # Adjusted join condition
            adjusted_threshold = self.cluster_params['eta_join'] * (1 + RSSI / self.cluster_params['RSSI_max'])
            if similarity_score > adjusted_threshold and neighbor_data['cluster_head'] == neighbor_data['vehicle_id']:
                # Exists a suitable cluster head candidate, join the cluster
                print('Vehicle %s joined cluster %s with similarity_score %f\n' % (self.vehicle_id, neighbor_data['cluster_head'], similarity_score))
                self.cluster_state['cluster_head'] = neighbor_data['cluster_head']
                vm = self.cav_nearby.get(neighbor_data['vehicle_id'], {'v2x_manager': None})['v2x_manager']
                if vm:
                    vm.cluster_state['members'][vehicle_id] = self.beacon()
                else:
                    print(f"{neighbor_data['vehicle_id']} is not neighbor")
                break


    def try_to_create_cluster(self):
        # Decide whether to join a cluster or create a new one
        if self.cluster_state['cluster_head'] is not None:
            return
        # Calculate the probability of creating a new cluster
        sim_avg = sum(self.cluster_state['similarity_scores'].values()) / len(self.cluster_state['similarity_scores']) if self.cluster_state['similarity_scores'] else 0
        p_create = sigmoid((sim_avg - self.cluster_params['eta_create']) / self.cluster_params['sigma'])
        if random.random() < p_create:
            # Create a new cluster
            self.cluster_state['cluster_head'] = self.vehicle_id
            print(f'Vehicle {self.vehicle_id} created new cluster with p_create:{p_create}\n')


    def update_cluster_membership(self):
        """
        Update cluster membership based on similarity scores and hysteresis mechanism.
        """

        # Check if the vehicle should leave the cluster
        similarity_score = self.cluster_state['similarity_scores'].get(self.cluster_state['cluster_head'], 0)
        if self.is_cluster_head() or self.cluster_state['cluster_head'] is None:
            similarity_score = 1.0
        if similarity_score < self.cluster_params['eta_leave']:
            print(f'Vehicle {self.vehicle_id} left cluster {self.cluster_state["cluster_head"]} with similarity_score {similarity_score}\n')
            self.cluster_state['members'].clear()
            self.cluster_state["cluster_head"] = None
            self.update_cluster()
            return

        #self.cluster_state['members'].clear()

        members = self.cluster_state['members'].copy()
        for vehicle_id, neighbor_data in members.items():
            if neighbor_data['cluster_head'] != self.cluster_state['cluster_head']:
                #no longer in the same cluster
                self.cluster_state['members'].pop(vehicle_id, None)

        for vehicle_id, neighbor_data in self.cluster_state['neighbors'].items():
            if neighbor_data['cluster_head'] == self.cluster_state['cluster_head']:
                #in the same cluster
                self.cluster_state['members'][vehicle_id] = neighbor_data
                

    def promote_shadow_head(self):
        """
        Update shadow head based on cluster head status and timeout.
        """
        # Elect a new shadow head if necessary
        if self.cluster_state['shadow_head'] is None:
            self.elect_cluster_head()
            return
        # Promote shadow head to cluster head
        self.cluster_state['cluster_head'] = self.cluster_state['shadow_head']
        self.cluster_state['shadow_head'] = None
        print(f'Shadow cluster head Vehicle {self.cluster_state["cluster_head"]} has been promoted to cluster head')
        self.elect_cluster_head()

    def elect_cluster_head(self):
        """
        Elect a shadow head from the cluster members.
        """
        #print('members:', self.cluster_state['members'].keys())
        if not self.cluster_state['members']:
            return
        # Elect the member with the highest priority score
        highest_priority = -1
        second_highest_priority = -1
        cluster_head = None
        shadow_head = None
        for vehicle_id, member_data in self.cluster_state['members'].items():
            if 'priority_score' in member_data:
                if member_data['priority_score'] > highest_priority:
                    highest_priority = member_data['priority_score']
                    cluster_head = vehicle_id
                elif member_data['priority_score'] > second_highest_priority:
                    second_highest_priority = member_data['priority_score']
                    shadow_head = vehicle_id
                    
        if cluster_head is not None and self.cluster_state['cluster_head'] is None:
            self.cluster_state['cluster_head'] = cluster_head
        if shadow_head is not None and self.cluster_state['shadow_head'] is None:
            self.cluster_state['shadow_head'] = shadow_head

        members = [vehicle_id for vehicle_id in self.cluster_state['members']]
        if self.is_cluster_head():
            print(f"cluster head:{self.vehicle_id}, shadow head:{self.cluster_state['shadow_head']}, cluster members:{members}\n")

    def search(self, receive_beacons=False):
        """
        Search for nearby vehicles and update cluster state based on received beacons.
        """
        vehicle_manager_dict = self.cav_world.get_vehicle_managers()

        self.tick += 1
        receive_beacons = self.tick >= (20/self.beacon_frequency) #send and receive beacon on 20/10=2(Hz)

        for vid, vm in vehicle_manager_dict.items():
            vehicle_id = vm.vehicle.id
            # Skip invalid or self
            if vehicle_id == self.vehicle_id or not vm.v2x_manager.get_ego_pos():
                continue
            distance = compute_distance(self.ego_pos[-1].location, vm.v2x_manager.get_ego_pos().location)
            # update v2x_manager.cav_nearby
            if distance < self.communication_range:
                self.cav_nearby.update({vm.vehicle.id: {
                    'vehicle_manager': vm,
                    'v2x_manager': vm.v2x_manager
                }})
            else:
                self.cav_nearby.pop(vm.vehicle.id, None)

            if receive_beacons:
                #print(f"{self.vehicle_id}  receive_beacons, tick: {self.tick}")
                self.tick = 0
                # Receive beacon from neighbor
                neighbor_beacon = vm.v2x_manager.beacon()

                # Update neighbor information
                if distance < self.communication_range:
                    self.cluster_state['neighbors'][vehicle_id] = neighbor_beacon
                    self.cluster_state['similarity_scores'][vehicle_id] = self.compute_similarity(neighbor_beacon)
                else:
                    self.cluster_state['neighbors'].pop(vehicle_id, None)
                    self.cluster_state['similarity_scores'].pop(vehicle_id, None)
                    if vehicle_id in self.cluster_state['members']:
                        # Remove from cluster membership
                        self.cluster_state['members'].pop(vehicle_id, None)
                    if vehicle_id == self.cluster_state['cluster_head']:
                        # Cluster head has moved out of range
                        self.cluster_state['cluster_head'] = None
                        self.promote_shadow_head()
                    elif vehicle_id == self.cluster_state['shadow_head']:
                        self.cluster_state['shadow_head'] = None
        if receive_beacons:
            # Update cluster membership
            # print("Update cluster membership")
            self.update_cluster()
            
