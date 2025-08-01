from opencda.core.common.cav_world import CavWorld
from opencda.core.common.v2x_manager import V2XManager
import math
from opencda.core.common.misc import compute_distance
import random
from opencda.log.logger_config import logger
# from pympler.asizeof import asizeof
from sys import getsizeof as asizeof
import weakref

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


# TODO: 
# 1. 簇内聚类分工，分担计算任务
# 2. 感知范围： LOS邻居并集最大
# 3. 簇内聚类做感知融合，多个聚类上传给簇头晚期融合（簇间再融合）


class ClusteringV2XManager(V2XManager):
    def __init__(self, cav_world, config_yaml, vid, vehicle_id=None, cluster_yaml=None):
        super(ClusteringV2XManager, self).__init__(cav_world, config_yaml, vid, vehicle_id)
        
        self.cp_model = 'default_model'

        self.t = 0 #Timer 
 
        self.beacon_frequency = 4 #(Hz)
        self.receive_beacon = False
        # 分簇协议相关参数 
        # self.cluster_params = {
        #     'd0': 50.0,           # 距离归一化参数 (单位: m)
        #     's0': 5.0,            # 速度归一化参数 (单位: m/s)
        #     'N_th': 4,            # 邻居数阈值
        #     'N_max': 4,            # 最大簇成员数
        #     'kappa': 0.05,         # 权重调节系数
        #     'eta_join': 0.60,      # 加入簇的阈值
        #     'eta_create': 0.40,    # 创建簇的阈值
        #     'RSSI_max': 100,      # 最大接收信号强度
        #     'epsilon': 0.1,       # 协同感知模型差异阈值
        #     'sigma': 1.0,         # 创建簇的概率调节参数
        #     'w1': 0.4,  # 通信质量权重
        #     'w2': 0.3,  # 计算能力权重
        #     'w3': 0.2,  # 速度一致性权重
        #     'T_timeout': 1.0,  # 簇头超时时间 (单位: s)
        #     'shadow_timeout': 0.5,  # 影子簇头超时时间 (单位: s)
        #     'eta_leave': 0.35,  # 离开阈值 (eta_join - delta_eta)
        #     'eta_elect': 0.10 #选举阈值(超过当前簇头优先级得分 + eta_elect 才能当选)
        # }
        self.cluster_params = cluster_yaml

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

        self.ego_must_be_leader = cluster_yaml.get('ego_must_be_leader', False)
        # print("ego_must_be_leader", self.ego_must_be_leader,  self.vehicle_id,  self.cav_world.ego_id)

    def id_to_rgb(self):
        vehicle_id = self.cluster_state['cluster_head']
        if vehicle_id is None:
            self.rgb = (255, 255, 0)
            return 

        hash_value = hash(f"vehicle_{vehicle_id}") & 0xFFFFFF
        r = int((hash_value >> 16) & 0xFF)
        g = int((hash_value >> 8) & 0xFF)
        b = int(hash_value & 0xFF)
        # r = (vehicle_id * 79) % 256
        # g = (vehicle_id * 101) % 256
        # b = (vehicle_id * 127) % 256
        r = 255 if r > 127 else 0
        g = 255 if g > 127 else 0
        b = 255 if b > 127 else 0
        self.rgb = (r, g, b)

    def get_cluster_members(self):
        """
        Returns a processed version of self.cluster_state
        """
        # Extract and transform the needed fields
        cluster_head_id = self.cluster_state.get('cluster_head')
        members_dict = self.cluster_state.get('members', {})
        neighbors_dict = self.cluster_state.get('neighbors', {})

        # Resolve cluster head
        cluster_head_vm = self.cav_world.get_vehicle_manager(cluster_head_id).v2x_manager if cluster_head_id is not None else None

        # Resolve members
        members_vm = {}
        for vehicle_id in members_dict:
            vm = self.cav_world.get_vehicle_manager(vehicle_id)
            members_vm[vehicle_id] = vm.v2x_manager

        # Resolve neighbors
        neighbors_vm = {}
        for vehicle_id in neighbors_dict:
            vm = self.cav_world.get_vehicle_manager(vehicle_id)
            neighbors_vm[vehicle_id] = vm.v2x_manager

        return {
            'cluster_head': cluster_head_vm,
            'members': members_vm,
            'neighbors': neighbors_vm,
        }

        
    def is_cluster_head(self):
        # logger.debug('vehicle_id', self.vehicle_id, self.cluster_state['cluster_head'], self.vehicle_id == self.cluster_state['cluster_head'])
        return self.vehicle_id == self.cluster_state['cluster_head']
    
    def tick(self):
        self.t += 1
        self.receive_beacon = self.t >= (self.cav_world.frequency/self.beacon_frequency) #send and receive beacon on 20/10=2(Hz)
        if self.receive_beacon:
            self.t = 0
        return self.receive_beacon

    def beacon(self):
        """
        Broadcast a beacon frame to nearby vehicles.
        The beacon contains:
            - Position and confidence
            - Motion state (speed, direction, acceleration)
            - Computing capability and communication quality
            - Cooperative perception model information
        """
        self.cluster_state['priority_score'] = self.compute_priority_score()
        beacon = {
            'vehicle_id': self.vehicle_id,
            'position': self.get_ego_pos(),
            'speed': self.get_ego_speed(),
            'direction': self.get_ego_dir(),
            'computing_capability': self.computing_capability,  # Placeholder for computing capability
            'communication_quality': 1.0, # Placeholder for communication quality
            'perception_model': self.cp_model,  # Placeholder for model info
            'cluster_head': self.cluster_state['cluster_head'],  # Placeholder for cluster head info
            'priority_score': self.cluster_state['priority_score'],
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
        speed_term = beta * math.exp(-speed_diff / self.cluster_params['s0']) * (cosTheta)
        similarity_score = distance_term + speed_term

        # Penalize if perception models differ
        if neighbor_model != self.cp_model:
            similarity_score *= 0.01

        return similarity_score

    def compute_average_cluster_speed(self):
        #logger.debug(self.cluster_state['members'])
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
        # print(self.vehicle_id, self.ego_id, self.cav_world.ego_id)
        if self.ego_must_be_leader and self.vehicle_id == self.cav_world.ego_id:
            # print("100!")
            return 100

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
        speed_consistency = 1.5 - sigmoid(abs(speed - avg_speed)) if avg_speed is not None else 1.0
        priority_score = (w1 * communication_quality +
                        w2 * computing_capability +
                        w3 * speed_consistency)
        
        # self.cluster_state['priority_score'] = priority_score
        return priority_score

    def update_cluster_join(self):
        """
        Update cluster membership based on similarity scores and network conditions.
        """
        # Check if the vehicle is already part of a cluster
        if self.cluster_state['cluster_head'] is None:
            self.look_for_existed_clusters()
            self.try_to_create_cluster()

        # self.elect_cluster_head()
        # self.check_current_leader()
        # self.update_cluster_membership()
        # self.check_current_members()

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
            if vehicle_id == self.cav_world.ego_id:
                logger.debug(f"vehicle {self.vehicle_id} checks ego, get similarity_score {similarity_score} and adjusted_threshold {adjusted_threshold}, neighbor_data['cluster_head']: {neighbor_data['cluster_head']}, neighbor_data['vehicle_id']: {neighbor_data['vehicle_id']}")
            if similarity_score > adjusted_threshold and neighbor_data['cluster_head'] == neighbor_data['vehicle_id']:
                # Exists a suitable cluster head candidate, try to join the cluster

                vm = self.cav_nearby.get(vehicle_id, {'v2x_manager': None})['v2x_manager']
                if vm:
                    if len(vm.cluster_state['members']) >= self.cluster_params['N_max']:
                        logger.debug(f"{neighbor_data['vehicle_id']}'s cluster is full, {len(vm.cluster_state['members'])} >= {self.cluster_params['N_max']} ")
                        continue
                    logger.debug('Vehicle %s joined cluster %s with similarity_score %f\n' % (self.vehicle_id, neighbor_data['cluster_head'], similarity_score))
                    self.cluster_state['cluster_head'] = neighbor_data['cluster_head']
                    vm.cluster_state['members'][self.vehicle_id] = self.beacon()
                else:
                    # the cluster head is too far 
                    logger.debug(f"{neighbor_data['vehicle_id']} is not neighbor")
                    continue

                break


    def try_to_create_cluster(self):
        # Decide whether to join a cluster or create a new one
        if self.cluster_state['cluster_head'] is not None:
            return
        # Calculate the probability of creating a new cluster
        sim_avg = sum(self.cluster_state['similarity_scores'].values()) / len(self.cluster_state['similarity_scores']) if self.cluster_state['similarity_scores'] else 0
        p_create = sigmoid((self.cluster_params['eta_create'] - sim_avg) / self.cluster_params['sigma'])
        if random.random() < p_create:
            # Create a new cluster
            self.cluster_state['cluster_head'] = self.vehicle_id
            logger.debug(f'Vehicle {self.vehicle_id} created new cluster with p_create:{p_create}')


    def update_cluster_membership(self):
        """
        Update cluster membership based on similarity scores and hysteresis mechanism.
        """
        # Check if the vehicle should leave the cluster
        similarity_score = self.cluster_state['similarity_scores'].get(self.cluster_state['cluster_head'], 0)
        if self.is_cluster_head() or self.cluster_state['cluster_head'] is None:
            similarity_score = 1.0
        if similarity_score < self.cluster_params['eta_leave']:
            vm = self.cav_nearby.get(self.cluster_state['cluster_head'], {'v2x_manager': None})['v2x_manager']
            if self.ego_must_be_leader and self.cluster_state['cluster_head'] == self.cav_world.ego_id:
                if vm and len(vm.cluster_state['members']) <= self.cluster_params['N_max']:
                    logger.debug(f"{self.vehicle_id} wanted to leave ego's cluster with similarity_score {similarity_score}, but stopped")
                    return
            logger.debug(f'Vehicle {self.vehicle_id} left cluster {self.cluster_state["cluster_head"]} with similarity_score {similarity_score}')
            self.cluster_state['members'].clear()
            self.cluster_state["cluster_head"] = None
            if vm and self.vehicle_id in vm.cluster_state['members']:
                del vm.cluster_state['members'][self.vehicle_id]
            self.update_cluster_join()
            return


    def check_current_leader(self):
        # Check if the current cluster head is really a leader
        cur_cluster_head = self.cluster_state['cluster_head']
        if cur_cluster_head != self.vehicle_id and cur_cluster_head is not None:
            if not cur_cluster_head in self.cluster_state['neighbors'].keys():
                logger.warning(f"Vehicle {self.vehicle_id} 's cluster head {cur_cluster_head} is not even a neighbor.")
                self.cluster_state['cluster_head'] = None
                return
            head_info = self.cluster_state['neighbors'][cur_cluster_head]
            if(head_info['cluster_head'] != cur_cluster_head):
                logger.debug(f"Vehicle {self.vehicle_id} 's cluster head {cur_cluster_head} is updated to {head_info['cluster_head']}.")
                self.cluster_state['cluster_head'] = head_info['cluster_head']
                


    def check_current_members(self):
        members = self.cluster_state['members'].copy()
        for vehicle_id, neighbor_data in members.items():
            if neighbor_data['cluster_head'] != self.cluster_state['cluster_head']:
                #no longer in the same cluster
                self.cluster_state['members'].pop(vehicle_id, None)

        for vehicle_id, neighbor_data in self.cluster_state['neighbors'].items():
            if neighbor_data['cluster_head'] == self.cluster_state['cluster_head']:
                #in the same cluster
                self.cluster_state['members'][vehicle_id] = neighbor_data
                
        if self.is_cluster_head():
            members = [vehicle_id for vehicle_id in self.cluster_state['members']]
            logger.debug(f"cluster head:{self.cluster_state['cluster_head']}, shadow head:{self.cluster_state['shadow_head']}, cluster members:{members}")

    def promote_shadow_head(self):
        """
        Update shadow head based on cluster head status and timeout.
        """
        # Elect a new shadow head if necessary
        cur_shadow_head = self.cluster_state['shadow_head']
        if cur_shadow_head is None:
            self.update_cluster_join()
            return
        # Promote shadow head to cluster head
        self.cluster_state['cluster_head'] = cur_shadow_head
        self.cluster_state['shadow_head'] = None
        logger.debug(f'Shadow cluster head Vehicle {cur_shadow_head} has been promoted to cluster head')
        self.elect_cluster_head()

    def elect_cluster_head(self):
        """
        Elect a shadow head from the cluster members.
        """
        #logger.debug('members:', self.cluster_state['members'].keys())
        if not self.cluster_state['members']:
            return
        # let the current leader hold the election, then the members will know the results next round.
        if self.is_cluster_head():
            # Elect the member with the highest priority score
            highest_priority = -1
            second_highest_priority = -1
            cluster_head = None
            shadow_head = None

            cur_priority_score = self.cluster_state['priority_score']
            eta_elect = self.cluster_params['eta_elect']
            highest_priority = cur_priority_score * (1 + eta_elect)
            for vehicle_id, member_data in self.cluster_state['members'].items():
                if 'priority_score' in member_data:
                    if member_data['priority_score'] > highest_priority:
                        highest_priority = member_data['priority_score']
                        cluster_head = vehicle_id
                        logger.debug(f"{vehicle_id}_{highest_priority} win the election over {self.vehicle_id}_{cur_priority_score}.")
                    elif member_data['priority_score'] > second_highest_priority:
                        second_highest_priority = member_data['priority_score']
                        shadow_head = vehicle_id
                        
            if cluster_head is not None: #and self.cluster_state['cluster_head'] is None:
                self.cluster_state['cluster_head'] = cluster_head
            if shadow_head is not None: #and self.cluster_state['shadow_head'] is None:
                self.cluster_state['shadow_head'] = shadow_head

            members = [vehicle_id for vehicle_id in self.cluster_state['members']]
            logger.debug(f"cluster head:{self.cluster_state['cluster_head']} with priority_score {highest_priority:.3f}, shadow head:{self.cluster_state['shadow_head']} with priority_score {second_highest_priority:.3f}")

    def search(self):
        """
        Search for nearby vehicles and update cluster state based on received beacons.
        """
        vehicle_manager_dict = self.cav_world.get_vehicle_managers()

        if self.ego_must_be_leader and self.vehicle_id == self.cav_world.ego_id:
            self.cluster_state['cluster_head'] = self.vehicle_id
            # logger.debug("ego_must_be_leader!")

        self.tick()

        for vid, vm in vehicle_manager_dict.items():
            vehicle_id = vm.vehicle.id
            # Skip invalid or self
            if vehicle_id == self.vehicle_id or not vm.v2x_manager.get_ego_pos():
                continue
            distance = compute_distance(self.ego_pos[-1].location, vm.v2x_manager.get_ego_pos().location)
            # update v2x_manager.cav_nearby
            if distance < self.communication_range:
                self.cav_nearby.update({vm.vehicle.id: {
                    'vehicle_manager': weakref.ref(vm)(),
                    'v2x_manager': weakref.ref(vm.v2x_manager)()
                }})
            else:
                self.cav_nearby.pop(vm.vehicle.id, None)

            if self.receive_beacon:
                #logger.debug(f"{self.vehicle_id}  receive_beacons, tick: {self.tick}")
                # Receive beacon from neighbor
                neighbor_beacon = vm.v2x_manager.beacon()

                # Update neighbor information
                if distance < self.communication_range and vm.is_ok:
                    self.cluster_state['neighbors'][vehicle_id] = neighbor_beacon
                    self.cluster_state['similarity_scores'][vehicle_id] = self.compute_similarity(neighbor_beacon)
                    if CavWorld.network_manager:
                        objects_size = asizeof(neighbor_beacon)
                        CavWorld.network_manager._update_communication_stats(objects_size, "control")
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

        # if self.receive_beacon:
        #     # Update cluster membership
        #     # logger.debug("Update cluster membership")
        #     self.update_cluster()
        #     self.id_to_rgb()
        #     if self.scheduler is not None and self.scheduler_type == 'clusterbased' and self.is_cluster_head():
        #         self.scheduler.update_scheduler(self.get_cluster_members())
