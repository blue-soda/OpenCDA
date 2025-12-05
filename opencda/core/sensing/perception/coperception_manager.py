from collections import OrderedDict
from pympler.asizeof import asizeof
from opencda.log.logger_config import logger

class CoperceptionManager:
    def __init__(self, vid, v2x_manager, coperception_libs, enable_network=False, network_manager=None):
        self.vid = vid
        self.v2x_manager = v2x_manager
        self.coperception_libs = coperception_libs
        self.ego_data_dict = None
        self.vehicles = None

        self.enable_network = enable_network
        self.network_manager = network_manager
        self.uploaded_cavs = {}
        self.uploading_cavs = {}
        self.all_cavs = {}
        self.uploading_data = None
        self.uploading_data_size = {}
        self.timeout_slots = 3 # number of time slots to wait before re-uploading data from a cav

    def get_coperception_cavs_dict(self) -> dict:
        # dict of {vid: {'vehicle_manager': vm, 'v2x_manager': v2x_manager}}
        return self.v2x_manager.cav_nearby
    
    def communicate(self, is_ego=False, ego_lidar_pose=None, use_ego_vehicles=False):
        data = {}
        if self.uploading_data:
            data = self.uploading_data
        else:
            self.all_cavs = self.get_coperception_cavs_dict()
            logger.info(f"CoperceptionManager {self.vid} preparing data from {list(self.all_cavs.keys())} CAVs.")
            data = self.prepare_and_transform_data_from_dict(self.all_cavs, is_ego, ego_lidar_pose, use_ego_vehicles)
            self.uploading_data = data
        if not self.enable_network:
            self.uploading_data = None
            return data
        return self.communicate_via_network()
    
    def communicate_via_network(self):
        # print("Coperception via network...")
        self.send_cams_via_network()
        return self.receive_cams_via_network()

    def send_cams_via_network(self):
        current_time_slot = self.network_manager.current_time_slot
        for cav_id, vm_dict in self.all_cavs.items():
            if cav_id in self.uploaded_cavs:
                continue
            if cav_id in self.uploading_cavs:
                if self.uploading_cavs[cav_id] > current_time_slot - self.timeout_slots:
                    continue
                else:
                    print(f"cav {cav_id} timeout, current_time_slot: {current_time_slot}, start_slot: {self.uploading_cavs[cav_id]}")
            self.uploading_cavs[cav_id] = current_time_slot
            cav_v2x_manager = vm_dict['v2x_manager']
            # data_size = asizeof(cav_data) - data_size_received
            if cav_id not in self.uploading_data_size:
                cav_data = self.uploading_data.get(cav_id, None)
                data_size = asizeof(cav_data) 
                self.uploading_data_size[cav_id] = data_size
                self.v2x_manager.scheduler.record_data_size_infos({(cav_id, self.v2x_manager.vehicle_id) : data_size}) # let scheduler know the data size 
                print(f"cav {cav_id} is uploading its data to {self.vid} for the FIRST time, size: {data_size} bytes at {self.network_manager.current_time_slot}.")
            else:
                data_size = self.uploading_data_size[cav_id]
                print(f"cav {cav_id} is uploading its data to {self.vid} AGAIN, size: {data_size} bytes at {self.network_manager.current_time_slot}.")
            self.v2x_manager.scheduler.schedule(cav_v2x_manager, self.v2x_manager, data_size)

    def receive_cams_via_network(self):
        # cams = self.network_manager.get_received_cams()
        cams = self.network_manager.get_received_cams(self.vid)
        received_data = {}
        for cam in cams.values():
            sender_id = cam.get('sender_id')
            receiver_id = cam.get('receiver_id')
            packet_size = cam.get('packet_size')
            data_size = self.uploading_data_size.get(sender_id, None)
            if not data_size:
                continue
            if packet_size < data_size * 0.90:
                self.uploading_data_size[sender_id] -= packet_size
                self.network_manager.pop_received_cams(self.vid)
                print(f"cav {sender_id} data upload to {receiver_id} incomplete. Received size: {packet_size} bytes, expected size: {data_size} bytes.")
                continue
            
            print(f"cav {sender_id} data upload to {receiver_id} succeeded. Received size: {packet_size} bytes, expected size: {data_size} bytes.")
            delay_infos = self.network_manager.pop_received_cams(receiver_id, sender_id)
            print(f"cav {sender_id} communication delay info: {delay_infos}")
            self.v2x_manager.scheduler.record_communication_delay_infos(delay_infos)
            data = self.uploading_data.get(sender_id, None)
            if data:
                self.uploaded_cavs[sender_id] = self.network_manager.current_time_slot
                received_data[sender_id] = data
                print(f"cav {sender_id} has uploaded its data to {self.vid} via network at {self.network_manager.current_time_slot}.")
            else:
                logger.warning(f"cav {sender_id} data not found in uploading_data of {self.vid}.")
        return received_data


    def all_data_uploaded(self, percent=0.6):
        uploaded_num = len(self.uploaded_cavs)
        all_cavs_num = len(self.all_cavs)
        if all_cavs_num == 0 or self.enable_network is False:
            return True
        ok = uploaded_num / all_cavs_num >= percent
        logger.info(f"Coperception data uploaded: {uploaded_num}/{all_cavs_num} ({uploaded_num / all_cavs_num:.2%})")
        return ok

    def clear_uploaded_and_uploading(self):
        self.uploaded_cavs = {}
        self.uploading_cavs = {}
        self.uploading_data = None

    def calculate_transformation(self, cav_id, cav_data, ego_pose=None):
        if ego_pose is None:
            t_matrix = self.coperception_libs.load_transformation_matrix(self.ego_data_dict, cav_data[cav_id]['params'])
        else:
            t_matrix = self.coperception_libs.load_transformation_matrix_from_pose(ego_pose, cav_data[cav_id]['params']['lidar_pose'])
        cav_data[cav_id]['params'].update(t_matrix)
        return cav_data
    
    def get_data_from_lidar(self, lidar):
        return lidar.data # 这一步获取了点云数据, 可以改为lidar.get_local_points_by_grid_ids([])按照网格划分获取点云数据
    
    def prepare_data(self, cav_id, camera, lidar, pos, localizer, agent, is_ego, use_ego_vehicles=False):
        data = {cav_id: OrderedDict()}
        data[cav_id]['ego'] = is_ego
        data[cav_id]['time_delay'] = self.coperception_libs.time_delay
        data[cav_id]['params'] = {}
        camera_data = self.coperception_libs.load_camera_data(lidar, camera)
        ego_data = self.coperception_libs.load_ego_data(localizer)
        plan_trajectory_data = self.coperception_libs.load_plan_trajectory(agent)
        lidar_pose_data = self.coperception_libs.load_cur_lidar_pose(lidar)
        data[cav_id]['params'].update(plan_trajectory_data)
        data[cav_id]['params'].update(camera_data)
        data[cav_id]['params'].update(ego_data)
        data[cav_id]['params'].update(lidar_pose_data)
        data[cav_id].update({'lidar_np': self.get_data_from_lidar(lidar)})
        # get base_data_dict
        if is_ego:
            self.ego_data_dict = data[cav_id]['params']
            self.vehicles = self.coperception_libs.load_vehicles(cav_id, pos, lidar)

        if use_ego_vehicles and self.vehicles:  # use ego's vehicles for others
            data[cav_id]['params'].update(self.vehicles)
        else:
            data[cav_id]['params'].update(self.coperception_libs.load_vehicles(cav_id, pos, lidar))

        return data

    def prepare_and_transform_data(self, vid, camera, lidar, pos, localizer, agent, is_ego, ego_lidar_pose=None, use_ego_vehicles=False):
        transformed_data = self.prepare_data(
            cav_id=vid,
            camera=camera,
            lidar=lidar,
            pos=pos,
            localizer=localizer,
            agent=agent,
            is_ego=is_ego,
            use_ego_vehicles=use_ego_vehicles
        )
        transformed_data = self.calculate_transformation(
            cav_id=vid,
            cav_data=transformed_data,
            ego_pose=ego_lidar_pose
        )
        return transformed_data

    def prepare_and_transform_data_from_managers(self, v2x_manager, localizer, agent, is_ego, ego_lidar_pose=None, use_ego_vehicles=False):
        nearby_data = self.prepare_and_transform_data(
            vid=v2x_manager.vehicle_id,
            camera=v2x_manager.get_ego_rgb_image(),
            lidar=v2x_manager.get_ego_lidar(),
            pos=v2x_manager.get_ego_pos(),
            localizer=localizer,
            agent=agent,
            is_ego=is_ego,
            ego_lidar_pose=ego_lidar_pose,
            use_ego_vehicles=use_ego_vehicles
        )
        return nearby_data
    
    def prepare_and_transform_data_from_dict(self, data: dict, is_ego=False, ego_lidar_pose=None, use_ego_vehicles=False):
        transformed_data = {}
        for vid, nearby_data_dict in data.items():
            if not nearby_data_dict:
                continue
            nearby_vm = nearby_data_dict['vehicle_manager']
            nearby_v2x_manager = nearby_data_dict['v2x_manager']
            nearby_data = self.prepare_and_transform_data_from_managers(
                v2x_manager=nearby_v2x_manager,
                localizer=nearby_vm.localizer,
                agent=nearby_vm.agent,
                is_ego=is_ego,
                ego_lidar_pose=ego_lidar_pose,
                use_ego_vehicles=use_ego_vehicles
            )
            transformed_data[vid] = nearby_data[vid]
        return transformed_data
    
    def send_objects_info_buffer(self, target_id, objects):
        target = self.get_coperception_cavs_dict().get(target_id, None)
        target_v2x_manager = target['v2x_manager'] if target else None
        if target_v2x_manager is not None:
            target_v2x_manager.set_buffer(source_id=self.vid)
            target_v2x_manager.set_buffer(objects=objects)