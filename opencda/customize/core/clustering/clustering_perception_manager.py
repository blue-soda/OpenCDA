from collections import OrderedDict
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager
from opencda.core.common.v2x_manager \
    import V2XManager
from opencda.core.common.cav_world \
    import CavWorld
from opencda.customize.core.clustering.clustering_coperception_manager import ClusteringCoperceptionManager
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion, o3d_visualizer_show_coperception, o3d_predict_bbox_to_object
from pympler.asizeof import asizeof
from opencda.core.sensing.perception.coperception_libs import CoperceptionLibs
from opencda.log.logger_config import logger

class ClusteringPerceptionManager(PerceptionManager):
    #static ego_data_dict
    ego_lidar_pose = None
    ego_vm = None
    ego_predict_box_tensors = []
    ego_predict_scores = []
    ego_gt_box_tensors = []
    ego_did_cp = False

    def __init__(self, v2x_manager, localization_manager, behavior_agent, vehicle,
                 config_yaml, cav_world, data_dump=False, carla_world=None, infra_id=None, enable_network=False, cluster_config=None):
        super().__init__(v2x_manager, localization_manager, behavior_agent, vehicle,
                 config_yaml, cav_world, data_dump, carla_world, infra_id, enable_network)
        self.communication_volume = 0.0
        self.co_manager = ClusteringCoperceptionManager(self.id, v2x_manager, self.coperception_libs)
        if ClusteringPerceptionManager.ego_vm is None:
            ClusteringPerceptionManager.ego_vm = cav_world.get_ego_vehicle_manager()
        self.do_cp = 0
        self.cp_data = {}
        self.ego_data = {}
        self.is_ego = False
        self.apply_late_fusion = False
        self.record_all_cavs = False
        self.predict_box_tensor = None
        self.gt_box_tensor = None
        self.predict_box_tensor_fusion = None
        self.gt_box_tensor_fusion = None
        if cluster_config:
            self.apply_late_fusion = cluster_config['apply_late_fusion']
            self.record_all_cavs = cluster_config['record_all_cavs']            

    #TODO: if self.record_all_cavs: do late fusion for all cavs in the cluster, not only ego.
    @staticmethod
    def update_ego_lidar_pose():
        ego_vm = ClusteringPerceptionManager.ego_vm
        ego_v2x_manager = ego_vm.v2x_manager
        lidar=ego_v2x_manager.get_ego_lidar()
        ClusteringPerceptionManager.ego_lidar_pose = CoperceptionLibs.load_cur_lidar_pose(lidar)['lidar_pose']

    @staticmethod
    def clear():
        ClusteringPerceptionManager.ego_predict_box_tensors = []
        ClusteringPerceptionManager.ego_predict_scores = []
        ClusteringPerceptionManager.ego_gt_box_tensors = []
    
    @staticmethod
    def get_boxes_size():
        return asizeof(ClusteringPerceptionManager.ego_predict_box_tensors) + \
            asizeof(ClusteringPerceptionManager.ego_predict_scores) + \
            asizeof(ClusteringPerceptionManager.ego_gt_box_tensors)


    def detect(self, ego_pos):
        """
        Detect surrounding objects. Currently only vehicle detection supported.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose.

        Returns
        -------
        objects : list
            A list that contains all detected obstacle vehicles.

        """
        self.ego_pos = ego_pos
        objects = {
            'vehicles': [],
            'traffic_lights': []
        }

        objects = self.coperception_mode(objects)
        self.count += 1

        return objects

    def inference(self, data, objects = {'vehicles': [], 'traffic_lights': []}, with_submit=False, with_update=True):
        # inference
        reformat_data_dict = self.ml_manager.opencood_dataset.get_item_test(data, ClusteringPerceptionManager.ego_lidar_pose)
        output_dict = self.ml_manager.opencood_dataset.collate_batch_test(
            [reformat_data_dict])  # should have batch size dim
        if 'ego' in output_dict and 'processed_lidar' in output_dict['ego'] and output_dict['ego']['processed_lidar']['voxel_coords'].numel() == 0:
            logger.debug('Warning: coords is empty.')
            return objects
        # if len(output_dict['ego']['processed_lidar']['pillar_features'].shape) == 1:
        #     logger.debug('Warning: pillar_features is 1-dim tensor.')
        #     return objects, None            
        batch_data = self.ml_manager.to_device(output_dict)
        predict_box_tensor, predict_score, gt_box_tensor = self.ml_manager.inference(batch_data, with_submit)
        if with_update and predict_box_tensor is not None and predict_score is not None and gt_box_tensor is not None:
            logger.debug(f'predict_box_tensor: {predict_box_tensor.shape}')
            logger.debug(f'predict_score : {predict_score.shape}')
            logger.debug(f'gt_box_tensor : {gt_box_tensor.shape}')
            print(f'{self.id} did cp')
            ClusteringPerceptionManager.ego_predict_box_tensors.append(predict_box_tensor)
            ClusteringPerceptionManager.ego_predict_scores.append(predict_score)
            ClusteringPerceptionManager.ego_gt_box_tensors.append(gt_box_tensor)
            self.predict_box_tensor = predict_box_tensor
            self.gt_box_tensor = gt_box_tensor
            if self.is_ego:
                ClusteringPerceptionManager.ego_did_cp = True
        # self.ml_manager.show_vis(pred_box_tensor, gt_box_tensor, batch_data) show predict results frame by frame
        objects = o3d_predict_bbox_to_object(objects, predict_box_tensor, self.lidar.sensor)
        # retrieve speed from server
        self.speed_retrieve(objects)
        self.transform_retrieve(objects)
            
        objects = self.retrieve_traffic_lights(objects)
        return objects

    def coperception_mode(self, objects):
        """
        Use OpenCOOD to detect objects
        Note that we only apply detection for ego, and transform all data into ego's lidar_pose
        """
        # self.ego_predict_box_tensors = []
        # self.ego_predict_scores = []
        # self.ego_gt_box_tensors = []
        
        if self.lidar.data is None:
            return objects
        #self.cav_world.update_global_ego_id(self.vehicle.id)
        if ClusteringPerceptionManager.ego_vm is None:
            ClusteringPerceptionManager.ego_vm = self.cav_world.get_ego_vehicle_manager()
        ego_id = self.cav_world.ego_id
        self.is_ego = self.id == ego_id
        ego_in_cluster = False
        self.update_ego_lidar_pose()
        if self.v2x_manager.receive_beacon:
            self.do_cp += 1
        did_cp = False
        # record_results = is_ego
        data = OrderedDict()
        if self.enable_communicate and self.do_cp > 0:
            if self.v2x_manager.is_cluster_head():  #cluster head do cp

                if ego_id in self.v2x_manager.cluster_state['members']:
                    ego_in_cluster = True
                    logger.debug(f"ego is in cluster {self.id}")
                if not self.record_all_cavs and not self.is_ego and not ego_in_cluster and not self.apply_late_fusion:
                    logger.debug(f"ego is not in cluster {self.id}, skipped")
                    return objects
          
                data_size = 0.0
                ego_data = self.co_manager.prepare_data_fixed(
                    cav_id=self.id,
                    camera=self.rgb_camera,
                    lidar=self.lidar,
                    pos=self.ego_pos,
                    localizer=self.localization_manager,
                    agent=self.behavior_agent,
                    is_ego=True,
                )
                ego_data = self.co_manager.calculate_transformation(
                    cav_id=self.id,
                    cav_data=ego_data,
                    ego_pose=ClusteringPerceptionManager.ego_lidar_pose
                )
                # data_size += asizeof(ego_data)
                ego_data_size = asizeof(ego_data)
                self.ego_data = ego_data

                success_members_num = 0  
                vehicles_inside_cluster = self.co_manager.communicate_inside_cluster()
                logger.debug(f'{self.v2x_manager.vehicle_id} is collecting data from {vehicles_inside_cluster.keys()}:')

                for vid, nearby_data_dict in vehicles_inside_cluster.items():
                    if not nearby_data_dict:
                        continue
                    nearby_vm = nearby_data_dict['vehicle_manager']
                    nearby_v2x_manager = nearby_data_dict['v2x_manager']
                    nearby_data = self.co_manager.prepare_data_fixed(
                        cav_id=vid,
                        camera=nearby_v2x_manager.get_ego_rgb_image(),
                        lidar=nearby_v2x_manager.get_ego_lidar(),
                        pos=nearby_v2x_manager.get_ego_pos(),
                        localizer=nearby_vm.localizer,
                        agent=nearby_vm.agent,
                        is_ego=False,
                    )
                    nearby_data = self.co_manager.calculate_transformation(
                        cav_id=vid,
                        cav_data=nearby_data,
                        ego_pose=ClusteringPerceptionManager.ego_lidar_pose
                    )
                    nearby_data_size = asizeof(nearby_data)

                    success = True
                    # print('self.enable_network', self.enable_network)
                    start_time = 1
                    if self.enable_network:
                        source, target = nearby_v2x_manager, self.v2x_manager
                        # def schedule(self, source: 'V2XManager', target: 'V2XManager', volume: float) -> Tuple[int, int, int, bool]:
                        subchannel, start_time, end_time, success = self.v2x_manager.scheduler.schedule(source, target, nearby_data_size)
                        logger.info(f"network {source.vehicle_id} to {target.vehicle_id}: {subchannel}, {start_time}, {end_time}, {success}")

                    if success:
                        success_members_num += 1
                        data_size += nearby_data_size
                        self.cp_data.update(nearby_data)
                        self.co_manager.uploaded_member[vid] = start_time
                        

                # count communication_volume
                if CavWorld.network_manager:
                    # V2XManager.network_manager.update_communication_volume(data_size, communication_type="collect")
                    CavWorld.network_manager._update_communication_stats(data_size, "upload")
                logger.debug(f'collect data size: {data_size}, ego data size: {ego_data_size}')

                # merge ego data with neighbor data to apply coperception
                logger.debug(f"head {self.id}, {len(vehicles_inside_cluster)} members, {success_members_num} successed")
                if self.do_cp > 0 and len(vehicles_inside_cluster) - success_members_num <= 0:
                    data.update(self.cp_data)
                    data.update(self.ego_data)
                    self.cp_data.clear()
                    if CavWorld.network_manager:
                        cur_time = CavWorld.network_manager.current_time_slot
                        time_slot = CavWorld.network_manager.time_slot
                        for vid, start_time in self.co_manager.uploaded_member.items():
                            CavWorld.network_manager._record_packet_latency((cur_time - start_time) * time_slot)
                    self.co_manager.uploaded_member.clear()
                    self.do_cp -= 1
                
                    # objects = self.inference(data, objects, with_submit=is_ego, with_update=(self.apply_late_fusion or ego_in_cluster or is_ego))
                    objects = self.inference(data, objects, with_submit=(not self.apply_late_fusion and self.is_ego), with_update=(self.apply_late_fusion or ego_in_cluster or self.is_ego))
                    if self.is_ego and not self.apply_late_fusion:
                        did_cp = True
                        # ClusteringPerceptionManager.clear()

                    self.objects = objects
                    self.co_manager.broadcast_inside_cluster(self.id, objects)
                    logger.debug(f"{self.id} is cluster head, detect {len(objects['vehicles'])} vehicles and {len(objects['traffic_lights'])} traffic_lights")
                
            else:
                #For other vehicles, 1. get results from cluster head 2. communicate with vehicles outside the cluster
                #Note that only ego vehicle need the real results.
                if self.is_ego: 
                    logger.debug(f'coperception: {self.v2x_manager.vehicle_id}')
                    # output_dict_all = {}
                    ego_data = self.co_manager.prepare_data(
                    cav_id=self.id,
                    camera=self.rgb_camera,
                    lidar=self.lidar,
                    pos=self.ego_pos,
                    localizer=self.localization_manager,
                    agent=self.behavior_agent,
                    is_ego=self.is_ego,
                    )
                    ego_data = self.co_manager.calculate_transformation(
                        cav_id=self.id,
                        cav_data=ego_data,
                        ego_pose=ClusteringPerceptionManager.ego_lidar_pose
                    )

                    objects_self = self.inference(ego_data, objects, with_submit=(not self.apply_late_fusion), with_update=True)  #detect objects on its own
                    self.objects = objects_self
                    logger.debug(f"{self.id}: {len(objects_self['vehicles'])} vehicles and {len(objects_self['traffic_lights'])} traffic_lights detected from self")

                    buffer = (self.v2x_manager.read_buffer()) #get results from cluster head
                    objects_cluster, cluster_head_id = buffer['objects'], buffer['source']
  
                    logger.debug(f"{self.id}: {len(objects_cluster['vehicles'])} vehicles and {len(objects_cluster['traffic_lights'])} traffic_lights detected from cluster head {cluster_head_id}")

                    if CavWorld.network_manager:
                        objects_size = self.get_boxes_size()
                        # V2XManager.network_manager.update_communication_volume(objects_size, communication_type="outside")
                        CavWorld.network_manager._update_communication_stats(objects_size, "inter")

                    # pred_box_tensor, pred_score, gt_box_tensor = self.ml_manager.naive_late_fusion(
                    #                                                 ClusteringPerceptionManager.ego_predict_box_tensors, 
                    #                                                 ClusteringPerceptionManager.ego_predict_scores, 
                    #                                                 ClusteringPerceptionManager.ego_gt_box_tensors)
                    if self.is_ego and not self.apply_late_fusion:
                        did_cp = True
                    # # ClusteringPerceptionManager.clear()

                    # if pred_box_tensor is not None:
                    #     self.ml_manager.submit_results(pred_box_tensor, pred_score, gt_box_tensor, with_stats=True)

                    

        # plot the opencood inference results
        if self.lidar_visualize and self.is_ego and not self.apply_late_fusion:# and did_cp:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # o3d_visualizer_show(
            #     self.o3d_vis,
            #     self.count,
            #     self.lidar.o3d_pointcloud,
            #     objects)
            o3d_visualizer_show_coperception(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                self.predict_box_tensor,
                self.gt_box_tensor,
                # ClusteringPerceptionManager.ego_predict_box_tensors, ClusteringPerceptionManager.ego_gt_box_tensors,
                True, 
                objects,
                take_screenshot=True)           
        
        if did_cp:
            ClusteringPerceptionManager.clear()

        return objects

    def submit_cp_results(self):

        if not self.is_ego or not self.apply_late_fusion:
            return

        predict_box_tensor, pred_score, gt_box_tensor = self.ml_manager.naive_late_fusion(
                                                        ClusteringPerceptionManager.ego_predict_box_tensors, 
                                                        ClusteringPerceptionManager.ego_predict_scores, 
                                                        ClusteringPerceptionManager.ego_gt_box_tensors)
        
        if predict_box_tensor is not None and gt_box_tensor is not None and ClusteringPerceptionManager.ego_did_cp:
            for tensor in ClusteringPerceptionManager.ego_predict_box_tensors:
                print(tensor.shape)
            print(len(ClusteringPerceptionManager.ego_predict_box_tensors), (predict_box_tensor.shape), gt_box_tensor.shape)
            # if(predict_box_tensor.shape[0] >  gt_box_tensor.shape[0] / 2) or True:

            self.predict_box_tensor_fusion = predict_box_tensor
            self.gt_box_tensor_fusion = gt_box_tensor
            self.ml_manager.submit_results(predict_box_tensor, pred_score, gt_box_tensor, with_stats=True)
            ClusteringPerceptionManager.ego_did_cp = False

        if self.lidar_visualize:
            # while self.lidar.data is None:
            #     continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            o3d_visualizer_show_coperception(
            self.o3d_vis,
            self.count,
            self.lidar.o3d_pointcloud,
            self.predict_box_tensor_fusion,
            self.gt_box_tensor_fusion,
            # ClusteringPerceptionManager.ego_predict_box_tensors, ClusteringPerceptionManager.ego_gt_box_tensors,
            True, 
            {},
            take_screenshot=True)  

        ClusteringPerceptionManager.clear()