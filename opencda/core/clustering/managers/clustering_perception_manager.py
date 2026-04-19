from collections import OrderedDict
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager
from opencda.core.common.v2x_manager \
    import V2XManager
from opencda.core.common.cav_world \
    import CavWorld
from opencda.core.clustering.managers.clustering_coperception_manager import ClusteringCoperceptionManager
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion, o3d_visualizer_show_coperception, o3d_predict_bbox_to_object, capture_spectator_view
from pympler.asizeof import asizeof
from opencda.core.sensing.perception.coperception_libs import CoperceptionLibs
from opencda.log.logger_config import logger

class ClusteringPerceptionManager(PerceptionManager):
    #static ego_data_dict
    ego_lidar_pose = None
    ego_vm = None
    ego_predict_box_tensors = {}
    ego_predict_scores = {}
    ego_gt_box_tensors = {}
    ego_did_cp = False
    ego_gt_box_ids = None
    ego_last_cp_used_remote_uploads = False
    ego_last_cp_remote_ids = []
    ego_last_cp_slot = None

    def __init__(self, v2x_manager, localization_manager, behavior_agent, vehicle,
                 config_yaml, cav_world, data_dump=False, carla_world=None, infra_id=None, enable_network=False):
        super().__init__(v2x_manager, localization_manager, behavior_agent, vehicle,
                 config_yaml, cav_world, data_dump, carla_world, infra_id, enable_network)
        self.communication_volume = 0.0
        self.co_manager = ClusteringCoperceptionManager(self.vid, v2x_manager, self.coperception_libs, enable_network, network_manager=CavWorld.network_manager)
        if ClusteringPerceptionManager.ego_vm is None:
            ClusteringPerceptionManager.ego_vm = cav_world.get_ego_vehicle_manager()
        self.doing_cp = False
        self.cp_data = {}
        self.is_ego = False
        self.predict_box_tensor = None
        self.gt_box_tensor = None
        self.predict_box_tensor_fusion = None
        self.gt_box_tensor_fusion = None
        self.final_drain_mode = False
        
        self.apply_late_fusion = True
        self.record_all_cavs = False       
        self.do_not_skip_any_cav = False

        self.screenshot_scenario_and_lidar_count = 10
        self.set_enable_grids(True) 
        # In networked co-simulation, evaluating every perception tick mixes many
        # frames that have no newly received cooperative data into the final AP.
        # Wait for a completed upload round instead.
        self.do_cp_every_tick = not enable_network

        self.vid_to_screen_shot = {}#{6, 11, 17}

    #TODO: if self.record_all_cavs: do late fusion for all cavs in the cluster, not only ego.
    @staticmethod
    def update_ego_lidar_pose():
        ego_vm = ClusteringPerceptionManager.ego_vm
        ego_v2x_manager = ego_vm.v2x_manager
        lidar=ego_v2x_manager.get_ego_lidar()
        ClusteringPerceptionManager.ego_lidar_pose = CoperceptionLibs.load_cur_lidar_pose(lidar)['lidar_pose']

    @staticmethod
    def clear():
        ClusteringPerceptionManager.ego_predict_box_tensors = {}
        ClusteringPerceptionManager.ego_predict_scores = {}
        ClusteringPerceptionManager.ego_gt_box_tensors = {}
        ClusteringPerceptionManager.ego_gt_box_ids = None
        ClusteringPerceptionManager.ego_last_cp_used_remote_uploads = False
        ClusteringPerceptionManager.ego_last_cp_remote_ids = []
        ClusteringPerceptionManager.ego_last_cp_slot = None
    
    @staticmethod
    def get_boxes_size():
        return asizeof(ClusteringPerceptionManager.ego_predict_box_tensors) + \
            asizeof(ClusteringPerceptionManager.ego_predict_scores) + \
            asizeof(ClusteringPerceptionManager.ego_gt_box_tensors) + \
            asizeof(ClusteringPerceptionManager.ego_gt_box_ids)


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

        tick = self.perception_tick()
        if tick:
            self.doing_cp = True

        self.ego_pos = ego_pos
        objects = {
            'vehicles': [],
            'traffic_lights': [],
            'is_skipped': False
        }

        objects = self.coperception_mode(objects)
        self.count += 1

        # plot the opencood inference results
        screenshot_lidar_with_transparent_bg = self.count == self.screenshot_scenario_and_lidar_count
        if tick and ((self.lidar_visualize and self.is_ego and not self.apply_late_fusion) or (self.vid in self.vid_to_screen_shot and screenshot_lidar_with_transparent_bg)):
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            if not self.o3d_vis:
                self.o3d_vis = o3d_visualizer_init(self.vid)
            o3d_visualizer_show_coperception(
                vis=self.o3d_vis,
                count=self.count,
                point_cloud=self.lidar.o3d_pointcloud,
                predict_bbx_tensor=self.predict_box_tensor,
                gt_box_tensor=self.gt_box_tensor,
                show_predict=self.is_ego, 
                show_gt=self.is_ego and not screenshot_lidar_with_transparent_bg, 
                objects=objects,
                take_screenshot=True,
                transparent_bg=screenshot_lidar_with_transparent_bg,
                vid=self.vid)    
            if screenshot_lidar_with_transparent_bg:
                capture_spectator_view(self.carla_world)
            
        return objects

    def inference(self, data, objects = {'vehicles': [], 'traffic_lights': []}, with_submit=False, with_update=True):
        # inference
        reformat_data_dict = self.ml_manager.opencood_dataset.get_item_test(data, ClusteringPerceptionManager.ego_lidar_pose)
        output_dict = self.ml_manager.opencood_dataset.collate_batch_test(
            [reformat_data_dict])  # should have batch size dim
        
        if 'ego' in output_dict and 'processed_lidar' in output_dict['ego'] and output_dict['ego']['processed_lidar']['voxel_coords'].numel() == 0:
            logger.debug('Warning: coords is empty.')
            return objects      
        
        batch_data = self.ml_manager.to_device(output_dict)
        ret = self.ml_manager.inference(batch_data, with_submit, return_object_ids=self.is_ego) #ego获取gt_box的id
        predict_box_tensor, predict_score, gt_box_tensor = ret[0:3]

        if with_update and predict_box_tensor is not None and predict_score is not None and gt_box_tensor is not None:
            logger.debug(f'predict_box_tensor: {predict_box_tensor.shape}')
            logger.debug(f'predict_score : {predict_score.shape}')
            logger.debug(f'gt_box_tensor : {gt_box_tensor.shape}')
            logger.debug(f'{self.vid} did cp')

            ClusteringPerceptionManager.ego_predict_box_tensors[self.vid] = predict_box_tensor
            ClusteringPerceptionManager.ego_predict_scores[self.vid] = predict_score
            if self.is_ego: #真值框只取ego附近的, 无需nms
                ClusteringPerceptionManager.ego_gt_box_tensors[self.vid] = gt_box_tensor
                ClusteringPerceptionManager.ego_gt_box_ids = ret[3]
                ClusteringPerceptionManager.ego_did_cp = True
                logger.info(
                    f"CP_EGO_DID_CP vid={self.vid} ego_id={self.cav_world.ego_id} "
                    f"head_id={self.v2x_manager.cluster_state.get('head_id', None)} "
                    f"members={sorted(list(self.v2x_manager.cluster_state.get('member_ids', [])))}"
                )

            self.predict_box_tensor = predict_box_tensor
            self.gt_box_tensor = gt_box_tensor

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
        
        if self.lidar.data is None:
            return objects

        if ClusteringPerceptionManager.ego_vm is None:
            ClusteringPerceptionManager.ego_vm = self.cav_world.get_ego_vehicle_manager()
        self.update_ego_lidar_pose()

        ego_id = self.cav_world.ego_id
        self.is_ego = self.vid == self.cav_world.ego_id
        ego_in_cluster = False
        did_cp = False

        # receive cluster members data
        if self.enable_communicate and self.v2x_manager.is_cluster_head():
            if self.final_drain_mode:
                self.receive_cluster_members_data()
            else:
                self.collect_cluster_members_data(is_ego=self.is_ego)

        if not self.doing_cp:
            # print("Perception skipped this tick.")
            pass

        else:
            if self.enable_communicate and self.v2x_manager.is_cluster_head():  # cluster head do cp
                if ego_id in self.v2x_manager.cluster_state['member_ids']:
                    ego_in_cluster = True
                    logger.debug(f"ego is in cluster {self.vid}")
                if not self.do_not_skip_any_cav and not self.record_all_cavs and not self.is_ego and not ego_in_cluster and not self.apply_late_fusion:
                    logger.debug(f"ego is not in cluster {self.vid}, skipped")
                    return objects

                # do cp when perception_tick is called, no matter whether all data is uploaded
                if self.do_cp_every_tick or self.co_manager.all_data_uploaded():
                    # reset tick
                    self.doing_cp = False
                    # collect ego data
                    self_data = self.collect_self_data(is_ego=self.is_ego)
                    self_data_size = self_data[self.vid]['lidar_np'].nbytes
                    logger.debug(f"head {self.vid}, collect ego data size: {self_data_size}")

                    data = OrderedDict()
                    remote_cp_data = OrderedDict(self.cp_data)
                    data.update(remote_cp_data)
                    data.update(self_data)
                    del self_data
                    self.cp_data.clear()

                    if self.enable_network:
                        cur_time = CavWorld.network_manager.current_time_slot
                        time_slot = CavWorld.network_manager.time_slot
                        for vid, start_time in self.co_manager.uploading_cavs.items():
                            CavWorld.network_manager._record_cp_latency((cur_time - start_time) * time_slot * 1000)  # ms
                    else:
                        cur_time = None

                    if self.is_ego:
                        ClusteringPerceptionManager.ego_last_cp_used_remote_uploads = len(remote_cp_data) > 0
                        ClusteringPerceptionManager.ego_last_cp_remote_ids = list(remote_cp_data.keys())
                        ClusteringPerceptionManager.ego_last_cp_slot = cur_time
                        logger.info(
                            f"CP_EVAL_FRAME ego={self.vid} head_id={self.v2x_manager.cluster_state.get('head_id', None)} "
                            f"slot={cur_time} "
                            f"remote_ids={ClusteringPerceptionManager.ego_last_cp_remote_ids} "
                            f"use_remote={ClusteringPerceptionManager.ego_last_cp_used_remote_uploads}"
                        )

                    objects = self.inference(data, objects, with_submit=(not self.apply_late_fusion and self.is_ego), with_update=(self.apply_late_fusion or ego_in_cluster or self.is_ego))
                    if self.is_ego and not self.apply_late_fusion:
                        did_cp = True

                    self.objects = objects
                    self.co_manager.broadcast_objects_info(objects)
                    logger.debug(f"{self.vid} is cluster head, detect {len(objects['vehicles'])} vehicles and {len(objects['traffic_lights'])} traffic_lights")

                    # collect cluster members data for the next cp
                    self.co_manager.clear_uploaded_and_uploading()
                    if not self.final_drain_mode:
                        self.collect_cluster_members_data(is_ego=self.is_ego)
                else:
                    logger.info(
                        f"CP_WAIT_FRAME ego={self.vid} slot={CavWorld.network_manager.current_time_slot if self.enable_network else None} "
                        f"uploaded={len(self.co_manager.uploaded_cavs)}/{self.co_manager.cavs_num}"
                    )
                    return self.objects if self.objects is not None else objects
                    
            else:
                #For other vehicles, 1. get results from cluster head 2. communicate with vehicles outside the cluster
                #Note that only ego vehicle need the real results.
                if self.is_ego or self.do_not_skip_any_cav: 
                    # reset tick
                    self.doing_cp = False
                    logger.debug(f'coperception: {self.v2x_manager.vehicle_id}')
                    # output_dict_all = {}
                    ego_data = self.collect_self_data(is_ego=self.is_ego)

                    objects_self = self.inference(ego_data, objects, with_submit=(not self.apply_late_fusion and self.is_ego), with_update=(self.apply_late_fusion or self.is_ego))  #detect objects on its own
                    self.objects = objects_self
                    logger.debug(f"{self.vid}: {len(objects_self['vehicles'])} vehicles and {len(objects_self['traffic_lights'])} traffic_lights detected from self")

                    buffer = (self.v2x_manager.read_buffer()) #get results from cluster head
                    objects_cluster, cluster_head_id = buffer['objects'], buffer['source_id']  
                    logger.debug(f"{self.vid}: {len(objects_cluster['vehicles'])} vehicles and {len(objects_cluster['traffic_lights'])} traffic_lights detected from cluster head {cluster_head_id}")

                    if self.enable_network:
                        objects_size = self.get_boxes_size()
                        # V2XManager.network_manager.update_communication_volume(objects_size, communication_type="outside")
                        CavWorld.network_manager._update_communication_stats(objects_size, "inter")

                    if self.is_ego and not self.apply_late_fusion:
                        did_cp = True
                        logger.debug("ego did cp (with its own data)")
        
        if did_cp:
            ClusteringPerceptionManager.clear()

        return objects

    def submit_cp_results(self):
        # submit cp results for ego vehicle after late fusion, called after all vehicles run_step
        logger.info(
            f"CP_SUBMIT_GATE vid={self.vid} is_ego={self.is_ego} "
            f"apply_late_fusion={self.apply_late_fusion} "
            f"ego_did_cp={ClusteringPerceptionManager.ego_did_cp} "
            f"ego_id={self.cav_world.ego_id} "
            f"head_id={self.v2x_manager.cluster_state.get('head_id', None)} "
            f"members={sorted(list(self.v2x_manager.cluster_state.get('member_ids', [])))}"
        )
        if not self.is_ego or not self.apply_late_fusion or not ClusteringPerceptionManager.ego_did_cp:
            return

        logger.debug("submit cp results")
        ego_predict_box_tensors_list = [ClusteringPerceptionManager.ego_predict_box_tensors[vid] for vid in ClusteringPerceptionManager.ego_predict_box_tensors.keys()]
        ego_predict_scores_list = [ClusteringPerceptionManager.ego_predict_scores[vid] for vid in ClusteringPerceptionManager.ego_predict_scores.keys()]
        ego_gt_box_tensors_list = [ClusteringPerceptionManager.ego_gt_box_tensors[vid] for vid in ClusteringPerceptionManager.ego_gt_box_tensors.keys()]

        ClusteringPerceptionManager.ego_did_cp = False

        target_ids = []
        for cluster in self.v2x_manager.all_clusters:
            target_ids.extend(cluster.members)

        logger.debug(f"target_ids: {target_ids}")
        logger.debug(f"ego_gt_box_ids: {ClusteringPerceptionManager.ego_gt_box_ids}")
        for idx, box_id in enumerate(ClusteringPerceptionManager.ego_gt_box_ids):
            if box_id in target_ids:
                gt_box = ClusteringPerceptionManager.ego_gt_box_tensors[self.vid][idx].unsqueeze(0)
                ego_predict_box_tensors_list.append(gt_box)
                ego_predict_scores_list.append(1.0)

        predict_box_tensor, pred_score = self.ml_manager.naive_late_fusion(
                                                        ego_predict_box_tensors_list, 
                                                        ego_predict_scores_list)
        gt_box_tensor, _ = self.ml_manager.naive_late_fusion(
                                                        ego_gt_box_tensors_list, 
                                                        None)

        if predict_box_tensor is not None and gt_box_tensor is not None:
            logger.debug("late fusion input: ")
            for tensor in ego_predict_box_tensors_list:
                logger.debug(f"tensor shape: {tensor.shape}")
            logger.debug(f"late fusion output - predict_box_tensor: {predict_box_tensor.shape}, gt_box_tensor: {gt_box_tensor.shape}")
            logger.info(
                f"CP_SUBMIT_FRAME ego={self.vid} slot={ClusteringPerceptionManager.ego_last_cp_slot} "
                f"remote_ids={ClusteringPerceptionManager.ego_last_cp_remote_ids} "
                f"with_stats={ClusteringPerceptionManager.ego_last_cp_used_remote_uploads}"
            )

            self.predict_box_tensor_fusion = predict_box_tensor
            self.gt_box_tensor_fusion = gt_box_tensor

            self.ml_manager.submit_results(
                predict_box_tensor,
                pred_score,
                gt_box_tensor,
                with_stats=ClusteringPerceptionManager.ego_last_cp_used_remote_uploads,
                force=True,
            )
            
        ClusteringPerceptionManager.clear() # 不及时清理会导致精度下降, 甚至爆显存

        if self.lidar_visualize:
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            transparent_bg = self.count == self.screenshot_scenario_and_lidar_count
            o3d_visualizer_show_coperception(
                vis=self.o3d_vis,
                count=self.count,
                point_cloud=self.lidar.o3d_pointcloud,
                predict_bbx_tensor=self.predict_box_tensor,
                gt_box_tensor=self.gt_box_tensor,
                show_predict=True, 
                show_gt=not transparent_bg, 
                objects={},
                take_screenshot=True,
                transparent_bg=transparent_bg)    
            if transparent_bg:
                capture_spectator_view(self.carla_world)

    def collect_self_data(self, is_ego=True):
        return self.co_manager.prepare_and_transform_data_from_managers(
            v2x_manager=self.v2x_manager,
            localizer=self.localization_manager,
            agent=self.behavior_agent,
            ego_lidar_pose=ClusteringPerceptionManager.ego_lidar_pose,
            use_ego_vehicles=True,
            is_ego=is_ego
        )
    
    def collect_cluster_members_data(self, is_ego=False):
        members_data = self.co_manager.communicate(
            ego_lidar_pose=ClusteringPerceptionManager.ego_lidar_pose,
            use_ego_vehicles=True,
            is_ego=is_ego
        )
        if members_data:
            self.cp_data.update(members_data)
    
    def receive_cluster_members_data(self):
        members_data = self.co_manager.communicate_via_network(try_to_send=False)
        if members_data:
            self.cp_data.update(members_data)

    def enable_final_drain(self, enable=True):
        self.final_drain_mode = enable

    def has_pending_final_drain(self):
        if not self.final_drain_mode or not self.enable_network:
            return False
        if not self.enable_communicate or not self.v2x_manager.is_cluster_head():
            return False
        if self.cp_data:
            return True
        if not self.co_manager.uploading_data:
            return False
        if self.co_manager.cavs_num <= 0:
            return False
        return len(self.co_manager.uploaded_cavs) < self.co_manager.cavs_num
