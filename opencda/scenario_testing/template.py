# -*- coding: utf-8 -*-
"""
Scenario testing: merging vehicle joining a platoon in the
customized 2-lane freeway simplified map sorely with carla
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time, save_yaml
from opencda.core.sensing.localization.localization_manager import LocalizationManager
from opencda.log.logger_config import logger
import os
from opencda.core.common.data_dumper import DataDumper

#cluster, rsu, platoon, data_dump, traffic
applications = []


file_name = "-"
cav_world, scenario_manager, eval_manager = None, None, None
single_cav_list, traffic_cav_list, rsu_list, platoon_list, uav_list = [], [], [], [], []
town_name, xodr_path_name, sumo_cfg_name = None, None, None

def run_scenario(opt, scenario_params, application=[], filename="-", town=None, xdor_path=None, sumo_cfg=None):
    global applications, town_name, xodr_path_name, file_name, sumo_cfg_name
    applications = application
    town_name = town
    xodr_path_name = xdor_path
    file_name = filename
    sumo_cfg_name = sumo_cfg

    try:
        init(opt, scenario_params)
        run(debug=opt.debug)
    finally:
        stop(opt)


def init(opt, scenario_params):
    global cav_world, scenario_manager, eval_manager, applications, single_cav_list, traffic_cav_list, rsu_list, uav_list, town_name, xodr_path_name, sumo_cfg_name
    scenario_params = add_current_time(scenario_params)
    # add params
    coperception_params, network_params = None, None
    if opt.apply_cp: #and 'coperception' in scenario_params:
        applications.append('coperception')
        coperception_params = scenario_params['coperception']
    if opt.network and 'network' in scenario_params:
        applications.append('network')
        network_params = scenario_params['network']
        # The networked CP scenario relies on clustering-aware V2X/perception
        # managers; without this flag VehicleManager falls back to the plain
        # PerceptionManager path and CP results are never submitted.
        if network_params.get('scheduler') == 'cluster' and 'cluster' not in applications:
            applications.append('cluster')
    if hasattr(opt, 'uav') and opt.uav and 'uav_list' in scenario_params:
        applications.append('uav')
    data_dump = 'data_dump' in applications
    if data_dump:
        dump_root = DataDumper.get_save_root(scenario_params['current_time'])
        if not os.path.exists(dump_root):
            os.makedirs(dump_root)
        save_yaml(scenario_params, os.path.join(dump_root, 'data_protocol.yaml'))

    # create CAV world
    cav_world = CavWorld(apply_ml=opt.apply_ml, 
                            apply_cp=opt.apply_cp, 
                            coperception_params=coperception_params,
                            network_params=network_params,
                            world_params=scenario_params['world'])

    # create scenario manager
    if sumo_cfg_name:
        # create co-simulation scenario manager
        scenario_manager = \
            sim_api.CoScenarioManager(scenario_params,
                                      opt.apply_ml,
                                      opt.apply_cp,
                                      opt.version,
                                      town=town_name,
                                      xodr_path=xodr_path_name,
                                      cav_world=cav_world,
                                      sumo_file_parent_path=sumo_cfg_name)
    else:
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                    opt.apply_ml,
                                                    opt.apply_cp,
                                                    opt.version,
                                                    town=town_name,
                                                    xodr_path=xodr_path_name,
                                                    cav_world=cav_world)

    if opt.record:
        scenario_manager.client. \
            start_recorder(f"{file_name}.log", True)

    single_cav_list = \
        scenario_manager.create_vehicle_manager(application=applications+['single'], data_dump=data_dump)

    # Tick world to initialize sensors (need enough ticks for lidar rotation)
    for _ in range(10):
        scenario_manager.tick()

    # create background traffic in carla
    traffic_manager, bg_veh_list, traffic_cav_list = \
        scenario_manager.create_traffic_carla(application=applications+['traffic'])
    
    if 'rsu' in applications:
        rsu_list = \
            scenario_manager.create_rsu_manager(data_dump=data_dump)

    if 'platoon' in applications:
        # create platoon members
        platoon_list = \
            scenario_manager.create_platoon_manager(
                data_dump=data_dump)

    # create UAV if enabled
    if 'uav' in applications:
        from opencda.core.common.uav_manager import UAVManager
        uav_base_config = scenario_params.get('uav_base', {})
        uav_list_config = scenario_params.get('uav_list', [])

        for uav_config in uav_list_config:
            mode = uav_config.get('mode', 'tracking')
            spawn_pos = uav_config.get('spawn_position', [0, 0, 0.3, 0, 0, 0])
            spawn_loc = carla.Location(x=spawn_pos[0], y=spawn_pos[1], z=spawn_pos[2])

            target_vehicle = None
            destination = None

            if mode == 'tracking':
                target_id = uav_config.get('target', 0)
                # target_id starts from 1, single_cav_list is 0-indexed
                if 0 < target_id <= len(single_cav_list):
                    target_vehicle = single_cav_list[target_id - 1].vehicle
            elif mode == 'navigation':
                dest = uav_config.get('destination', [0, 0, 60])
                destination = carla.Location(x=dest[0], y=dest[1], z=dest[2])

            uav_manager = UAVManager(uav_config, uav_base_config, scenario_manager.world,
                                     cav_world, target_vehicle, destination, application=applications)
            uav_vid = 900 + len(uav_list)
            uav_manager.spawn_drone(spawn_loc, vid=uav_vid)
            uav_manager.takeoff()
            uav_list.append(uav_manager)

    if 'network' in applications and getattr(cav_world, 'network_manager', None):
        cav_world.network_manager.mark_vehicle_registration_complete()

    # create evaluation manager
    eval_manager = \
        EvaluationManager(scenario_manager.cav_world,
                            script_name=file_name,
                            current_time=scenario_params['current_time'])
        

def run(debug=True):
    global scenario_manager
    while True:
        _tick_once(debug=debug)


def _tick_once(debug=True):
    global scenario_manager, applications, single_cav_list, traffic_cav_list, platoon_list, rsu_list, uav_list

    all_cavs = single_cav_list + traffic_cav_list
    spectator = scenario_manager.world.get_spectator()
    if 'platoon' in applications:
        spectator_vehicle = platoon_list[0].vehicle_manager_list[1].vehicle
    else:
        spectator_vehicle = single_cav_list[0].vehicle

    if debug:
        debug_helper = scenario_manager.world.debug

    scenario_manager.tick()
    transform = spectator_vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location +
        carla.Location(
            z=180),
        carla.Rotation(
            pitch=-
            90)))

    for platoon in platoon_list:
        platoon.update_information()
        platoon.run_step()

    for i, single_cav in enumerate(single_cav_list):
        if single_cav.v2x_manager.in_platoon():
            single_cav_list.pop(i)
            continue

        single_cav.update_data()
        if debug:
            draw_string(debug_helper, single_cav)

    for traffic_cav in traffic_cav_list:
        traffic_cav.update_data()
        check_is_out_sight(transform, traffic_cav)
        if debug:
            draw_string(debug_helper, traffic_cav)

    for cav in all_cavs:
        cav.update_info(update_data=False)

    if 'coperception' in applications:
        for cav in all_cavs:
            if hasattr(cav.perception_manager, 'submit_cp_results'):
                cav.submit_cp_results()

    for cav in all_cavs:
        control = cav.run_step()
        if control:
            cav.vehicle.apply_control(control)

    for rsu in rsu_list:
        rsu.update_info()
        rsu.run_step()

    for uav in uav_list:
        uav.update_info()
        uav.run_step()

    if 'network' in applications:
        cav_world.network_manager.advance_time_slot()


def _tick_final_drain(debug=True):
    global scenario_manager, applications, single_cav_list, traffic_cav_list

    all_cavs = single_cav_list + traffic_cav_list
    if not all_cavs:
        return

    spectator = scenario_manager.world.get_spectator()
    spectator_vehicle = single_cav_list[0].vehicle if single_cav_list else all_cavs[0].vehicle

    if debug:
        debug_helper = scenario_manager.world.debug

    scenario_manager.tick()
    transform = spectator_vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=180),
        carla.Rotation(pitch=-90)))

    for i, single_cav in enumerate(single_cav_list):
        if single_cav.v2x_manager.in_platoon():
            single_cav_list.pop(i)
            continue

        single_cav.update_data()
        if debug:
            draw_string(debug_helper, single_cav)

    for traffic_cav in traffic_cav_list:
        traffic_cav.update_data()
        check_is_out_sight(transform, traffic_cav)
        if debug:
            draw_string(debug_helper, traffic_cav)

    for cav in all_cavs:
        cav.update_info(update_data=False)

    if 'coperception' in applications:
        for cav in all_cavs:
            if hasattr(cav.perception_manager, 'submit_cp_results'):
                cav.submit_cp_results()

    if 'network' in applications:
        cav_world.network_manager.advance_time_slot()


def _run_final_drain(debug=False):
    global cav_world, applications, single_cav_list, traffic_cav_list

    if 'network' not in applications or 'coperception' not in applications or cav_world is None:
        return
    if CavWorld.network_manager is None:
        return

    final_drain_slots = int(CavWorld.network_manager.config.get('final_drain_slots', 0))
    if final_drain_slots <= 0:
        return

    all_cavs = single_cav_list + traffic_cav_list
    drainable = [
        cav for cav in all_cavs
        if hasattr(cav.perception_manager, 'enable_final_drain')
    ]
    if not drainable:
        return

    previous_freeze = getattr(cav_world, 'freeze_cluster_updates', False)
    cav_world.freeze_cluster_updates = True

    for cav in drainable:
        cav.perception_manager.enable_final_drain(True)

    try:
        for drain_slot in range(1, final_drain_slots + 1):
            pending = [
                cav.vid for cav in drainable
                if cav.perception_manager.has_pending_final_drain()
            ]
            if not pending:
                logger.info(f"FINAL_DRAIN done before slot {drain_slot}, no pending uploads.")
                break
            logger.info(
                f"FINAL_DRAIN slot={drain_slot}/{final_drain_slots} "
                f"pending_heads={pending} time_slot={CavWorld.network_manager.current_time_slot}"
            )
            _tick_final_drain(debug=debug)
    finally:
        cav_world.freeze_cluster_updates = previous_freeze
        for cav in drainable:
            cav.perception_manager.enable_final_drain(False)

def stop(opt):
    global cav_world, scenario_manager, eval_manager, uav_list
    try:
        _run_final_drain(debug=getattr(opt, 'debug', False))
        if eval_manager:
            eval_manager.evaluate()
        if 'coperception' in applications and cav_world:
            cav_world.ml_manager.evaluate_final_average_precision()

        if opt.record and scenario_manager:
            scenario_manager.client.stop_recorder()

        for uav in uav_list:
            uav.destroy()
        for rsu in rsu_list:
            rsu.destroy()

    finally:
        if scenario_manager:
            scenario_manager.close()


def draw_string(debug_helper, cav):
    global cav_world
    vehicle_location = cav.vehicle.get_transform().location
    color = cav.v2x_manager.rgb

    if 'coperception' in applications and hasattr(cav.v2x_manager, 'cluster_state'):
        cluster_head = str(cav.v2x_manager.cluster_state['head_id'])
    else:
        cluster_head = ""

    debug_helper.draw_string(vehicle_location + carla.Location(z=2.5),
        # f"{cav.vehicle.id}, {cluster_head}",
        f"{cav_world.get_vid(cav.vehicle.id)}, {cluster_head}",
        life_time=0.1, persistent_lines=True, draw_shadow=False,
        color=carla.Color(*color))
    

def check_is_out_sight(transform, cav):
    global cav_world
    is_out_of_sight = LocalizationManager.is_vehicle_out_of_sight( \
        cav.vehicle.get_transform().location, transform.location)
    
    if cav.is_ok and is_out_of_sight:
        cav.is_ok = False
        logger.debug(f"bg_vehicle {cav_world.get_vid(cav.vehicle.id)} is out of range.")

    elif not cav.is_ok and not is_out_of_sight:
        cav.is_ok = True
        logger.debug(f"bg_vehicle {cav_world.get_vid(cav.vehicle.id)} is back.")
