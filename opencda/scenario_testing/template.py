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

#cluster, rsu, platoon, data_dump
applications = []


file_name = "-"
cav_world, scenario_manager, eval_manager = None, None, None
single_cav_list, traffic_cav_list, rsu_list, platoon_list = [], [], [], []
town_name, xodr_path_name, sumo_cfg_name = None, None, None

def run_scenario(opt, scenario_params, application=[], filename="-", town=None, xdor_path=None, sumo_cfg=None):
    global applications, town_name, xodr_path_name, file_name, sumo_cfg_name
    applications = application
    town_name = town
    xodr_path_name = xdor_path
    file_name = filename
    sumo_cfg_name = sumo_cfg
    init(opt, scenario_params)
    try:
        run()
    finally:
        stop(opt)


def init(opt, scenario_params):
    global cav_world, scenario_manager, eval_manager, applications, single_cav_list, traffic_cav_list, rsu_list, town_name, xodr_path_name, sumo_cfg_name
    scenario_params = add_current_time(scenario_params)

    # add params
    coperception_params, network_params = None, None
    if opt.apply_cp: #and 'coperception' in scenario_params:
        applications.append('coperception')
        coperception_params = scenario_params['coperception']
    if opt.network and 'network' in scenario_params:
        applications.append('network')
        network_params = scenario_params['network']
    data_dump = 'data_dump' in applications

    # create CAV world
    cav_world = CavWorld(apply_ml=opt.apply_ml, 
                            apply_cp=opt.apply_cp, 
                            coperception_params=coperception_params,
                            network_params=network_params,)

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

    # create background traffic in carla
    traffic_manager, bg_veh_list = \
        scenario_manager.create_traffic_carla()
    
    if 'rsu' in applications:
        rsu_list = \
            scenario_manager.create_rsu_manager(data_dump=data_dump)

    if 'cluster' in applications:
        traffic_cav_list = \
            scenario_manager.create_vehicle_manager_for_traffic(bg_veh_list, application=applications+['traffic'])

    if 'platoon' in applications:
        # create platoon members
        platoon_list = \
            scenario_manager.create_platoon_manager(
                data_dump=data_dump)
        
    # create evaluation manager
    eval_manager = \
        EvaluationManager(scenario_manager.cav_world,
                            script_name=file_name,
                            current_time=scenario_params['current_time'])
        

def run(debug=True):
    global scenario_manager, applications, single_cav_list, traffic_cav_list, rsu_list

    spectator = scenario_manager.world.get_spectator()
    if 'platoon' in applications:
        spectator_vehicle = platoon_list[0].vehicle_manager_list[1].vehicle
    else:
        spectator_vehicle = single_cav_list[0].vehicle

    if debug:
        debug_helper = scenario_manager.world.debug 

    while True:
        scenario_manager.tick()
        transform = spectator_vehicle.get_transform()
        spectator.set_transform(carla.Transform(
            transform.location +
            carla.Location(
                z=70),
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

            single_cav.update_info()
            if debug:
                draw_string(debug_helper, single_cav)

        for traffic_cav in traffic_cav_list:
            traffic_cav.update_info()
            check_is_out_sight(transform, traffic_cav)

            if debug:
                draw_string(debug_helper, traffic_cav)
        
        for single_cav in single_cav_list:               
            control = single_cav.run_step()
            single_cav.vehicle.apply_control(control)

        for rsu in rsu_list:
            rsu.update_info()
            rsu.run_step()

        if 'network' in applications:
            cav_world.network_manager.advance_time_slot()


def stop(opt):
    global cav_world, scenario_manager, eval_manager
    try:
        eval_manager.evaluate()
        if 'coperception' in applications:
            cav_world.ml_manager.evaluate_final_average_precision()

        if opt.record:
            scenario_manager.client.stop_recorder()
    
    finally:
        scenario_manager.close()


def draw_string(debug_helper, cav):
    vehicle_location = cav.vehicle.get_transform().location
    color = cav.v2x_manager.rgb

    if 'cluster' in applications:
        cluster_head = str(cav.v2x_manager.cluster_state['cluster_head'])
    else:
        cluster_head = ""

    debug_helper.draw_string(vehicle_location + carla.Location(z=2.5),
        f"{cav.vehicle.id}, {cluster_head}",
        life_time=0.1, persistent_lines=True, draw_shadow=False,
        color=carla.Color(*color))
    

def check_is_out_sight(transform, cav):
    is_out_of_sight = LocalizationManager.is_vehicle_out_of_sight( \
        cav.vehicle.get_transform().location, transform.location)
    
    if cav.is_ok and is_out_of_sight:
        cav.is_ok = False
        logger.debug(f"bg_vehicle {cav.vehicle.id} is out of range.")

    elif not cav.is_ok and not is_out_of_sight:
        cav.is_ok = True
        logger.debug(f"bg_vehicle {cav.vehicle.id} is back.")