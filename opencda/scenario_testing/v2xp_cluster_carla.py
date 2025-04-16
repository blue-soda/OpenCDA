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

def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)
        application = ['single', 'cooperative', 'cluster']
        # create CAV world
        cav_world = CavWorld(apply_ml=opt.apply_ml, 
                             apply_cp=opt.apply_cp, 
                             coperception_params=scenario_params['coperception'],
                             network_params=scenario_params['network'],)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.apply_cp,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("v2xp_cluster_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=application, data_dump=False)
        # rsu_list = \
        #     scenario_manager.create_rsu_manager(data_dump=False)

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()
        
        traffic_cav_list = []
        if 'cluster' in application:
            traffic_cav_list = \
                scenario_manager.create_vehicle_manager_for_traffic(bg_veh_list)


        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='coop_town06',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()
        debug_helper = scenario_manager.world.debug 
        while True:
            scenario_manager.tick()
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=70),
                carla.Rotation(
                    pitch=-
                    90)))

            for single_cav in single_cav_list:
                single_cav.update_info()
                vehicle_location = single_cav.vehicle.get_transform().location
                color = single_cav.v2x_manager.rgb
                cluster_head = single_cav.v2x_manager.cluster_state['cluster_head']
                debug_helper.draw_string(vehicle_location + carla.Location(z=2.5),
                            f"EGO: {single_cav.vehicle.id}, {cluster_head}",
                            life_time=0.1,persistent_lines=True,draw_shadow=False,
                            color=carla.Color(*color))

            for traffic_cav in traffic_cav_list:
                traffic_cav.update_info()
                isOutOfSight = LocalizationManager.is_vehicle_out_of_sight( \
                    traffic_cav.vehicle.get_transform().location, transform.location)
                if traffic_cav.is_ok and isOutOfSight:
                    traffic_cav.is_ok = False
                    logger.debug(f"bg_vehicle {traffic_cav.vehicle.id} is out of range.")
                elif not traffic_cav.is_ok and not isOutOfSight:
                    traffic_cav.is_ok = True
                    logger.debug(f"bg_vehicle {traffic_cav.vehicle.id} is back.")
                vehicle_location = traffic_cav.vehicle.get_transform().location
                color = traffic_cav.v2x_manager.rgb
                cluster_head = traffic_cav.v2x_manager.cluster_state['cluster_head']
                debug_helper.draw_string(vehicle_location + carla.Location(z=2.5),
                            f"{traffic_cav.vehicle.id}, {int(not isOutOfSight)}, {cluster_head}",
                            life_time=0.1,persistent_lines=True,draw_shadow=False,
                            color=carla.Color(*color))
            
            for single_cav in single_cav_list:               
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)
            # for rsu in rsu_list:
            #     rsu.update_info()
            #     rsu.run_step()

    finally:
        eval_manager.evaluate()
        if 'cooperative' in application:
            cav_world.ml_manager.evaluate_final_average_precision()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()
        #