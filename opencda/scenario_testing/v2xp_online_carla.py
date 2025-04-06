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

def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)
        application = ['single', 'cooperative', 'cluster']
        # create CAV world
        cav_world = CavWorld(apply_ml=opt.apply_ml,
                             apply_coperception=True,
                             coperception_params=scenario_params['coperception'])

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.apply_cp,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("v2xp_online_carla.log", True)

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

            i = 0
            while i < len(traffic_cav_list):
                traffic_cav = traffic_cav_list[i]
                traffic_cav.update_info()
                if traffic_cav.is_ok and LocalizationManager.is_vehicle_out_of_sight(traffic_cav.vehicle.get_transform().location, transform.location):
                    traffic_cav.is_ok = False
                    traffic_cav_list.pop(i)
                    print(f"bg_vehicle {i} is out of range.")
                else:
                    i += 1
            
            for single_cav in single_cav_list:               
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)
            # for rsu in rsu_list:
            #     rsu.update_info()
            #     rsu.run_step()

    finally:
        eval_manager.evaluate()
        cav_world.ml_manager.evaluate_final_average_precision()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()
        #