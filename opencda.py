# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import importlib
import os
import sys

# Remove problematic paths that shadow the properly installed opencood
# The broken opencood at airv2x-perception causes import errors
_paths_to_remove = [
    r'c:\workspace\airv2x-perception',
    r'c:\workspace\pycarlanet',
]
for _p in _paths_to_remove:
    if _p in sys.path:
        sys.path.remove(_p)

# Ensure the properly installed opencood/opencood is found
# The pip-installed opencood has opencood modules at c:\workspace\opencda\opencda\opencood
# But C:\Workspace\OpenCDA\opencda (same path, case differences) is a broken namespace
# Add the proper nested path first so it takes precedence
_proper_opencood_path = r'C:\Workspace\OpenCDA\opencda\opencda'
if os.path.exists(_proper_opencood_path):
    # Remove the broken CWD opencood path and add the proper nested path
    _cwd_opencood = r'C:\Workspace\OpenCDA\opencda'
    if _cwd_opencood in sys.path:
        sys.path.remove(_cwd_opencood)
    # Insert at front to take precedence
    if _proper_opencood_path not in sys.path:
        sys.path.insert(0, _proper_opencood_path)

from omegaconf import OmegaConf

from opencda.version import __version__
from opencda.log.logger_config import logger

def restore_carla_rendering(client_port, timeout=5.0):
    """
    Re-enable CARLA rendering before a scenario starts.
    Scenario shutdown may intentionally leave CARLA in no-rendering mode.
    """
    try:
        import carla
        client = carla.Client('localhost', client_port)
        client.set_timeout(timeout)
        world = client.get_world()
        settings = world.get_settings()
        if settings.no_rendering_mode:
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            print("CARLA rendering restored.")
    except Exception as e:
        logger.warning(f"Unable to restore CARLA rendering before scenario start: {e}")


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # add arguments to the parser
    parser.add_argument('-t', "--test_scenario", required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
    parser.add_argument("--record", action='store_true',
                        help='whether to record and save the simulation process to .log file')
    parser.add_argument("--apply_ml",
                        action='store_true',
                        help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                             'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument("--apply_cp",
                        action='store_true',
                        help='whether to apply coperception.')
    parser.add_argument("--prediction",
                        action='store_true',
                        help='whether to enable prediction.')
    parser.add_argument("--network",
                        action='store_true',
                        help='whether to enable network.')
    parser.add_argument('-v', "--version", type=str, default='0.9.11',
                        help='Specify the CARLA simulator version, default'
                             'is 0.9.11, 0.9.12 is also supported.')
    parser.add_argument("--debug",
                        action='store_true',
                        help='whether to enable debug mode.')
    parser.add_argument("--uav",
                        action='store_true',
                        help='whether to enable UAV.')
    parser.add_argument("--dump",
                        action='store_true',
                        help='whether to dump OPV2V-style sensor data instead of running online ML inference.')
    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


def main():
    # parse the arguments
    opt = arg_parse()
    logger.info(opt)
    # print the version of OpenCDA
    print("OpenCDA Version: %s" % __version__)
    # set the default yaml file
    default_yaml = config_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/default.yaml')
    # set the yaml file for the specific testing scenario
    config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'opencda/scenario_testing/config_yaml/%s.yaml' % opt.test_scenario)
    # coperception default yaml
    coperception_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/enable_coperception.yaml')

    # open scenario default yaml
    open_scenario_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/openscenario_default.yaml')

    prediction_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/enable_prediction.yaml')
    
    network_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/enable_network.yaml')

    uav_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/enable_uav.yaml')

    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)
    open_scenario_dict = OmegaConf.load(open_scenario_yaml)
    network_dict = OmegaConf.load(network_yaml)

    network_dict['enable_network']['network']['enabled'] = opt.network

    # coperception & prediction
    coperception_dict = OmegaConf.load(coperception_yaml)
    enable_prediction_dict = OmegaConf.load(prediction_yaml)
    uav_dict = OmegaConf.load(uav_yaml)
    # merge the dictionaries
    scene_dict = OmegaConf.merge(default_dict, scene_dict, open_scenario_dict)
    # import the testing script
    experiment_dict = OmegaConf.merge(coperception_dict, enable_prediction_dict, network_dict, uav_dict)
    # add network_dict here

    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)
    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(
            "opencda/scenario_testing/config_yaml/%s.yaml not found!" % opt.test_cenario)

    # get the function for running the scenario from the testing script
    scenario_runner = getattr(testing_scenario, 'run_scenario')

    logger.debug(experiment_dict)
    scenario_params = scene_dict

    if opt.apply_cp:
        scenario_params = OmegaConf.merge(scenario_params, experiment_dict['enable_coperception'])
    if opt.network:
        scenario_params = OmegaConf.merge(scenario_params, experiment_dict['enable_network'])
        scenario_params['vehicle_base']['v2x'].update(network_dict['enable_network'])
        scenario_params['traffic_vehicle_base']['v2x'].update(network_dict['enable_network'])
    if opt.prediction:
        scenario_params = OmegaConf.merge(scenario_params, experiment_dict['enable_prediction'])
    if opt.uav:
        scenario_params = OmegaConf.merge(scenario_params, uav_dict)

    scenario_params['vehicle_base']['sensing']['perception']['coperception'] = opt.apply_cp
    scenario_params['vehicle_base']['sensing']['perception']['activate'] = opt.apply_ml
    if opt.dump:
        scenario_params['vehicle_base']['sensing']['perception']['coperception'] = False
        scenario_params['vehicle_base']['sensing']['perception']['activate'] = False
        scenario_params['traffic_vehicle_base']['sensing']['perception']['coperception'] = False
        scenario_params['traffic_vehicle_base']['sensing']['perception']['activate'] = False

    #ignore deprecated warning 
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=UserWarning, message="nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, message="nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.")
    logger.debug(scenario_params)
    restore_carla_rendering(scenario_params['world']['client_port'])
    scenario_runner(opt, scenario_params)


if __name__ == '__main__':
    # try:
        main()
    # except KeyboardInterrupt:
    #     print(' - Exited by user.')
