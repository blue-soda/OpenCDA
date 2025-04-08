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
from omegaconf import OmegaConf

from opencda.version import __version__
from opencda.log.logger_config import logger

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
    parser.add_argument('-v', "--version", type=str, default='0.9.11',
                        help='Specify the CARLA simulator version, default'
                             'is 0.9.11, 0.9.12 is also supported.')
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

    # open scenario default yaml
    prediction_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/enable_prediction.yaml')

    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)
    open_scenario_dict = OmegaConf.load(open_scenario_yaml)

    # coperception & prediction
    coperception_dict = OmegaConf.load(coperception_yaml)
    enable_prediction_dict = OmegaConf.load(prediction_yaml)
    # merge the dictionaries
    scene_dict = OmegaConf.merge(default_dict, scene_dict, open_scenario_dict)
    # import the testing script
    experiment_dict = OmegaConf.merge(coperception_dict, enable_prediction_dict)
    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)
    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(
            "opencda/scenario_testing/config_yaml/%s.yaml not found!" % opt.test_cenario)

    # get the function for running the scenario from the testing script
    scenario_runner = getattr(testing_scenario, 'run_scenario')

    from opencda.constants import Profile
    experiment_profile = Profile.PREDICTION_OPENCOOD_CAV
    for profile in experiment_profile.profiles():
        scenario_params = OmegaConf.merge(scene_dict, experiment_dict[profile])
    scenario_params['vehicle_base']['sensing']['perception']['coperception'] = opt.apply_cp
    scenario_params['vehicle_base']['sensing']['perception']['activate'] = opt.apply_ml

    #ignore deprecated warning 
    import warnings
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=UserWarning, message="nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    scenario_runner(opt, scenario_params)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
