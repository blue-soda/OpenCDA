import os

import opencda.scenario_testing.template as template


def run_scenario(opt, scenario_params):
    filename = os.path.splitext(os.path.basename(__file__))[0]
    application = ['data_dump']
    town = 'Town03'

    # Data dumping should use CARLA ground truth labels and raw sensor frames.
    # Keep coperception/cluster state available, but do not run online ML
    # detection while exporting labels.
    scenario_params['vehicle_base']['sensing']['perception']['activate'] = False
    scenario_params['traffic_vehicle_base']['sensing']['perception']['activate'] = False

    max_ticks = int(os.environ.get('OPENCDA_DATADUMP_TICKS', '140'))

    try:
        template.applications = application
        template.town_name = town
        template.xodr_path_name = None
        template.file_name = filename
        template.sumo_cfg_name = None
        template.init(opt, scenario_params)
        for _ in range(max_ticks):
            template._tick_once(debug=opt.debug)
    finally:
        template.stop(opt)
