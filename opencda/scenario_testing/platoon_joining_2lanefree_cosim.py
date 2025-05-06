import opencda.scenario_testing.template as template
import os

def run_scenario(opt, scenario_params):
    filename = os.path.splitext(os.path.basename(__file__))[0]
    application = ['platoon']
    current_path = os.path.dirname(os.path.realpath(__file__))
    xodr_path = os.path.join(
        current_path,
        '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr')
    sumo_cfg = os.path.join(current_path,
        '../assets/2lane_freeway_simplified')
    template.run_scenario(opt=opt, scenario_params=scenario_params, filename=filename, application=application, xdor_path=xodr_path, sumo_cfg=sumo_cfg)
