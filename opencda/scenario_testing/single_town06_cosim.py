import opencda.scenario_testing.template as template
import os

def run_scenario(opt, scenario_params):
    filename = os.path.splitext(os.path.basename(__file__))[0]
    application = []
    town = 'Town06'
    current_path = os.path.dirname(os.path.realpath(__file__))
    sumo_cfg = os.path.join(current_path,
                            '../assets/Town06')
    template.run_scenario(opt=opt, scenario_params=scenario_params, filename=filename, application=application, town=town, sumo_cfg=sumo_cfg)

