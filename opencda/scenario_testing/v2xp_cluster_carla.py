import opencda.scenario_testing.template as template
import os

def run_scenario(opt, scenario_params):
    filename = os.path.splitext(os.path.basename(__file__))[0]
    application = ['cluster']
    town = 'Town06'
    template.run_scenario(opt=opt, scenario_params=scenario_params, filename=filename, application=application, town=town)

