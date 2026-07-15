import opencda.scenario_testing.template as template
import os

def run_scenario(opt, scenario_params):
    filename = os.path.splitext(os.path.basename(__file__))[0]
    if getattr(opt, 'dump', False):
        application = ['data_dump']
        max_ticks = int(os.environ.get('OPENCDA_DATADUMP_TICKS', '140'))
        town = 'Town03'
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
        return

    application = ['cluster']
    town = 'Town03'
    # town = 'Town05'
    template.run_scenario(opt=opt, scenario_params=scenario_params, filename=filename, application=application, town=town)

