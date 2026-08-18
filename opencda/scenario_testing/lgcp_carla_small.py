"""Small LGCP Town03 scenario.

This runner intentionally reuses the LGCP CARLA lifecycle while pairing it
with a smaller YAML configuration. Keeping a separate scenario name lets us
create easier diagnostic datasets without changing the 100-vehicle setup.
"""

from opencda.scenario_testing.lgcp_carla import run_scenario
