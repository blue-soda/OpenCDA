import math
from opencda.core.common.misc import compute_distance

def get_interference_contribution(source_vm, target_vm):
    """
    Calculate the interference contribution of a single source-target pair.
    Args:
        source_vm (VehicleManager): Source vehicle manager.
        target_vm (VehicleManager): Target vehicle manager.
    Returns:
        float: The interference contribution.
    """
    if not source_vm or not target_vm:
        return 0.0
    distance = calculate_distance(source_vm.v2x_manager, target_vm.v2x_manager)
    channel_gain = calculate_channel_gain(distance)
    tx_power = source_vm.v2x_manager.tx_power
    noise_level = target_vm.v2x_manager.noise_level
    return tx_power * channel_gain / noise_level

def calculate_distance(source_vehicle, target_vehicle):
    """Calculate the Euclidean distance between two vehicles."""
    source_pos = source_vehicle.get_ego_pos().location
    target_pos = target_vehicle.get_ego_pos().location
    return compute_distance(source_pos, target_pos)

def calculate_snr(tx_power, noise_level, distance):
    """Calculate the signal-to-noise ratio (SNR)."""
    return tx_power * calculate_channel_gain(distance) / noise_level

def calculate_channel_gain(distance):
    """
    Calculate the channel gain based on distance.

    Args:
        distance (float): Distance between source and target vehicles.

    Returns:
        float: The channel gain.
    """
    # Simplified path loss model: channel gain decreases with distance
    path_loss_exponent = 2.0  # Path loss exponent (free space = 2)
    reference_distance = 1.0  # Reference distance (1 meter)
    reference_gain = 1.0  # Reference gain at the reference distance
    return reference_gain / (distance / reference_distance) ** path_loss_exponent

def calculate_available_data_rate(subchannel_bandwidth, snr, interference):
    """Calculate the available data rate based on SNR and interference."""
    return subchannel_bandwidth * math.log2(1 + snr / interference)
