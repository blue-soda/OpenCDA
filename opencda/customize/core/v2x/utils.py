import math
from opencda.core.common.misc import compute_distance

def get_interference_contribution(source_vm, target_vm):
    """
    Calculate the interference contribution of a single source-target pair.
    Args:
        source_vm (V2XManager): Source vehicle manager.
        target_vm (V2XManager): Target vehicle manager.
    Returns:
        float: The interference contribution (W).
    """
    if not source_vm or not target_vm:
        return 0.0
    distance = calculate_distance(source_vm, target_vm)
    channel_gain = calculate_channel_gain(distance)
    tx_power = source_vm.tx_power
    tx_power_w = 10 ** (source_vm.tx_power / 10) / 1000  # dBm转W
    return tx_power_w * channel_gain

def calculate_distance(source_vm, target_vm):
    """Calculate the Euclidean distance between two vehicles."""
    source_pos = source_vm.get_ego_pos().location
    target_pos = target_vm.get_ego_pos().location
    return compute_distance(source_pos, target_pos)

def calculate_snr(tx_power, noise_level, distance):
    """Calculate the signal-to-noise ratio (SNR)."""
    tx_power_w = 10 ** (tx_power / 10) / 1000 # dBm转W
    return tx_power_w * calculate_channel_gain(distance) / noise_level

def calculate_sinr(tx_power, interference_power, noise_power):
    """Calculate the Signal-to-Interference-plus-Noise Ratio (SINR) in dB."""
    denominator = interference_power + noise_power
    linear_value = tx_power / denominator if denominator > 0 else 0
    return 10 * math.log10(linear_value) if linear_value > 0 else -math.inf

def calculate_channel_gain(distance, path_loss_exponent=2.0):
    """
    Calculate the channel gain based on distance.

    Args:
        distance (float): Distance between source and target vehicles.

    Returns:
        float: The channel gain.
    """
    # Simplified path loss model: channel gain decreases with distance
    # path_loss_exponent = 2.0  # Path loss exponent (free space = 2)
    reference_distance = 10.0  # Reference distance (10 meters)
    reference_gain = 1.0  # Reference gain at the reference distance
    return reference_gain / (distance / reference_distance) ** path_loss_exponent

def calculate_available_data_rate(subchannel_bandwidth, sinr):
    """Calculate the available data rate(bps) based on SNR and interference."""
    return subchannel_bandwidth * math.log2(1 + sinr)

def is_link_conflict(s1, t1, s2, t2, sinr_threshold_dB) -> bool:
    """Determine if two links conflict based on SINR threshold."""
    """s1-t1 and s2-t2 are two links, four V2XManager objects."""
    if not all([s1, t1, s2, t2]):
        return False
    
    interf_power = get_interference_contribution(s2, t1)
    tx_power_w = 10 ** (s1.tx_power / 10) / 1000
    distance = calculate_distance(s1, t1)
    channel_gain = calculate_channel_gain(distance)
    signal_power = tx_power_w * channel_gain
    sinr_dB = calculate_sinr(signal_power, interf_power, t1.noise_power)
    
    interf_power_rev = get_interference_contribution(s1, t2)
    tx_power_w2 = 10 ** (s2.tx_power / 10) / 1000
    distance2 = calculate_distance(s2, t2)
    channel_gain2 = calculate_channel_gain(distance2)
    signal_power2 = tx_power_w2 * channel_gain2
    sinr_dB_rev = calculate_sinr(signal_power2, interf_power_rev, t2.noise_power)
    
    return sinr_dB < sinr_threshold_dB or sinr_dB_rev < sinr_threshold_dB