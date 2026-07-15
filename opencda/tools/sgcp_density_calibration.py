# -*- coding: utf-8 -*-
"""
Export SGCP density calibration statistics from an OPV2V-style data dump.

The tool does not run CARLA or OpenCOOD. It rebuilds the same offline LiDAR
grid state used by SGCP replay and summarizes the empirical rho distribution
that motivates the density utility function f(rho).
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import OfflineCavWorld
from opencda.core.clustering.utils.metrics import density_score
from opencda.tools.offline_replay import (
    extract_lidar_density_threshold,
    select_cav_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export SGCP f(rho) density calibration statistics.')
    parser.add_argument('--dataset-root', required=True,
                        help='Root folder containing scenario subfolders.')
    parser.add_argument('--scenario-id', default=None,
                        help='Scenario folder name. Defaults to the first one.')
    parser.add_argument('--ego-cav-id', default='1',
                        help='Ego CAV id used to load frame transforms.')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Number of frames to use. Use 0 for all frames.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from within the scenario.')
    parser.add_argument('--cav-count', type=int, default=None,
                        help='Use the first N CAVs in numeric order.')
    parser.add_argument('--cav-ids', default=None,
                        help='Comma-separated CAV ids, e.g. 1,2,3.')
    parser.add_argument('--thresholds', default='0.5,1.0,2.0,3.0,4.0',
                        help='Comma-separated rho_th values to summarize.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for calibration CSV and notes.')
    return parser.parse_args()


def load_protocol(dataset, scenario_id):
    protocol_path = os.path.join(
        dataset.scenarios[scenario_id]['path'],
        'data_protocol.yaml')
    if not os.path.exists(protocol_path):
        return {}
    with open(protocol_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)


def parse_thresholds(value):
    thresholds = []
    for item in value.split(','):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    if not thresholds:
        raise ValueError('--thresholds must contain at least one value')
    return thresholds


def select_timestamps(dataset, scenario_id, start_index, max_frames):
    timestamps = dataset.scenarios[scenario_id]['timestamps']
    if max_frames == 0:
        return timestamps[start_index:]
    return timestamps[start_index:start_index + max_frames]


def percentile(values, q):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, q))


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_frame(scenario_id, timestamp, world):
    densities = []
    nonzero = 0
    high_default = 0
    grid_total = 0
    point_total = 0

    for vehicle_id, vm in world.get_vehicle_managers().items():
        lidar = vm.perception_manager.lidar
        vehicle_densities = list(lidar.grid_density_dict.values())
        densities.extend(vehicle_densities)
        grid_total += len(vehicle_densities)
        nonzero += sum(1 for value in vehicle_densities if value > 0.0)
        high_default += sum(
            1 for value in vehicle_densities
            if value >= lidar.density_threshold)
        point_total += int(lidar.local_data.shape[0])

    density_array = np.asarray(densities, dtype=np.float64)
    nonzero_density_array = density_array[density_array > 0.0]
    mean_density = float(np.mean(density_array)) if len(density_array) else 0.0
    nonzero_mean = (
        float(np.mean(nonzero_density_array))
        if len(nonzero_density_array) else 0.0)

    return OrderedDict({
        'scenario_id': scenario_id,
        'timestamp': timestamp,
        'cav_count': len(world.get_vehicle_managers()),
        'grid_count': grid_total,
        'point_count': point_total,
        'nonzero_grid_ratio': (
            0.0 if grid_total == 0 else float(nonzero) / grid_total),
        'default_high_density_ratio': (
            0.0 if grid_total == 0 else float(high_default) / grid_total),
        'density_mean': mean_density,
        'density_p50': percentile(density_array, 50),
        'density_p75': percentile(density_array, 75),
        'density_p90': percentile(density_array, 90),
        'density_p95': percentile(density_array, 95),
        'density_p99': percentile(density_array, 99),
        'density_max': float(np.max(density_array)) if len(density_array) else 0.0,
        'nonzero_density_mean': nonzero_mean,
        'nonzero_density_p50': percentile(nonzero_density_array, 50),
        'nonzero_density_p75': percentile(nonzero_density_array, 75),
        'nonzero_density_p90': percentile(nonzero_density_array, 90),
        'nonzero_density_p95': percentile(nonzero_density_array, 95),
        'nonzero_density_p99': percentile(nonzero_density_array, 99),
    }), density_array


def threshold_rows(thresholds, all_densities):
    rows = []
    total = len(all_densities)
    nonzero_mask = all_densities > 0.0
    nonzero_total = int(np.sum(nonzero_mask))
    for threshold in thresholds:
        high_mask = all_densities >= threshold
        high_count = int(np.sum(high_mask))
        utility_values = np.asarray(
            [density_score(value, threshold) for value in all_densities],
            dtype=np.float64)
        rows.append(OrderedDict({
            'rho_th': '%.6f' % threshold,
            'grid_count': total,
            'nonzero_grid_count': nonzero_total,
            'high_density_grid_count': high_count,
            'high_density_ratio_all': (
                '0.000000' if total == 0
                else '%.6f' % (float(high_count) / total)),
            'high_density_ratio_nonzero': (
                '0.000000' if nonzero_total == 0
                else '%.6f' % (float(np.sum(high_mask & nonzero_mask)) /
                               nonzero_total)),
            'f_rho_mean': '%.6f' % (
                float(np.mean(utility_values)) if total else 0.0),
            'f_rho_p50': '%.6f' % percentile(utility_values, 50),
            'f_rho_p90': '%.6f' % percentile(utility_values, 90),
        }))
    return rows


def curve_rows(thresholds, all_densities):
    max_density = float(np.max(all_densities)) if len(all_densities) else 0.0
    upper = max(5.0, np.ceil(max_density * 2.0) / 2.0)
    bins = np.linspace(0.0, upper, int(upper / 0.25) + 1)
    counts, edges = np.histogram(all_densities, bins=bins)
    rows = []
    total = float(len(all_densities))
    for index, count in enumerate(counts):
        rho_mid = float((edges[index] + edges[index + 1]) / 2.0)
        row = OrderedDict({
            'rho_bin_left': '%.6f' % float(edges[index]),
            'rho_bin_right': '%.6f' % float(edges[index + 1]),
            'rho_mid': '%.6f' % rho_mid,
            'grid_count': int(count),
            'grid_ratio': (
                '0.000000' if total == 0.0 else '%.6f' % (count / total)),
        })
        for threshold in thresholds:
            row['f_rho_sigmoid_th_%.3g' % threshold] = (
                '%.6f' % density_score(rho_mid, threshold))
        rows.append(row)
    return rows


def global_summary_rows(all_densities):
    nonzero_density_array = all_densities[all_densities > 0.0]
    total = len(all_densities)
    nonzero_total = len(nonzero_density_array)
    return [OrderedDict({
        'grid_count': total,
        'nonzero_grid_count': nonzero_total,
        'nonzero_grid_ratio': (
            '0.000000' if total == 0
            else '%.6f' % (float(nonzero_total) / total)),
        'density_mean_all': '%.6f' % (
            float(np.mean(all_densities)) if total else 0.0),
        'density_p95_all': '%.6f' % percentile(all_densities, 95),
        'density_p99_all': '%.6f' % percentile(all_densities, 99),
        'density_max_all': '%.6f' % (
            float(np.max(all_densities)) if total else 0.0),
        'density_mean_nonzero': '%.6f' % (
            float(np.mean(nonzero_density_array))
            if nonzero_total else 0.0),
        'density_p50_nonzero': '%.6f' % percentile(
            nonzero_density_array, 50),
        'density_p75_nonzero': '%.6f' % percentile(
            nonzero_density_array, 75),
        'density_p90_nonzero': '%.6f' % percentile(
            nonzero_density_array, 90),
        'density_p95_nonzero': '%.6f' % percentile(
            nonzero_density_array, 95),
        'density_p99_nonzero': '%.6f' % percentile(
            nonzero_density_array, 99),
        'density_max_nonzero': '%.6f' % (
            float(np.max(nonzero_density_array))
            if nonzero_total else 0.0),
    })]


def main():
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    dataset = OPV2VFrameDataset(args.dataset_root)
    scenario_id = args.scenario_id or next(iter(dataset.scenarios))
    protocol = load_protocol(dataset, scenario_id)
    default_threshold = extract_lidar_density_threshold(protocol)
    selected_cav_ids = select_cav_ids(
        dataset,
        scenario_id,
        ego_cav_id=args.ego_cav_id,
        cav_count=args.cav_count,
        cav_ids=args.cav_ids)
    timestamps = select_timestamps(
        dataset,
        scenario_id,
        args.start_index,
        args.max_frames)

    os.makedirs(args.output_dir, exist_ok=True)

    frame_rows = []
    density_chunks = []
    for timestamp in timestamps:
        frame = dataset.load_frame(
            scenario_id,
            timestamp,
            ego_cav_id=args.ego_cav_id,
            cav_ids=selected_cav_ids)
        world = OfflineCavWorld(frame, ego_id=args.ego_cav_id,
                                protocol=protocol)
        row, densities = summarize_frame(scenario_id, timestamp, world)
        frame_rows.append(row)
        density_chunks.append(densities)

    all_densities = (
        np.concatenate(density_chunks)
        if density_chunks else np.asarray([], dtype=np.float64))

    frame_fields = [
        'scenario_id', 'timestamp', 'cav_count', 'grid_count', 'point_count',
        'nonzero_grid_ratio', 'default_high_density_ratio', 'density_mean',
        'density_p50', 'density_p75', 'density_p90', 'density_p95',
        'density_p99', 'density_max', 'nonzero_density_mean',
        'nonzero_density_p50', 'nonzero_density_p75',
        'nonzero_density_p90', 'nonzero_density_p95',
        'nonzero_density_p99',
    ]
    write_csv(os.path.join(args.output_dir, 'frame_density_summary.csv'),
              frame_rows, frame_fields)

    global_fields = list(global_summary_rows(all_densities)[0].keys())
    write_csv(os.path.join(args.output_dir, 'global_density_summary.csv'),
              global_summary_rows(all_densities), global_fields)

    threshold_fields = [
        'rho_th', 'grid_count', 'nonzero_grid_count',
        'high_density_grid_count', 'high_density_ratio_all',
        'high_density_ratio_nonzero', 'f_rho_mean', 'f_rho_p50',
        'f_rho_p90',
    ]
    write_csv(os.path.join(args.output_dir, 'threshold_summary.csv'),
              threshold_rows(thresholds, all_densities), threshold_fields)

    curve = curve_rows(thresholds, all_densities)
    curve_fields = list(curve[0].keys()) if curve else [
        'rho_bin_left', 'rho_bin_right', 'rho_mid', 'grid_count', 'grid_ratio']
    write_csv(os.path.join(args.output_dir, 'f_rho_curve.csv'),
              curve, curve_fields)

    note_path = os.path.join(args.output_dir, 'run_notes.md')
    with open(note_path, 'w') as stream:
        stream.write('# SGCP density calibration\n\n')
        stream.write('- scenario_id: `%s`\n' % scenario_id)
        stream.write('- frames: `%d`\n' % len(timestamps))
        stream.write('- ego_cav_id: `%s`\n' % args.ego_cav_id)
        stream.write('- cav_ids: `%s`\n' % (
            ','.join(selected_cav_ids) if selected_cav_ids else 'all'))
        stream.write('- default_rho_th_from_protocol: `%.6f`\n' %
                     default_threshold)
        stream.write('- thresholds: `%s`\n' % ','.join(
            '%.6f' % value for value in thresholds))
        stream.write('\nOutputs:\n\n')
        stream.write('- `frame_density_summary.csv`\n')
        stream.write('- `global_density_summary.csv`\n')
        stream.write('- `threshold_summary.csv`\n')
        stream.write('- `f_rho_curve.csv`\n')

    print('SGCP density calibration written to %s' % args.output_dir)
    print('frames=%d grids=%d default_rho_th=%.3f' % (
        len(timestamps), len(all_densities), default_threshold))


if __name__ == '__main__':
    main()
