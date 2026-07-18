# -*- coding: utf-8 -*-
"""
Estimate raw area-slice bytes for flat selective-sharing baselines.

The selected agents come from lgcp_subset_ablation_eval.py. The area cells come
from an LGCP hierarchy assignment plan. For each flat selected non-ego agent,
this tool crops raw LiDAR points to the planned LGCP areas and reports the
area-slice byte proxy. It does not rerun perception.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencood.utils.transformation_utils import x_to_world


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate flat baseline raw area-slice bytes.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--subset-frame-records', required=True)
    parser.add_argument('--ablation-summary', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_agents(value):
    return [item for item in str(value).split(';') if item]


def pointcloud_to_world(points, lidar_pose):
    if points.size == 0:
        return points[:, :3]
    matrix = x_to_world(lidar_pose)
    xyz = points[:, :3]
    homo = np.concatenate(
        [xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)],
        axis=1)
    return np.dot(matrix, homo.T).T[:, :3]


def crop_area(points_world, center_x, center_y, grid_x, grid_y):
    if points_world.size == 0:
        return np.zeros((0,), dtype=bool)
    half_x = grid_x / 2.0
    half_y = grid_y / 2.0
    return (
        (points_world[:, 0] >= center_x - half_x) &
        (points_world[:, 0] < center_x + half_x) &
        (points_world[:, 1] >= center_y - half_y) &
        (points_world[:, 1] < center_y + half_y)
    )


def build_area_lookup(assignments):
    areas = defaultdict(list)
    for row in assignments:
        areas[row['timestamp']].append((
            row['area_id'],
            float(row['area_center_x']),
            float(row['area_center_y'])))
    return areas


def build_ap_lookup(summary_rows):
    return {
        (row['method'], row['budget']): row
        for row in summary_rows
    }


def load_world_points(dataset, scenario_id, timestamp, cav_id):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=cav_id,
        cav_ids=[cav_id],
        add_transformation=False)
    key = next(iter(frame.keys()))
    cav = frame[key]
    return pointcloud_to_world(
        cav['lidar_np'],
        cav['params']['lidar_pose'])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = OPV2VFrameDataset(args.dataset_root)
    subset_rows = read_csv(args.subset_frame_records)
    areas_by_timestamp = build_area_lookup(read_csv(args.assignment_plan))
    ap_lookup = build_ap_lookup(read_csv(args.ablation_summary))
    point_cache = {}

    detail_rows = []
    stats = defaultdict(lambda: {
        'frames': 0,
        'selected_agents': 0,
        'non_ego_agents': 0,
        'area_rows': 0,
        'slice_points': 0,
        'slice_bytes': 0,
    })

    for row in subset_rows:
        scenario_id = row.get('scenario_id') or args.scenario_id
        timestamp = row['timestamp']
        method = row['method']
        budget = row['budget']
        areas = areas_by_timestamp.get(timestamp, [])
        selected = parse_agents(row['selected_agents'])
        non_ego = [agent for agent in selected
                   if str(agent) != str(args.ego_cav_id)]

        frame_points = 0
        frame_area_rows = 0
        for agent in non_ego:
            key = (scenario_id, timestamp, str(agent))
            if key not in point_cache:
                point_cache[key] = load_world_points(
                    dataset,
                    scenario_id,
                    timestamp,
                    str(agent))
            points_world = point_cache[key]
            for area_id, center_x, center_y in areas:
                mask = crop_area(
                    points_world,
                    center_x,
                    center_y,
                    args.grid_size_x,
                    args.grid_size_y)
                point_count = int(np.count_nonzero(mask))
                frame_points += point_count
                frame_area_rows += 1

        frame_bytes = frame_points * args.bytes_per_point
        detail_rows.append(OrderedDict({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'method': method,
            'budget': budget,
            'planned_areas': len(areas),
            'selected_agents': ';'.join(selected),
            'non_ego_agents': ';'.join(non_ego),
            'non_ego_count': len(non_ego),
            'area_agent_rows': frame_area_rows,
            'slice_points': frame_points,
            'slice_bytes': frame_bytes,
        }))

        item = stats[(method, budget)]
        item['frames'] += 1
        item['selected_agents'] += len(selected)
        item['non_ego_agents'] += len(non_ego)
        item['area_rows'] += frame_area_rows
        item['slice_points'] += frame_points
        item['slice_bytes'] += frame_bytes

    summary_rows = []
    for (method, budget), item in sorted(stats.items()):
        frames = max(1, item['frames'])
        ap = ap_lookup.get((method, budget), {})
        summary_rows.append(OrderedDict({
            'method': method,
            'budget': budget,
            'frames': item['frames'],
            'selected_mean': '%.6f' % (
                item['selected_agents'] / float(frames)),
            'non_ego_selected_mean': '%.6f' % (
                item['non_ego_agents'] / float(frames)),
            'area_agent_rows_mean': '%.6f' % (
                item['area_rows'] / float(frames)),
            'slice_points_total': item['slice_points'],
            'slice_points_mean': '%.6f' % (
                item['slice_points'] / float(frames)),
            'slice_bytes_total': item['slice_bytes'],
            'slice_bytes_mean': '%.6f' % (
                item['slice_bytes'] / float(frames)),
            'bytes_per_point': args.bytes_per_point,
            'ap_03': ap.get('ap_03', ''),
            'ap_05': ap.get('ap_05', ''),
            'ap_07': ap.get('ap_07', ''),
            'gt_total': ap.get('gt_total', ''),
            'pred_samples': ap.get('pred_samples', ''),
        }))

    write_csv(os.path.join(args.output_dir, 'flat_area_slice_frame_records.csv'),
              ['scenario_id', 'timestamp', 'method', 'budget',
               'planned_areas', 'selected_agents', 'non_ego_agents',
               'non_ego_count', 'area_agent_rows', 'slice_points',
               'slice_bytes'],
              detail_rows)
    write_csv(os.path.join(args.output_dir, 'flat_area_slice_summary.csv'),
              ['method', 'budget', 'frames', 'selected_mean',
               'non_ego_selected_mean', 'area_agent_rows_mean',
               'slice_points_total', 'slice_points_mean',
               'slice_bytes_total', 'slice_bytes_mean', 'bytes_per_point',
               'ap_03', 'ap_05', 'ap_07', 'gt_total', 'pred_samples'],
              summary_rows)

    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'subset_frame_records': os.path.abspath(args.subset_frame_records),
        'ablation_summary': os.path.abspath(args.ablation_summary),
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'ego_cav_id': str(args.ego_cav_id),
        'grid_size_x': args.grid_size_x,
        'grid_size_y': args.grid_size_y,
        'bytes_per_point': args.bytes_per_point,
        'note': (
            'Raw area-slice bytes for existing flat selected-agent decisions; '
            'does not rerun perception.'),
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# Flat Area-Slice Byte Accounting\n\n')
        stream.write('This run preserves existing flat baseline selected ')
        stream.write('agents and estimates bytes if only the planned LGCP ')
        stream.write('area cells were transmitted as raw point slices. It ')
        stream.write('does not rerun perception.\n\n')
        stream.write('- frame rows: `%d`\n' % len(detail_rows))
        stream.write('- summary rows: `%d`\n' % len(summary_rows))

    print('Wrote flat area-slice accounting to %s' % args.output_dir)
    for row in summary_rows:
        print('%s budget=%s slice_bytes_mean=%s AP50=%s' % (
            row['method'], row['budget'],
            row['slice_bytes_mean'], row['ap_05']))


if __name__ == '__main__':
    main()
