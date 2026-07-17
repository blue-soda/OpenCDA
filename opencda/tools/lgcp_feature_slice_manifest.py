# -*- coding: utf-8 -*-
"""
Create an offline LGCP area-specific feature-slice manifest.

This is the next step after hierarchy assignment planning. It does not extract
neural feature tensors yet; instead, it slices raw LiDAR points by LGCP area in
world coordinates and records the per-area/per-agent slice size that a real
feature-slice transport path should replace.
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
        description='Export LGCP area-specific feature-slice manifest.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Limit timestamps. 0 means all in plan.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def selected_timestamps(assignments, max_frames):
    timestamps = sorted({row['timestamp'] for row in assignments})
    if max_frames > 0:
        timestamps = timestamps[:max_frames]
    return timestamps


def frame_member_ids(assignments):
    ids_by_timestamp = defaultdict(set)
    for row in assignments:
        timestamp = row['timestamp']
        leader = str(row['leader_id'])
        ids_by_timestamp[timestamp].add(leader)
        for member in row['group_members'].split(';'):
            if member:
                ids_by_timestamp[timestamp].add(str(member))
    return ids_by_timestamp


def build_manifest(dataset, scenario_id, assignments, args):
    timestamps = selected_timestamps(assignments, args.max_frames)
    timestamp_set = set(timestamps)
    assignments = [row for row in assignments if row['timestamp'] in timestamp_set]
    ids_by_timestamp = frame_member_ids(assignments)
    assignments_by_timestamp = defaultdict(list)
    for row in assignments:
        assignments_by_timestamp[row['timestamp']].append(row)

    slice_rows = []
    frame_rows = []
    area_rows = []

    for timestamp in timestamps:
        cav_ids = sorted(ids_by_timestamp[timestamp], key=lambda item: int(item))
        frame = dataset.load_frame(
            scenario_id,
            timestamp,
            cav_ids=cav_ids,
            add_transformation=False)
        world_points = {}
        raw_counts = {}
        for cav_key, cav in frame.items():
            cav_id = str(cav_key)
            points = cav['lidar_np']
            raw_counts[cav_id] = int(points.shape[0])
            world_points[cav_id] = pointcloud_to_world(
                points,
                cav['params']['lidar_pose'])

        frame_slice_points = 0
        frame_upload_points = 0
        frame_self_points = 0
        frame_upload_bytes = 0
        frame_self_bytes = 0

        for assignment in assignments_by_timestamp[timestamp]:
            area_id = assignment['area_id']
            center_x = float(assignment['area_center_x'])
            center_y = float(assignment['area_center_y'])
            leader = str(assignment['leader_id'])
            group_members = [
                str(item) for item in assignment['group_members'].split(';')
                if item]
            area_point_total = 0
            area_upload_points = 0
            area_self_points = 0

            for member in group_members:
                points_world = world_points.get(member)
                if points_world is None:
                    point_count = 0
                    raw_count = 0
                else:
                    mask = crop_area(
                        points_world,
                        center_x,
                        center_y,
                        args.grid_size_x,
                        args.grid_size_y)
                    point_count = int(np.count_nonzero(mask))
                    raw_count = raw_counts.get(member, 0)

                is_leader = member == leader
                upload_type = 'leader_self' if is_leader else 'member_to_leader'
                byte_proxy = point_count * args.bytes_per_point
                frame_slice_points += point_count
                area_point_total += point_count
                if is_leader:
                    frame_self_points += point_count
                    frame_self_bytes += byte_proxy
                    area_self_points += point_count
                else:
                    frame_upload_points += point_count
                    frame_upload_bytes += byte_proxy
                    area_upload_points += point_count

                slice_rows.append(OrderedDict({
                    'timestamp': timestamp,
                    'area_id': area_id,
                    'area_center_x': '%.3f' % center_x,
                    'area_center_y': '%.3f' % center_y,
                    'agent_id': member,
                    'leader_id': leader,
                    'upload_type': upload_type,
                    'raw_point_count': raw_count,
                    'slice_point_count': point_count,
                    'slice_ratio': '%.6f' % (
                        point_count / float(raw_count or 1)),
                    'byte_proxy': byte_proxy,
                }))

            area_rows.append(OrderedDict({
                'timestamp': timestamp,
                'area_id': area_id,
                'leader_id': leader,
                'group_size': len(group_members),
                'area_slice_points': area_point_total,
                'member_upload_points': area_upload_points,
                'leader_self_points': area_self_points,
                'member_upload_bytes': area_upload_points * args.bytes_per_point,
                'leader_self_bytes': area_self_points * args.bytes_per_point,
            }))

        frame_rows.append(OrderedDict({
            'timestamp': timestamp,
            'areas': len(assignments_by_timestamp[timestamp]),
            'slice_rows': sum(
                int(row['timestamp'] == timestamp) for row in slice_rows),
            'slice_points_total': frame_slice_points,
            'member_upload_points': frame_upload_points,
            'leader_self_points': frame_self_points,
            'member_upload_bytes': frame_upload_bytes,
            'leader_self_bytes': frame_self_bytes,
        }))

    return slice_rows, area_rows, frame_rows


def summarize(frame_rows):
    if not frame_rows:
        return []
    fields = [
        'areas',
        'slice_rows',
        'slice_points_total',
        'member_upload_points',
        'leader_self_points',
        'member_upload_bytes',
        'leader_self_bytes',
    ]
    row = OrderedDict({'frames': len(frame_rows)})
    for field in fields:
        values = np.asarray([float(item[field]) for item in frame_rows],
                            dtype=np.float64)
        row[field + '_mean'] = '%.6f' % float(np.mean(values))
        row[field + '_max'] = '%.6f' % float(np.max(values))
    return [row]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = OPV2VFrameDataset(args.dataset_root)
    assignments = read_csv(args.assignment_plan)
    slice_rows, area_rows, frame_rows = build_manifest(
        dataset,
        args.scenario_id,
        assignments,
        args)
    summary_rows = summarize(frame_rows)

    write_csv(os.path.join(args.output_dir, 'feature_slice_manifest.csv'),
              ['timestamp', 'area_id', 'area_center_x', 'area_center_y',
               'agent_id', 'leader_id', 'upload_type', 'raw_point_count',
               'slice_point_count', 'slice_ratio', 'byte_proxy'],
              slice_rows)
    write_csv(os.path.join(args.output_dir, 'feature_slice_area_summary.csv'),
              ['timestamp', 'area_id', 'leader_id', 'group_size',
               'area_slice_points', 'member_upload_points',
               'leader_self_points', 'member_upload_bytes',
               'leader_self_bytes'],
              area_rows)
    write_csv(os.path.join(args.output_dir, 'feature_slice_frame_summary.csv'),
              ['timestamp', 'areas', 'slice_rows', 'slice_points_total',
               'member_upload_points', 'leader_self_points',
               'member_upload_bytes', 'leader_self_bytes'],
              frame_rows)
    write_csv(os.path.join(args.output_dir, 'feature_slice_summary.csv'),
              ['frames'] + [
                  field + suffix
                  for field in [
                      'areas', 'slice_rows', 'slice_points_total',
                      'member_upload_points', 'leader_self_points',
                      'member_upload_bytes', 'leader_self_bytes']
                  for suffix in ('_mean', '_max')
              ],
              summary_rows)

    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'grid_size_x': args.grid_size_x,
        'grid_size_y': args.grid_size_y,
        'bytes_per_point': args.bytes_per_point,
        'max_frames': args.max_frames,
        'note': 'Raw LiDAR area-slice proxy; not neural feature tensors.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Feature Slice Manifest\n\n')
        stream.write('This run crops raw LiDAR points by LGCP area in world ')
        stream.write('coordinates to materialize the area-specific slice ')
        stream.write('interface. It is a proxy for future neural feature ')
        stream.write('slicing, not the final feature tensor transport.\n\n')
        stream.write('- slice_rows: `%d`\n' % len(slice_rows))
        stream.write('- area_rows: `%d`\n' % len(area_rows))
        stream.write('- frame_rows: `%d`\n' % len(frame_rows))

    print('Wrote LGCP feature slice manifest to %s' % args.output_dir)
    if summary_rows:
        row = summary_rows[0]
        print('frames=%s areas_mean=%s upload_points_mean=%s '
              'upload_bytes_mean=%s self_points_mean=%s' % (
                  row['frames'],
                  row['areas_mean'],
                  row['member_upload_points_mean'],
                  row['member_upload_bytes_mean'],
                  row['leader_self_points_mean']))


if __name__ == '__main__':
    main()
