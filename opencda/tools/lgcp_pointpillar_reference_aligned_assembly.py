# -*- coding: utf-8 -*-
"""
Assemble LGCP leader feature slices in a shared reference lidar frame.

This tool maps each world-coordinate LGCP area cell to a reference CAV lidar
frame and places leader-local feature slices onto that reference canvas. The
placement is an approximate nearest-neighbor resize, not a learned or
geometrically exact feature warp; it is intended to diagnose coordinate
alignment before claiming model-level AP.
"""

import argparse
import csv
import math
import os
from collections import OrderedDict, defaultdict

import numpy as np
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencood.utils.transformation_utils import x_to_world


DEFAULT_LIDAR_RANGE = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]
DEFAULT_VOXEL_SIZE = [0.4, 0.4, 4.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Assemble LGCP feature slices in a reference lidar frame.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--leader-root', required=True)
    parser.add_argument('--leader-feature-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-cav-id', default='1')
    parser.add_argument('--feature-key', default='leader_scatter_mean')
    parser.add_argument('--canvas-height', type=int, default=200)
    parser.add_argument('--canvas-width', type=int, default=704)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--dtype', choices=['float32', 'float16'],
                        default='float16')
    parser.add_argument('--lidar-range', nargs=6, type=float,
                        default=DEFAULT_LIDAR_RANGE)
    parser.add_argument('--voxel-size', nargs=3, type=float,
                        default=DEFAULT_VOXEL_SIZE)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_by_timestamp(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['timestamp']].append(row)
    return OrderedDict(sorted(grouped.items()))


def key(row):
    return (row['timestamp'], row['area_id'], str(row['leader_id']))


def world_to_lidar_xy(world_x, world_y, lidar_pose):
    matrix = x_to_world(lidar_pose)
    inv = np.linalg.inv(matrix)
    point = np.asarray([world_x, world_y, 0.0, 1.0], dtype=np.float64)
    local = np.dot(inv, point)
    return float(local[0]), float(local[1])


def wrap_degrees(value):
    wrapped = (value + 180.0) % 360.0 - 180.0
    return wrapped


def target_bounds(row, reference_pose, args):
    x_min, y_min = float(args.lidar_range[0]), float(args.lidar_range[1])
    voxel_x, voxel_y = float(args.voxel_size[0]), float(args.voxel_size[1])
    center_x = float(row['area_center_x'])
    center_y = float(row['area_center_y'])
    corners = [
        (center_x - args.grid_size_x / 2.0,
         center_y - args.grid_size_y / 2.0),
        (center_x + args.grid_size_x / 2.0,
         center_y - args.grid_size_y / 2.0),
        (center_x - args.grid_size_x / 2.0,
         center_y + args.grid_size_y / 2.0),
        (center_x + args.grid_size_x / 2.0,
         center_y + args.grid_size_y / 2.0),
    ]
    local = [world_to_lidar_xy(x, y, reference_pose) for x, y in corners]
    local_x = [item[0] for item in local]
    local_y = [item[1] for item in local]
    x0 = math.floor((min(local_x) - x_min) / voxel_x)
    x1 = math.ceil((max(local_x) - x_min) / voxel_x)
    y0 = math.floor((min(local_y) - y_min) / voxel_y)
    y1 = math.ceil((max(local_y) - y_min) / voxel_y)
    x0 = max(0, min(args.canvas_width, x0))
    x1 = max(0, min(args.canvas_width, x1))
    y0 = max(0, min(args.canvas_height, y0))
    y1 = max(0, min(args.canvas_height, y1))
    return x0, x1, y0, y1


def resize_nearest(feature, height, width):
    if height <= 0 or width <= 0:
        return feature[:, :, :0, :0]
    src_h, src_w = feature.shape[2], feature.shape[3]
    if src_h == height and src_w == width:
        return feature
    y_idx = np.linspace(0, src_h - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, width).round().astype(np.int64)
    return feature[:, :, y_idx][:, :, :, x_idx]


def to_dtype(array, dtype_name):
    if dtype_name == 'float16':
        return array.astype(np.float16)
    return array.astype(np.float32)


def save_npz(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def shape_string(array):
    return 'x'.join(str(int(item)) for item in array.shape)


def load_frame_poses(dataset, scenario_id, timestamp, cav_ids, reference_id):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=reference_id,
        cav_ids=list(OrderedDict((str(item), None) for item in cav_ids).keys()))
    poses = {}
    for cav_id, cav in frame.items():
        poses[str(cav_id)] = cav['params']['lidar_pose']
    return poses


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = OPV2VFrameDataset(args.dataset_root)
    assignments = {key(row): row for row in read_csv(args.assignment_plan)}
    leader_rows = read_csv(args.leader_feature_manifest)

    frame_manifest = []
    alignment_rows = []
    for timestamp, rows in grouped_by_timestamp(leader_rows).items():
        leader_ids = [str(row['leader_id']) for row in rows]
        cav_ids = [args.reference_cav_id] + leader_ids
        poses = load_frame_poses(
            dataset,
            args.scenario_id,
            timestamp,
            cav_ids,
            args.reference_cav_id)
        if args.reference_cav_id not in poses:
            raise RuntimeError('Reference CAV %s not found for %s' % (
                args.reference_cav_id,
                timestamp))
        reference_pose = poses[args.reference_cav_id]

        accum = np.zeros(
            (1, args.channels, args.canvas_height, args.canvas_width),
            dtype=np.float32)
        counts = np.zeros(
            (1, 1, args.canvas_height, args.canvas_width),
            dtype=np.float32)
        used_rows = 0
        skipped_rows = 0
        yaw_abs = []
        resize_ratios = []

        for row in rows:
            assignment = assignments.get(key(row))
            if assignment is None:
                skipped_rows += 1
                continue
            leader_pose = poses.get(str(row['leader_id']))
            if leader_pose is None:
                skipped_rows += 1
                continue
            path = os.path.join(args.leader_root, row['leader_slice_file'])
            data = np.load(path)
            if args.feature_key not in data:
                skipped_rows += 1
                continue
            feature = data[args.feature_key].astype(np.float32)
            source_bounds = [int(item) for item in data['scatter_bounds_xyxy']]
            target_x0, target_x1, target_y0, target_y1 = target_bounds(
                assignment,
                reference_pose,
                args)
            target_h = target_y1 - target_y0
            target_w = target_x1 - target_x0
            if target_h <= 0 or target_w <= 0:
                skipped_rows += 1
                continue
            resized = resize_nearest(feature, target_h, target_w)
            accum[:, :, target_y0:target_y1, target_x0:target_x1] += resized
            counts[:, :, target_y0:target_y1, target_x0:target_x1] += 1.0
            used_rows += 1

            source_h = max(1, source_bounds[3] - source_bounds[2])
            source_w = max(1, source_bounds[1] - source_bounds[0])
            resize_ratio = (target_h * target_w) / float(source_h * source_w)
            yaw_delta = wrap_degrees(
                float(leader_pose[4]) - float(reference_pose[4]))
            yaw_abs.append(abs(yaw_delta))
            resize_ratios.append(resize_ratio)
            alignment_rows.append(OrderedDict({
                'timestamp': timestamp,
                'area_id': row['area_id'],
                'leader_id': row['leader_id'],
                'reference_cav_id': args.reference_cav_id,
                'source_x0': source_bounds[0],
                'source_x1': source_bounds[1],
                'source_y0': source_bounds[2],
                'source_y1': source_bounds[3],
                'target_x0': target_x0,
                'target_x1': target_x1,
                'target_y0': target_y0,
                'target_y1': target_y1,
                'source_cells': source_h * source_w,
                'target_cells': target_h * target_w,
                'resize_area_ratio': '%.6f' % resize_ratio,
                'leader_yaw': '%.6f' % float(leader_pose[4]),
                'reference_yaw': '%.6f' % float(reference_pose[4]),
                'yaw_delta_deg': '%.6f' % yaw_delta,
                'abs_yaw_delta_deg': '%.6f' % abs(yaw_delta),
            }))

        nonzero_mask = counts > 0
        canvas = np.zeros_like(accum)
        repeated = nonzero_mask.repeat(args.channels, axis=1)
        repeated_counts = counts.repeat(args.channels, axis=1)
        canvas[repeated] = accum[repeated] / repeated_counts[repeated]
        coverage_cells = int(np.count_nonzero(counts[0, 0]))
        overlap_cells = int(np.count_nonzero(counts[0, 0] > 1))
        max_overlap = int(np.max(counts)) if coverage_cells else 0
        rel_path = os.path.join('reference_frames', '%s_%s_ref%s.npz' % (
            timestamp,
            args.feature_key,
            args.reference_cav_id))
        compressed_bytes = save_npz(
            os.path.join(args.output_dir, rel_path),
            {
                'timestamp': np.asarray(timestamp),
                'reference_cav_id': np.asarray(args.reference_cav_id),
                'feature_key': np.asarray(args.feature_key),
                'reference_canvas': to_dtype(canvas, args.dtype),
                'coverage_count': counts.astype(np.uint16),
            })
        frame_manifest.append(OrderedDict({
            'timestamp': timestamp,
            'reference_cav_id': args.reference_cav_id,
            'reference_frame_file': rel_path.replace('\\', '/'),
            'feature_key': args.feature_key,
            'canvas_shape': shape_string(canvas),
            'input_rows': len(rows),
            'used_rows': used_rows,
            'skipped_rows': skipped_rows,
            'coverage_cells': coverage_cells,
            'coverage_ratio': '%.6f' % (
                coverage_cells / float(args.canvas_height * args.canvas_width)),
            'overlap_cells': overlap_cells,
            'max_overlap': max_overlap,
            'mean_abs_yaw_delta_deg': (
                '%.6f' % float(np.mean(yaw_abs)) if yaw_abs else '0.000000'),
            'max_abs_yaw_delta_deg': (
                '%.6f' % float(np.max(yaw_abs)) if yaw_abs else '0.000000'),
            'mean_resize_area_ratio': (
                '%.6f' % float(np.mean(resize_ratios))
                if resize_ratios else '0.000000'),
            'compressed_npz_bytes': compressed_bytes,
        }))

    if alignment_rows:
        write_csv(os.path.join(args.output_dir, 'alignment_manifest.csv'),
                  list(alignment_rows[0].keys()),
                  alignment_rows)
    if frame_manifest:
        write_csv(os.path.join(args.output_dir, 'reference_frame_manifest.csv'),
                  list(frame_manifest[0].keys()),
                  frame_manifest)
    summary = summarize(frame_manifest)
    write_csv(os.path.join(args.output_dir, 'reference_alignment_summary.csv'),
              list(summary.keys()),
              [summary])
    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'leader_root': os.path.abspath(args.leader_root),
        'leader_feature_manifest': os.path.abspath(
            args.leader_feature_manifest),
        'reference_cav_id': args.reference_cav_id,
        'feature_key': args.feature_key,
        'grid_size_x': args.grid_size_x,
        'grid_size_y': args.grid_size_y,
        'lidar_range': [float(item) for item in args.lidar_range],
        'voxel_size': [float(item) for item in args.voxel_size],
        'note': (
            'Reference-frame approximate assembly; nearest resize only, '
            'no rotation/learned feature warp.'),
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Reference-Aligned Feature Assembly\n\n')
        stream.write('This run maps world-coordinate LGCP areas to a shared ')
        stream.write('reference CAV lidar frame and places leader feature ')
        stream.write('slices there using nearest-neighbor resize. It is a ')
        stream.write('coordinate-alignment diagnostic, not valid AP.\n\n')
        stream.write('- frames: `%s`\n' % summary['frames'])
        stream.write('- mean abs yaw delta: `%s`\n' %
                     summary['mean_abs_yaw_delta_deg'])
        stream.write('- mean coverage ratio: `%s`\n' %
                     summary['mean_coverage_ratio'])

    print('Wrote LGCP reference-aligned assembly to %s' % args.output_dir)
    print('frames=%s mean_abs_yaw_delta=%s coverage=%s' % (
        summary['frames'],
        summary['mean_abs_yaw_delta_deg'],
        summary['mean_coverage_ratio']))


def summarize(rows):
    if not rows:
        return OrderedDict({
            'frames': 0,
            'compressed_npz_bytes': 0,
            'mean_coverage_ratio': '0.000000',
            'mean_abs_yaw_delta_deg': '0.000000',
            'max_abs_yaw_delta_deg': '0.000000',
            'mean_resize_area_ratio': '0.000000',
            'max_overlap': 0,
        })
    return OrderedDict({
        'frames': len(rows),
        'compressed_npz_bytes': sum(
            int(row['compressed_npz_bytes']) for row in rows),
        'mean_coverage_ratio': '%.6f' % (
            sum(float(row['coverage_ratio']) for row in rows) / len(rows)),
        'mean_abs_yaw_delta_deg': '%.6f' % (
            sum(float(row['mean_abs_yaw_delta_deg']) for row in rows) /
            len(rows)),
        'max_abs_yaw_delta_deg': '%.6f' % max(
            float(row['max_abs_yaw_delta_deg']) for row in rows),
        'mean_resize_area_ratio': '%.6f' % (
            sum(float(row['mean_resize_area_ratio']) for row in rows) /
            len(rows)),
        'max_overlap': max(int(row['max_overlap']) for row in rows),
    })


if __name__ == '__main__':
    main()
