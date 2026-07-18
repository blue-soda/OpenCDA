# -*- coding: utf-8 -*-
"""
Coordinate-aware assembly for LGCP PointPillar leader feature slices.

For each target cell in a shared reference lidar frame, this tool maps the
cell center to world coordinates, then back into the leader lidar frame, and
samples the corresponding leader-local feature slice. It is a nearest-neighbor
feature warp smoke for validating coordinate alignment.
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
        description='Coordinate-warp LGCP feature slices into a reference frame.')
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


def row_key(row):
    return (row['timestamp'], row['area_id'], str(row['leader_id']))


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


def transform_point(matrix, x, y):
    point = np.asarray([x, y, 0.0, 1.0], dtype=np.float64)
    out = np.dot(matrix, point)
    return float(out[0]), float(out[1])


def wrap_degrees(value):
    return (value + 180.0) % 360.0 - 180.0


def target_bounds(row, reference_pose, args):
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
    inv_ref = np.linalg.inv(x_to_world(reference_pose))
    local = [transform_point(inv_ref, x, y) for x, y in corners]
    local_x = [item[0] for item in local]
    local_y = [item[1] for item in local]
    x_min, y_min = float(args.lidar_range[0]), float(args.lidar_range[1])
    voxel_x, voxel_y = float(args.voxel_size[0]), float(args.voxel_size[1])
    x0 = math.floor((min(local_x) - x_min) / voxel_x)
    x1 = math.ceil((max(local_x) - x_min) / voxel_x)
    y0 = math.floor((min(local_y) - y_min) / voxel_y)
    y1 = math.ceil((max(local_y) - y_min) / voxel_y)
    return (
        max(0, min(args.canvas_width, x0)),
        max(0, min(args.canvas_width, x1)),
        max(0, min(args.canvas_height, y0)),
        max(0, min(args.canvas_height, y1)),
    )


def sample_feature_into_canvas(feature, source_bounds, assignment, leader_pose,
                               reference_pose, accum, counts, args):
    x0, x1, y0, y1 = target_bounds(assignment, reference_pose, args)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, (x0, x1, y0, y1)

    x_min, y_min = float(args.lidar_range[0]), float(args.lidar_range[1])
    voxel_x, voxel_y = float(args.voxel_size[0]), float(args.voxel_size[1])
    ref_to_world = x_to_world(reference_pose)
    world_to_leader = np.linalg.inv(x_to_world(leader_pose))
    area_min_x = float(assignment['area_center_x']) - args.grid_size_x / 2.0
    area_max_x = float(assignment['area_center_x']) + args.grid_size_x / 2.0
    area_min_y = float(assignment['area_center_y']) - args.grid_size_y / 2.0
    area_max_y = float(assignment['area_center_y']) + args.grid_size_y / 2.0
    source_x0, _source_x1, source_y0, _source_y1 = source_bounds

    target_cells = 0
    sampled_cells = 0
    for ty in range(y0, y1):
        ref_y = y_min + (ty + 0.5) * voxel_y
        for tx in range(x0, x1):
            ref_x = x_min + (tx + 0.5) * voxel_x
            world_x, world_y = transform_point(ref_to_world, ref_x, ref_y)
            if (world_x < area_min_x or world_x > area_max_x or
                    world_y < area_min_y or world_y > area_max_y):
                continue
            target_cells += 1
            leader_x, leader_y = transform_point(
                world_to_leader,
                world_x,
                world_y)
            leader_ix = int(round((leader_x - x_min) / voxel_x - 0.5))
            leader_iy = int(round((leader_y - y_min) / voxel_y - 0.5))
            src_x = leader_ix - source_x0
            src_y = leader_iy - source_y0
            if (0 <= src_y < feature.shape[2] and
                    0 <= src_x < feature.shape[3]):
                accum[:, :, ty, tx] += feature[:, :, src_y, src_x]
                counts[:, :, ty, tx] += 1.0
                sampled_cells += 1
    return target_cells, sampled_cells, (x0, x1, y0, y1)


def to_dtype(array, dtype_name):
    if dtype_name == 'float16':
        return array.astype(np.float16)
    return array.astype(np.float32)


def shape_string(array):
    return 'x'.join(str(int(item)) for item in array.shape)


def save_npz(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = OPV2VFrameDataset(args.dataset_root)
    assignments = {row_key(row): row for row in read_csv(args.assignment_plan)}
    leader_rows = read_csv(args.leader_feature_manifest)

    frame_rows = []
    warp_rows = []
    for timestamp, rows in grouped_by_timestamp(leader_rows).items():
        leader_ids = [str(row['leader_id']) for row in rows]
        poses = load_frame_poses(
            dataset,
            args.scenario_id,
            timestamp,
            [args.reference_cav_id] + leader_ids,
            args.reference_cav_id)
        reference_pose = poses[args.reference_cav_id]
        accum = np.zeros(
            (1, args.channels, args.canvas_height, args.canvas_width),
            dtype=np.float32)
        counts = np.zeros(
            (1, 1, args.canvas_height, args.canvas_width),
            dtype=np.float32)
        used_rows = 0
        skipped_rows = 0
        target_total = 0
        sampled_total = 0
        yaw_abs = []

        for row in rows:
            assignment = assignments.get(row_key(row))
            leader_pose = poses.get(str(row['leader_id']))
            if assignment is None or leader_pose is None:
                skipped_rows += 1
                continue
            data = np.load(os.path.join(args.leader_root,
                                        row['leader_slice_file']))
            if args.feature_key not in data:
                skipped_rows += 1
                continue
            feature = data[args.feature_key].astype(np.float32)
            source_bounds = [int(item) for item in data['scatter_bounds_xyxy']]
            target_cells, sampled_cells, bounds = sample_feature_into_canvas(
                feature,
                source_bounds,
                assignment,
                leader_pose,
                reference_pose,
                accum,
                counts,
                args)
            if target_cells <= 0:
                skipped_rows += 1
                continue
            used_rows += 1
            target_total += target_cells
            sampled_total += sampled_cells
            yaw_delta = wrap_degrees(
                float(leader_pose[4]) - float(reference_pose[4]))
            yaw_abs.append(abs(yaw_delta))
            warp_rows.append(OrderedDict({
                'timestamp': timestamp,
                'area_id': row['area_id'],
                'leader_id': row['leader_id'],
                'reference_cav_id': args.reference_cav_id,
                'source_bounds_xyxy': ';'.join(str(item)
                                               for item in source_bounds),
                'target_bounds_xyxy': '%d;%d;%d;%d' % bounds,
                'target_cells': target_cells,
                'sampled_cells': sampled_cells,
                'sample_ratio': '%.6f' % (
                    sampled_cells / float(target_cells)
                    if target_cells else 0.0),
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
        rel_path = os.path.join('warped_frames', '%s_%s_ref%s.npz' % (
            timestamp,
            args.feature_key,
            args.reference_cav_id))
        compressed_bytes = save_npz(
            os.path.join(args.output_dir, rel_path),
            {
                'timestamp': np.asarray(timestamp),
                'reference_cav_id': np.asarray(args.reference_cav_id),
                'feature_key': np.asarray(args.feature_key),
                'warped_canvas': to_dtype(canvas, args.dtype),
                'coverage_count': counts.astype(np.uint16),
            })
        frame_rows.append(OrderedDict({
            'timestamp': timestamp,
            'reference_cav_id': args.reference_cav_id,
            'warped_frame_file': rel_path.replace('\\', '/'),
            'feature_key': args.feature_key,
            'canvas_shape': shape_string(canvas),
            'input_rows': len(rows),
            'used_rows': used_rows,
            'skipped_rows': skipped_rows,
            'target_cells': target_total,
            'sampled_cells': sampled_total,
            'sample_ratio': '%.6f' % (
                sampled_total / float(target_total)
                if target_total else 0.0),
            'coverage_cells': coverage_cells,
            'coverage_ratio': '%.6f' % (
                coverage_cells / float(args.canvas_height * args.canvas_width)),
            'overlap_cells': overlap_cells,
            'max_overlap': max_overlap,
            'mean_abs_yaw_delta_deg': (
                '%.6f' % float(np.mean(yaw_abs)) if yaw_abs else '0.000000'),
            'max_abs_yaw_delta_deg': (
                '%.6f' % float(np.max(yaw_abs)) if yaw_abs else '0.000000'),
            'compressed_npz_bytes': compressed_bytes,
        }))

    if warp_rows:
        write_csv(os.path.join(args.output_dir, 'coordinate_warp_manifest.csv'),
                  list(warp_rows[0].keys()),
                  warp_rows)
    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'coordinate_warp_frame_manifest.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    summary = summarize(frame_rows)
    write_csv(os.path.join(args.output_dir, 'coordinate_warp_summary.csv'),
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
        'note': 'Nearest-neighbor coordinate warp; no learned alignment.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Coordinate-Warp Feature Assembly\n\n')
        stream.write('This run samples leader-local feature slices by mapping ')
        stream.write('each reference-frame cell center through world coordinates ')
        stream.write('back to the leader lidar frame.\n\n')
        stream.write('- frames: `%s`\n' % summary['frames'])
        stream.write('- sample ratio: `%s`\n' % summary['mean_sample_ratio'])
        stream.write('- coverage ratio: `%s`\n' %
                     summary['mean_coverage_ratio'])

    print('Wrote LGCP coordinate-warp assembly to %s' % args.output_dir)
    print('frames=%s sample_ratio=%s coverage=%s' % (
        summary['frames'],
        summary['mean_sample_ratio'],
        summary['mean_coverage_ratio']))


def summarize(rows):
    if not rows:
        return OrderedDict({
            'frames': 0,
            'compressed_npz_bytes': 0,
            'mean_sample_ratio': '0.000000',
            'mean_coverage_ratio': '0.000000',
            'mean_abs_yaw_delta_deg': '0.000000',
            'max_abs_yaw_delta_deg': '0.000000',
            'max_overlap': 0,
        })
    return OrderedDict({
        'frames': len(rows),
        'compressed_npz_bytes': sum(
            int(row['compressed_npz_bytes']) for row in rows),
        'mean_sample_ratio': '%.6f' % (
            sum(float(row['sample_ratio']) for row in rows) / len(rows)),
        'mean_coverage_ratio': '%.6f' % (
            sum(float(row['coverage_ratio']) for row in rows) / len(rows)),
        'mean_abs_yaw_delta_deg': '%.6f' % (
            sum(float(row['mean_abs_yaw_delta_deg']) for row in rows) /
            len(rows)),
        'max_abs_yaw_delta_deg': '%.6f' % max(
            float(row['max_abs_yaw_delta_deg']) for row in rows),
        'max_overlap': max(int(row['max_overlap']) for row in rows),
    })


if __name__ == '__main__':
    main()
