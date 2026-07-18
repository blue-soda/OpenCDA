# -*- coding: utf-8 -*-
"""
Probe PointPillar intermediate feature tensors for LGCP area slicing.

This tool records the tensor shapes produced by the OpenCOOD PointPillar
intermediate model and maps LGCP world-coordinate area cells to leader-local
BEV feature map ranges. It does not crop or modify model features yet.
"""

import argparse
import csv
import math
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import load_coperception_params
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.utils.transformation_utils import x_to_world


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe LGCP PointPillar intermediate feature geometry.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_attentive')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--max-frames', type=int, default=1)
    parser.add_argument('--max-areas-per-frame', type=int, default=5)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_members(value):
    return [item for item in str(value).split(';') if item]


def selected_plan_rows(rows, max_frames, max_areas_per_frame):
    timestamps = sorted({row['timestamp'] for row in rows})
    if max_frames:
        timestamps = timestamps[:max_frames]
    selected = []
    for timestamp in timestamps:
        frame_rows = [row for row in rows if row['timestamp'] == timestamp]
        if max_areas_per_frame:
            frame_rows = frame_rows[:max_areas_per_frame]
        selected.extend(frame_rows)
    return selected


def run_intermediate_inference(manager, dataset, scenario_id, row):
    leader_id = str(row['leader_id'])
    members = parse_members(row['group_members'])
    if leader_id not in members:
        members = [leader_id] + members
    members = list(OrderedDict((str(member), None) for member in members).keys())

    frame = dataset.load_frame(
        scenario_id,
        row['timestamp'],
        ego_cav_id=leader_id,
        cav_ids=members)
    ego = next(cav for cav in frame.values() if cav['ego'])
    ego_lidar_pose = ego['params']['lidar_pose']

    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        ego_lidar_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    batch_data = manager.to_device(output_dict)

    captured = {}
    handles = []

    def scatter_hook(_module, _inputs, output):
        captured['scatter_spatial_features_shape'] = shape_string(
            output.get('spatial_features'))

    def backbone_hook(_module, _inputs, output):
        captured['backbone_spatial_features_2d_shape'] = shape_string(
            output.get('spatial_features_2d'))

    model = manager.model
    handles.append(model.scatter.register_forward_hook(scatter_hook))
    handles.append(model.backbone.register_forward_hook(backbone_hook))
    with torch.no_grad():
        model(batch_data['ego'])
    for handle in handles:
        handle.remove()

    return ego_lidar_pose, members, captured


def shape_string(tensor):
    if tensor is None:
        return ''
    return 'x'.join(str(int(item)) for item in tensor.shape)


def world_to_lidar_xy(world_x, world_y, lidar_pose):
    matrix = x_to_world(lidar_pose)
    inv = np.linalg.inv(matrix)
    point = np.asarray([world_x, world_y, 0.0, 1.0], dtype=np.float64)
    local = np.dot(inv, point)
    return float(local[0]), float(local[1])


def map_area_to_feature(row, lidar_pose, model_args, grid_x, grid_y):
    lidar_range = model_args['lidar_range']
    voxel_size = model_args['voxel_size']
    x_min, y_min = float(lidar_range[0]), float(lidar_range[1])
    x_max, y_max = float(lidar_range[3]), float(lidar_range[4])
    voxel_x, voxel_y = float(voxel_size[0]), float(voxel_size[1])
    feature_stride = 2
    scatter_w = int(round((x_max - x_min) / voxel_x))
    scatter_h = int(round((y_max - y_min) / voxel_y))
    fused_w = int(math.ceil(scatter_w / feature_stride))
    fused_h = int(math.ceil(scatter_h / feature_stride))

    center_x = float(row['area_center_x'])
    center_y = float(row['area_center_y'])
    corners = [
        (center_x - grid_x / 2.0, center_y - grid_y / 2.0),
        (center_x + grid_x / 2.0, center_y - grid_y / 2.0),
        (center_x - grid_x / 2.0, center_y + grid_y / 2.0),
        (center_x + grid_x / 2.0, center_y + grid_y / 2.0),
    ]
    local_corners = [world_to_lidar_xy(x, y, lidar_pose) for x, y in corners]
    local_center = world_to_lidar_xy(center_x, center_y, lidar_pose)

    local_x = [item[0] for item in local_corners]
    local_y = [item[1] for item in local_corners]
    voxel_ix_min = math.floor((min(local_x) - x_min) / voxel_x)
    voxel_ix_max = math.ceil((max(local_x) - x_min) / voxel_x)
    voxel_iy_min = math.floor((min(local_y) - y_min) / voxel_y)
    voxel_iy_max = math.ceil((max(local_y) - y_min) / voxel_y)

    feat_ix_min = math.floor(voxel_ix_min / feature_stride)
    feat_ix_max = math.ceil(voxel_ix_max / feature_stride)
    feat_iy_min = math.floor(voxel_iy_min / feature_stride)
    feat_iy_max = math.ceil(voxel_iy_max / feature_stride)

    in_range = (
        max(local_x) >= x_min and min(local_x) < x_max and
        max(local_y) >= y_min and min(local_y) < y_max)
    scatter_cells = clipped_area(
        voxel_ix_min,
        voxel_ix_max,
        voxel_iy_min,
        voxel_iy_max,
        scatter_w,
        scatter_h)
    fused_cells = clipped_area(
        feat_ix_min,
        feat_ix_max,
        feat_iy_min,
        feat_iy_max,
        fused_w,
        fused_h)

    return {
        'local_center_x': local_center[0],
        'local_center_y': local_center[1],
        'voxel_ix_min': voxel_ix_min,
        'voxel_ix_max': voxel_ix_max,
        'voxel_iy_min': voxel_iy_min,
        'voxel_iy_max': voxel_iy_max,
        'feature_ix_min': feat_ix_min,
        'feature_ix_max': feat_ix_max,
        'feature_iy_min': feat_iy_min,
        'feature_iy_max': feat_iy_max,
        'in_lidar_range': in_range,
        'scatter_grid_w': scatter_w,
        'scatter_grid_h': scatter_h,
        'fused_grid_w': fused_w,
        'fused_grid_h': fused_h,
        'scatter_slice_cells': scatter_cells,
        'fused_slice_cells': fused_cells,
        'scatter_slice_float32_bytes_per_cav': scatter_cells * 64 * 4,
        'fused_slice_float32_bytes': fused_cells * 384 * 4,
    }


def clipped_area(x_min, x_max, y_min, y_max, width, height):
    clipped_x_min = max(0, min(width, x_min))
    clipped_x_max = max(0, min(width, x_max))
    clipped_y_min = max(0, min(height, y_min))
    clipped_y_max = max(0, min(height, y_max))
    return max(0, clipped_x_max - clipped_x_min) * max(
        0,
        clipped_y_max - clipped_y_min)


def load_model_geometry(manager):
    hypes = yaml_utils.load_yaml(None, manager.opt)
    model_args = hypes['model']['args']
    return {
        'lidar_range': model_args['lidar_range'],
        'voxel_size': model_args['voxel_size'],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assignment_rows = selected_plan_rows(
        read_csv(args.assignment_plan),
        args.max_frames,
        args.max_areas_per_frame)
    dataset = OPV2VFrameDataset(args.dataset_root)
    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    model_geometry = load_model_geometry(manager)

    rows = []
    for row in assignment_rows:
        lidar_pose, members, captured = run_intermediate_inference(
            manager,
            dataset,
            args.scenario_id,
            row)
        mapping = map_area_to_feature(
            row,
            lidar_pose,
            model_geometry,
            args.grid_size_x,
            args.grid_size_y)
        rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': row['timestamp'],
            'area_id': row['area_id'],
            'leader_id': row['leader_id'],
            'group_members': ';'.join(members),
            'group_size': len(members),
            'area_center_x': row['area_center_x'],
            'area_center_y': row['area_center_y'],
            'leader_lidar_x': '%.6f' % float(lidar_pose[0]),
            'leader_lidar_y': '%.6f' % float(lidar_pose[1]),
            'leader_lidar_yaw': '%.6f' % float(lidar_pose[4]),
            'local_center_x': '%.6f' % mapping['local_center_x'],
            'local_center_y': '%.6f' % mapping['local_center_y'],
            'voxel_ix_min': mapping['voxel_ix_min'],
            'voxel_ix_max': mapping['voxel_ix_max'],
            'voxel_iy_min': mapping['voxel_iy_min'],
            'voxel_iy_max': mapping['voxel_iy_max'],
            'feature_ix_min': mapping['feature_ix_min'],
            'feature_ix_max': mapping['feature_ix_max'],
            'feature_iy_min': mapping['feature_iy_min'],
            'feature_iy_max': mapping['feature_iy_max'],
            'in_lidar_range': int(mapping['in_lidar_range']),
            'scatter_grid_w': mapping['scatter_grid_w'],
            'scatter_grid_h': mapping['scatter_grid_h'],
            'fused_grid_w': mapping['fused_grid_w'],
            'fused_grid_h': mapping['fused_grid_h'],
            'scatter_slice_cells': mapping['scatter_slice_cells'],
            'fused_slice_cells': mapping['fused_slice_cells'],
            'scatter_slice_float32_bytes_per_cav':
                mapping['scatter_slice_float32_bytes_per_cav'],
            'scatter_slice_float32_bytes_for_group':
                mapping['scatter_slice_float32_bytes_per_cav'] * len(members),
            'fused_slice_float32_bytes':
                mapping['fused_slice_float32_bytes'],
            'scatter_spatial_features_shape': captured.get(
                'scatter_spatial_features_shape', ''),
            'backbone_spatial_features_2d_shape': captured.get(
                'backbone_spatial_features_2d_shape', ''),
        }))

    write_csv(os.path.join(args.output_dir, 'pointpillar_feature_probe.csv'),
              ['scenario_id', 'timestamp', 'area_id', 'leader_id',
               'group_members', 'group_size', 'area_center_x',
               'area_center_y', 'leader_lidar_x', 'leader_lidar_y',
               'leader_lidar_yaw', 'local_center_x', 'local_center_y',
               'voxel_ix_min', 'voxel_ix_max', 'voxel_iy_min',
               'voxel_iy_max', 'feature_ix_min', 'feature_ix_max',
               'feature_iy_min', 'feature_iy_max', 'in_lidar_range',
               'scatter_grid_w', 'scatter_grid_h', 'fused_grid_w',
               'fused_grid_h', 'scatter_slice_cells', 'fused_slice_cells',
               'scatter_slice_float32_bytes_per_cav',
               'scatter_slice_float32_bytes_for_group',
               'fused_slice_float32_bytes',
               'scatter_spatial_features_shape',
               'backbone_spatial_features_2d_shape'],
              rows)

    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'fusion_method': args.fusion_method,
        'max_frames': args.max_frames,
        'max_areas_per_frame': args.max_areas_per_frame,
        'grid_size_x': args.grid_size_x,
        'grid_size_y': args.grid_size_y,
        'lidar_range': [float(item) for item in model_geometry['lidar_range']],
        'voxel_size': [float(item) for item in model_geometry['voxel_size']],
        'feature_stride_assumption': 2,
        'scatter_slice_channels': 64,
        'fused_slice_channels': 384,
        'dtype_for_byte_estimate': 'float32',
        'note': 'Probe only; no neural feature crop is applied.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP PointPillar Feature Probe\n\n')
        stream.write('This probe records PointPillar intermediate tensor ')
        stream.write('shapes and maps LGCP world area cells to leader-local ')
        stream.write('BEV feature index ranges. It does not crop features.\n\n')
        stream.write('- rows: `%d`\n' % len(rows))
        if rows:
            stream.write('- scatter shape: `%s`\n' %
                         rows[0]['scatter_spatial_features_shape'])
            stream.write('- fused feature shape: `%s`\n' %
                         rows[0]['backbone_spatial_features_2d_shape'])

    print('Wrote LGCP PointPillar feature probe to %s' % args.output_dir)
    if rows:
        print('rows=%d scatter=%s fused=%s' % (
            len(rows),
            rows[0]['scatter_spatial_features_shape'],
            rows[0]['backbone_spatial_features_2d_shape']))


if __name__ == '__main__':
    main()
