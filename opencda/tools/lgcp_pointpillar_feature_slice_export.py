# -*- coding: utf-8 -*-
"""
Export cropped PointPillar intermediate feature slices for LGCP areas.

This is a minimal model-level hierarchy adapter: it runs the OpenCOOD
PointPillar intermediate model, captures scatter and fused BEV tensors, crops
the LGCP area ranges, and writes per-area slice files plus a manifest.
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.lgcp_pointpillar_feature_probe import (
    load_model_geometry,
    map_area_to_feature,
    parse_members,
    read_csv,
    selected_plan_rows,
    shape_string,
    write_csv,
)
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export LGCP PointPillar feature slices.')
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
    parser.add_argument('--slice-level', choices=['scatter', 'fused', 'both'],
                        default='both')
    parser.add_argument('--dtype', choices=['float32', 'float16'],
                        default='float16')
    return parser.parse_args()


def run_forward_capture(manager, dataset, scenario_id, row):
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
        tensor = output.get('spatial_features')
        if tensor is not None:
            captured['scatter'] = tensor.detach().cpu()

    def backbone_hook(_module, _inputs, output):
        tensor = output.get('spatial_features_2d')
        if tensor is not None:
            captured['fused'] = tensor.detach().cpu()

    handles.append(manager.model.scatter.register_forward_hook(scatter_hook))
    handles.append(manager.model.backbone.register_forward_hook(backbone_hook))
    with torch.no_grad():
        manager.model(batch_data['ego'])
    for handle in handles:
        handle.remove()

    return ego_lidar_pose, members, captured


def clipped_range(start, end, size):
    clipped_start = max(0, min(size, int(start)))
    clipped_end = max(0, min(size, int(end)))
    return clipped_start, clipped_end


def crop_tensor(tensor, x_min, x_max, y_min, y_max):
    _, _, height, width = tensor.shape
    x0, x1 = clipped_range(x_min, x_max, width)
    y0, y1 = clipped_range(y_min, y_max, height)
    return tensor[:, :, y0:y1, x0:x1], x0, x1, y0, y1


def tensor_to_numpy(tensor, dtype_name):
    array = tensor.numpy()
    if dtype_name == 'float16':
        return array.astype(np.float16)
    return array.astype(np.float32)


def safe_area_id(area_id):
    return str(area_id).replace('/', '_').replace('\\', '_')


def save_slice(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def shape_or_empty(tensor):
    if tensor is None:
        return ''
    return shape_string(tensor)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    slice_dir = os.path.join(args.output_dir, 'slices')

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

    manifest_rows = []
    for row in assignment_rows:
        lidar_pose, members, captured = run_forward_capture(
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

        scatter_slice = None
        fused_slice = None
        scatter_bounds = ('', '', '', '')
        fused_bounds = ('', '', '', '')
        payload = {
            'timestamp': np.asarray(row['timestamp']),
            'area_id': np.asarray(row['area_id']),
            'leader_id': np.asarray(str(row['leader_id'])),
            'group_members': np.asarray(';'.join(members)),
        }

        if args.slice_level in ('scatter', 'both') and 'scatter' in captured:
            scatter_slice, *scatter_bounds = crop_tensor(
                captured['scatter'],
                mapping['voxel_ix_min'],
                mapping['voxel_ix_max'],
                mapping['voxel_iy_min'],
                mapping['voxel_iy_max'])
            payload['scatter'] = tensor_to_numpy(scatter_slice, args.dtype)
            payload['scatter_bounds_xyxy'] = np.asarray(scatter_bounds)

        if args.slice_level in ('fused', 'both') and 'fused' in captured:
            fused_slice, *fused_bounds = crop_tensor(
                captured['fused'],
                mapping['feature_ix_min'],
                mapping['feature_ix_max'],
                mapping['feature_iy_min'],
                mapping['feature_iy_max'])
            payload['fused'] = tensor_to_numpy(fused_slice, args.dtype)
            payload['fused_bounds_xyxy'] = np.asarray(fused_bounds)

        filename = '%s_area%s_leader%s.npz' % (
            row['timestamp'],
            safe_area_id(row['area_id']),
            row['leader_id'])
        rel_path = os.path.join('slices', filename)
        compressed_bytes = save_slice(
            os.path.join(args.output_dir, rel_path),
            payload)

        scatter_elements = int(scatter_slice.numel()) if scatter_slice is not None else 0
        fused_elements = int(fused_slice.numel()) if fused_slice is not None else 0
        bytes_per_value = 2 if args.dtype == 'float16' else 4
        manifest_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': row['timestamp'],
            'area_id': row['area_id'],
            'leader_id': row['leader_id'],
            'group_members': ';'.join(members),
            'group_size': len(members),
            'slice_file': rel_path.replace('\\', '/'),
            'dtype': args.dtype,
            'scatter_source_shape': shape_or_empty(captured.get('scatter')),
            'fused_source_shape': shape_or_empty(captured.get('fused')),
            'scatter_slice_shape': shape_or_empty(scatter_slice),
            'fused_slice_shape': shape_or_empty(fused_slice),
            'scatter_x0': scatter_bounds[0],
            'scatter_x1': scatter_bounds[1],
            'scatter_y0': scatter_bounds[2],
            'scatter_y1': scatter_bounds[3],
            'fused_x0': fused_bounds[0],
            'fused_x1': fused_bounds[1],
            'fused_y0': fused_bounds[2],
            'fused_y1': fused_bounds[3],
            'scatter_elements': scatter_elements,
            'fused_elements': fused_elements,
            'uncompressed_bytes': (
                scatter_elements + fused_elements) * bytes_per_value,
            'compressed_npz_bytes': compressed_bytes,
            'in_lidar_range': int(mapping['in_lidar_range']),
        }))

    manifest_path = os.path.join(args.output_dir, 'feature_slice_manifest.csv')
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    if fieldnames:
        write_csv(manifest_path, fieldnames, manifest_rows)

    summary = summarize(manifest_rows)
    write_csv(os.path.join(args.output_dir, 'feature_slice_summary.csv'),
              list(summary.keys()),
              [summary])

    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'fusion_method': args.fusion_method,
        'max_frames': args.max_frames,
        'max_areas_per_frame': args.max_areas_per_frame,
        'grid_size_x': args.grid_size_x,
        'grid_size_y': args.grid_size_y,
        'slice_level': args.slice_level,
        'dtype': args.dtype,
        'lidar_range': [float(item) for item in model_geometry['lidar_range']],
        'voxel_size': [float(item) for item in model_geometry['voxel_size']],
        'note': 'Feature crop export smoke; no leader or RSU fusion yet.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP PointPillar Feature Slice Export\n\n')
        stream.write('This run saves cropped PointPillar feature tensors for ')
        stream.write('LGCP area cells. It does not fuse the saved slices.\n\n')
        stream.write('- rows: `%s`\n' % summary['rows'])
        stream.write('- compressed bytes: `%s`\n' %
                     summary['compressed_npz_bytes'])
        stream.write('- uncompressed bytes: `%s`\n' %
                     summary['uncompressed_bytes'])

    print('Wrote LGCP PointPillar feature slices to %s' % args.output_dir)
    print('rows=%s compressed_npz_bytes=%s uncompressed_bytes=%s' % (
        summary['rows'],
        summary['compressed_npz_bytes'],
        summary['uncompressed_bytes']))


def summarize(rows):
    if not rows:
        return OrderedDict({
            'rows': 0,
            'compressed_npz_bytes': 0,
            'uncompressed_bytes': 0,
            'mean_compressed_npz_bytes': 0.0,
            'mean_uncompressed_bytes': 0.0,
            'scatter_elements': 0,
            'fused_elements': 0,
        })
    compressed = sum(int(row['compressed_npz_bytes']) for row in rows)
    uncompressed = sum(int(row['uncompressed_bytes']) for row in rows)
    scatter_elements = sum(int(row['scatter_elements']) for row in rows)
    fused_elements = sum(int(row['fused_elements']) for row in rows)
    return OrderedDict({
        'rows': len(rows),
        'compressed_npz_bytes': compressed,
        'uncompressed_bytes': uncompressed,
        'mean_compressed_npz_bytes': '%.6f' % (compressed / len(rows)),
        'mean_uncompressed_bytes': '%.6f' % (uncompressed / len(rows)),
        'scatter_elements': scatter_elements,
        'fused_elements': fused_elements,
    })


if __name__ == '__main__':
    main()
