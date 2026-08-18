# -*- coding: utf-8 -*-
"""
Probe LGCP area-conditioned V2X-ViT compressed feature payloads.

This tool reuses the LGCP area point-slice pipeline, but forwards each leader
packet to the V2X-ViT backbone stage:

area point slices -> VFE -> scatter -> BaseBEVBackbone -> shrink -> compressor

The V2X-ViT checkpoint's NaiveCompressor decodes features before fusion during
normal forward. For communication accounting we count the encoder bottleneck
latent, which is the feature that would be transmitted before decoding/fusion.
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import load_coperception_params
from opencda.tools.lgcp_pointpillar_rsu_bev_fusion import (
    build_area_leader_points,
    grouped_by_timestamp,
    load_frame_for_reference,
    parse_members,
    read_csv,
    resolve_reference_pose,
    selected_timestamps,
    unique_strings,
    write_csv,
)
from opencood.utils.transformation_utils import x_to_world


def parse_args():
    parser = argparse.ArgumentParser(
        description='LGCP V2X-ViT compressed feature byte probe.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_v2xvit')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--reference-cav-id', default='1')
    parser.add_argument('--reference-pose', nargs=6, type=float, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1)
    parser.add_argument('--max-areas-per-frame', type=int, default=0)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--feature-dtype',
                        choices=['float16', 'float32', 'int8'],
                        default='float16')
    parser.add_argument('--crop-halo-cells', type=int, default=1,
                        help='Extra compressed-grid cells around each area.')
    return parser.parse_args()


def shape_string(tensor):
    return 'x'.join(str(int(item)) for item in tensor.shape)


def bytes_per_value(dtype):
    if dtype in ('float16', 'int8'):
        return 1 if dtype == 'int8' else 2
    return 4


def nonzero_cells(feature):
    mask = torch.any(feature.detach().cpu() != 0, dim=1)
    return int(torch.count_nonzero(mask).item())


def encode_backbone_features(manager, point_batches):
    valid = []
    valid_indices = []
    preprocessor = manager.opencood_dataset.pre_processor
    for index, points in enumerate(point_batches):
        if points is None or points.shape[0] == 0:
            continue
        processed = preprocessor.preprocess(points.astype(np.float32))
        if processed['voxel_features'].shape[0] == 0:
            continue
        valid.append(processed)
        valid_indices.append(index)
    if not valid:
        return None, None, None, []

    collated = preprocessor.collate_batch(valid)
    batch_dict = {
        'voxel_features': collated['voxel_features'].to(manager.device),
        'voxel_coords': collated['voxel_coords'].to(manager.device),
        'voxel_num_points': collated['voxel_num_points'].to(manager.device),
        'record_len': torch.tensor([len(valid)],
                                   dtype=torch.int64,
                                   device=manager.device),
    }
    model = manager.model
    with torch.no_grad():
        batch_dict = model.pillar_vfe(batch_dict)
        batch_dict = model.scatter(batch_dict)
        scatter = batch_dict['spatial_features']
        batch_dict = model.backbone(batch_dict)
        backbone = batch_dict['spatial_features_2d']
        shrink = model.shrink_conv(backbone) if model.shrink_flag else backbone
        if model.compression:
            compressed = model.naive_compressor.encoder(shrink)
        else:
            compressed = shrink
    return scatter, shrink, compressed, valid_indices


def world_xy_to_reference_xy(xy_world, reference_pose):
    points = np.asarray(
        [[xy_world[0], xy_world[1], 0.0, 1.0]],
        dtype=np.float32)
    matrix = np.linalg.inv(x_to_world(reference_pose))
    return np.dot(points, matrix.T)[0, :2]


def area_crop_indices(row, reference_pose, grid_size_x, grid_size_y,
                      lidar_range, feature_h, feature_w, halo):
    center_world = (
        float(row['area_center_x']),
        float(row['area_center_y']))
    center_ref = world_xy_to_reference_xy(center_world, reference_pose)
    x_min, y_min = float(lidar_range[0]), float(lidar_range[1])
    x_max, y_max = float(lidar_range[3]), float(lidar_range[4])
    res_x = (x_max - x_min) / float(feature_w)
    res_y = (y_max - y_min) / float(feature_h)

    x0 = int(np.floor((center_ref[0] - grid_size_x / 2.0 - x_min) / res_x))
    x1 = int(np.ceil((center_ref[0] + grid_size_x / 2.0 - x_min) / res_x))
    y0 = int(np.floor((center_ref[1] - grid_size_y / 2.0 - y_min) / res_y))
    y1 = int(np.ceil((center_ref[1] + grid_size_y / 2.0 - y_min) / res_y))

    x0 = max(0, x0 - halo)
    y0 = max(0, y0 - halo)
    x1 = min(feature_w, x1 + halo)
    y1 = min(feature_h, y1 + halo)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0, 0
    return x0, x1, y0, y1, int((x1 - x0) * (y1 - y0))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assignment_rows = read_csv(args.assignment_plan)
    grouped = grouped_by_timestamp(assignment_rows)
    timestamps = selected_timestamps(
        grouped,
        args.start_index,
        args.max_frames)

    dataset = OPV2VFrameDataset(args.dataset_root)
    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    lidar_range = getattr(manager.model, 'lidar_range', None)
    if lidar_range is None:
        # PointPillarTransformer stores lidar range only in config args.
        lidar_range = yaml.load(
            open(os.path.join(
                coperception_params['models'][args.fusion_method],
                'config.yaml')),
            Loader=yaml.Loader)['model']['args']['lidar_range']

    area_rows = []
    frame_rows = []
    value_bytes = bytes_per_value(args.feature_dtype)

    for frame_index, timestamp in enumerate(timestamps, start=1):
        reference_pose, reference_label = resolve_reference_pose(
            dataset,
            args.scenario_id,
            timestamp,
            args.reference_cav_id,
            args.reference_pose)
        frame_plan = grouped[timestamp]
        if args.max_areas_per_frame:
            frame_plan = frame_plan[:args.max_areas_per_frame]

        required_cavs = [args.reference_cav_id]
        for row in frame_plan:
            required_cavs.append(str(row['leader_id']))
            required_cavs.extend(parse_members(row['group_members']))
        frame = load_frame_for_reference(
            dataset,
            args.scenario_id,
            timestamp,
            unique_strings(required_cavs),
            reference_pose,
            args.reference_cav_id)

        leader_packets = []
        point_batches = []
        for row in frame_plan:
            packet = build_area_leader_points(
                frame,
                row,
                reference_pose,
                args.grid_size_x,
                args.grid_size_y)
            leader_packets.append((row, packet))
            point_batches.append(packet['points_reference'])

        scatter, shrink, compressed, valid_indices = encode_backbone_features(
            manager,
            point_batches)
        if compressed is None:
            continue
        valid_packets = [leader_packets[index] for index in valid_indices]
        compressed_h = int(compressed.shape[2])
        compressed_w = int(compressed.shape[3])
        compressed_c = int(compressed.shape[1])
        shrink_c = int(shrink.shape[1])

        frame_member_bytes = 0
        frame_compressed_full = int(compressed.numel() * value_bytes)
        frame_compressed_sparse = 0
        frame_compressed_crop = 0
        frame_crop_cells = 0
        frame_area_points = 0
        frame_scatter_sparse = 0

        for local_index, (row, packet) in enumerate(valid_packets):
            compressed_one = compressed[local_index:local_index + 1]
            scatter_one = scatter[local_index:local_index + 1]
            nz_comp = nonzero_cells(compressed_one)
            nz_scatter = nonzero_cells(scatter_one)
            x0, x1, y0, y1, crop_cells = area_crop_indices(
                row,
                reference_pose,
                args.grid_size_x,
                args.grid_size_y,
                lidar_range,
                compressed_h,
                compressed_w,
                args.crop_halo_cells)
            compressed_sparse_bytes = int(nz_comp * compressed_c * value_bytes)
            compressed_crop_bytes = int(
                crop_cells * compressed_c * value_bytes)
            scatter_sparse_bytes = int(nz_scatter * scatter.shape[1] *
                                       value_bytes)
            frame_member_bytes += int(packet['member_upload_bytes'])
            frame_compressed_sparse += compressed_sparse_bytes
            frame_compressed_crop += compressed_crop_bytes
            frame_crop_cells += crop_cells
            frame_area_points += int(packet['area_points_total'])
            frame_scatter_sparse += scatter_sparse_bytes
            area_rows.append(OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'area_id': row['area_id'],
                'leader_id': packet['leader_id'],
                'group_size': len(packet['members']),
                'area_points_total': int(packet['area_points_total']),
                'member_upload_bytes': int(packet['member_upload_bytes']),
                'scatter_nonzero_cells': nz_scatter,
                'scatter_sparse_bytes_same_dtype': scatter_sparse_bytes,
                'compressed_nonzero_cells': nz_comp,
                'compressed_sparse_bytes': compressed_sparse_bytes,
                'compressed_crop_cells': crop_cells,
                'compressed_crop_bytes': compressed_crop_bytes,
                'crop_x0': x0,
                'crop_x1': x1,
                'crop_y0': y0,
                'crop_y1': y1,
            }))

        frame_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'reference': reference_label,
            'planned_areas': len(frame_plan),
            'valid_leader_features': len(valid_packets),
            'feature_dtype': args.feature_dtype,
            'scatter_shape': shape_string(scatter),
            'shrink_shape': shape_string(shrink),
            'compressed_shape': shape_string(compressed),
            'member_upload_bytes': frame_member_bytes,
            'area_points_total': frame_area_points,
            'scatter_sparse_bytes_same_dtype': frame_scatter_sparse,
            'compressed_full_bytes': frame_compressed_full,
            'compressed_sparse_bytes': frame_compressed_sparse,
            'compressed_crop_bytes': frame_compressed_crop,
            'compressed_crop_cells': frame_crop_cells,
            'compressed_channels': compressed_c,
            'shrink_channels': shrink_c,
            'compressed_h': compressed_h,
            'compressed_w': compressed_w,
        }))
        print('frame=%s/%s timestamp=%s leaders=%s compressed=%s crop_bytes=%s' %
              (frame_index, len(timestamps), timestamp, len(valid_packets),
               shape_string(compressed), frame_compressed_crop))

    if area_rows:
        write_csv(os.path.join(args.output_dir, 'v2xvit_feature_area_rows.csv'),
                  list(area_rows[0].keys()),
                  area_rows)
    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'v2xvit_feature_frame_rows.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    summary = OrderedDict({
        'frames': len(frame_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': args.fusion_method,
        'feature_dtype': args.feature_dtype,
        'crop_halo_cells': args.crop_halo_cells,
        'member_upload_bytes': sum(
            int(row['member_upload_bytes']) for row in frame_rows),
        'scatter_sparse_bytes_same_dtype': sum(
            int(row['scatter_sparse_bytes_same_dtype'])
            for row in frame_rows),
        'compressed_full_bytes': sum(
            int(row['compressed_full_bytes']) for row in frame_rows),
        'compressed_sparse_bytes': sum(
            int(row['compressed_sparse_bytes']) for row in frame_rows),
        'compressed_crop_bytes': sum(
            int(row['compressed_crop_bytes']) for row in frame_rows),
        'area_points_total': sum(
            int(row['area_points_total']) for row in frame_rows),
    })
    write_csv(os.path.join(args.output_dir, 'v2xvit_feature_summary.csv'),
              list(summary.keys()),
              [summary])
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)
    print('Wrote LGCP V2X-ViT feature probe outputs to %s' %
          args.output_dir)


if __name__ == '__main__':
    main()
