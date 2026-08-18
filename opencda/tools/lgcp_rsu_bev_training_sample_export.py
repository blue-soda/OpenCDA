# -*- coding: utf-8 -*-
"""
Export LGCP RSU BEV training samples.

The exporter reuses the current reference-aligned point-slice pipeline:
area point slices -> shared reference frame -> PointPillar scatter BEV.

It stores sparse BEV cells instead of full dense scatter canvases so the
artifacts are usable as training data prototypes without writing hundreds of
MB per frame.
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
    build_gt_batch,
    build_planned_area_bounds,
    encode_scatter_features,
    filter_boxes_to_planned_areas,
    grouped_by_timestamp,
    load_frame_for_reference,
    parse_members,
    read_csv,
    resolve_reference_pose,
    selected_timestamps,
    shape_string,
    unique_strings,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export sparse LGCP RSU BEV training samples.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_attentive')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--reference-cav-id', default='1')
    parser.add_argument('--reference-pose', nargs=6, type=float, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1)
    parser.add_argument('--max-areas-per-frame', type=int, default=0)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--feature-dtype',
                        choices=['float16', 'float32'],
                        default='float16')
    parser.add_argument('--gt-scope',
                        choices=['planned_areas', 'full'],
                        default='planned_areas')
    return parser.parse_args()


def sparse_scatter_payload(leader_features, dtype):
    features_cpu = leader_features.detach().cpu()
    nonzero_mask = torch.any(features_cpu != 0, dim=1)
    indices = torch.nonzero(nonzero_mask, as_tuple=False).numpy()
    if indices.size == 0:
        values = np.empty((0, features_cpu.shape[1]), dtype=np.float16)
    else:
        leader_idx = indices[:, 0]
        y_idx = indices[:, 1]
        x_idx = indices[:, 2]
        values = features_cpu[leader_idx, :, y_idx, x_idx].numpy()
    if dtype == 'float16':
        values = values.astype(np.float16)
    else:
        values = values.astype(np.float32)
    return indices.astype(np.int16), values


def save_sample(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def make_sample_payload(args, timestamp, reference_label, reference_pose,
                        frame_plan, valid_packets, leader_features,
                        gt_box_tensor):
    indices, values = sparse_scatter_payload(
        leader_features,
        args.feature_dtype)
    area_ids = [row['area_id'] for row, _packet in valid_packets]
    leader_ids = [packet['leader_id'] for _row, packet in valid_packets]
    group_sizes = [len(packet['members']) for _row, packet in valid_packets]
    member_upload_bytes = [
        int(packet['member_upload_bytes']) for _row, packet in valid_packets
    ]
    leader_own_bytes = [
        int(packet['leader_own_bytes']) for _row, packet in valid_packets
    ]
    area_points = [
        int(packet['area_points_total']) for _row, packet in valid_packets
    ]
    planned_centers = [
        [float(row.get('area_center_x', 0.0)),
         float(row.get('area_center_y', 0.0))]
        for row in frame_plan
    ]
    return {
        'scenario_id': np.asarray(args.scenario_id),
        'timestamp': np.asarray(timestamp),
        'reference_label': np.asarray(reference_label),
        'reference_pose': np.asarray(reference_pose, dtype=np.float32),
        'dense_shape': np.asarray(leader_features.shape, dtype=np.int32),
        'sparse_indices': indices,
        'sparse_features': values,
        'area_ids': np.asarray(area_ids),
        'leader_ids': np.asarray(leader_ids),
        'group_sizes': np.asarray(group_sizes, dtype=np.int16),
        'member_upload_bytes': np.asarray(member_upload_bytes,
                                          dtype=np.int64),
        'leader_own_bytes': np.asarray(leader_own_bytes, dtype=np.int64),
        'area_points': np.asarray(area_points, dtype=np.int32),
        'planned_area_centers': np.asarray(planned_centers,
                                           dtype=np.float32),
        'gt_boxes': gt_box_tensor.detach().cpu().numpy().astype(np.float32),
        'gt_scope': np.asarray(args.gt_scope),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    sample_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

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

    manifest_rows = []
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

        leader_features, valid_indices = encode_scatter_features(
            manager,
            point_batches)
        if leader_features is None:
            print('frame=%s/%s timestamp=%s skipped_empty' % (
                frame_index,
                len(timestamps),
                timestamp))
            continue

        valid_packets = [leader_packets[index] for index in valid_indices]
        gt_batch = build_gt_batch(
            manager,
            dataset,
            args.scenario_id,
            timestamp,
            reference_pose,
            args.reference_cav_id)
        gt_box_tensor = manager.opencood_dataset.post_processor.generate_gt_bbx(
            gt_batch)
        if args.gt_scope == 'planned_areas':
            planned_bounds = build_planned_area_bounds(
                frame_plan,
                args.grid_size_x,
                args.grid_size_y)
            gt_box_tensor, _ = filter_boxes_to_planned_areas(
                gt_box_tensor,
                None,
                reference_pose,
                planned_bounds)

        sample_rel_path = os.path.join(
            'samples',
            '%s_rsu_bev_sparse.npz' % timestamp)
        sample_path = os.path.join(args.output_dir, sample_rel_path)
        payload = make_sample_payload(
            args,
            timestamp,
            reference_label,
            reference_pose,
            frame_plan,
            valid_packets,
            leader_features,
            gt_box_tensor)
        sample_bytes = save_sample(sample_path, payload)
        sparse_cells = int(payload['sparse_indices'].shape[0])
        manifest_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'sample_file': sample_rel_path.replace('\\', '/'),
            'sample_npz_bytes': sample_bytes,
            'reference': reference_label,
            'reference_x': '%.6f' % float(reference_pose[0]),
            'reference_y': '%.6f' % float(reference_pose[1]),
            'reference_yaw': '%.6f' % float(reference_pose[4]),
            'planned_areas': len(frame_plan),
            'valid_leader_features': int(leader_features.shape[0]),
            'leader_feature_shape': shape_string(leader_features),
            'sparse_cells': sparse_cells,
            'feature_dtype': args.feature_dtype,
            'gt_scope': args.gt_scope,
            'gt_boxes': int(gt_box_tensor.shape[0]),
            'member_upload_bytes': int(np.sum(payload['member_upload_bytes'])),
            'leader_sparse_feature_bytes': int(
                sparse_cells * leader_features.shape[1] *
                (2 if args.feature_dtype == 'float16' else 4)),
        }))
        print('frame=%s/%s timestamp=%s leaders=%s sparse_cells=%s gt=%s' % (
            frame_index,
            len(timestamps),
            timestamp,
            int(leader_features.shape[0]),
            sparse_cells,
            int(gt_box_tensor.shape[0])))

    if manifest_rows:
        write_csv(os.path.join(args.output_dir, 'sample_manifest.csv'),
                  list(manifest_rows[0].keys()),
                  manifest_rows)
    summary = OrderedDict({
        'frames': len(manifest_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': coperception_params['fusion_method'],
        'feature_dtype': args.feature_dtype,
        'gt_scope': args.gt_scope,
        'sample_npz_bytes': sum(
            int(row['sample_npz_bytes']) for row in manifest_rows),
        'member_upload_bytes': sum(
            int(row['member_upload_bytes']) for row in manifest_rows),
        'leader_sparse_feature_bytes': sum(
            int(row['leader_sparse_feature_bytes'])
            for row in manifest_rows),
        'gt_boxes': sum(int(row['gt_boxes']) for row in manifest_rows),
    })
    write_csv(os.path.join(args.output_dir, 'sample_summary.csv'),
              list(summary.keys()),
              [summary])
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump({
            'dataset_root': os.path.abspath(args.dataset_root),
            'scenario_id': args.scenario_id,
            'assignment_plan': os.path.abspath(args.assignment_plan),
            'fusion_method': coperception_params['fusion_method'],
            'coperception_yaml': args.coperception_yaml,
            'reference_cav_id': args.reference_cav_id,
            'reference_pose': args.reference_pose,
            'start_index': args.start_index,
            'max_frames': args.max_frames,
            'max_areas_per_frame': args.max_areas_per_frame,
            'grid_size_x': args.grid_size_x,
            'grid_size_y': args.grid_size_y,
            'feature_dtype': args.feature_dtype,
            'gt_scope': args.gt_scope,
            'timestamps': timestamps,
        }, stream, sort_keys=False)
    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP RSU BEV Training Samples\n\n')
        stream.write('Sparse PointPillar scatter BEV samples exported for ')
        stream.write('future LGCP RSU aggregation head training.\n\n')
        stream.write('- frames: `%s`\n' % summary['frames'])
        stream.write('- sample npz bytes: `%s`\n' %
                     summary['sample_npz_bytes'])
        stream.write('- sparse feature bytes: `%s`\n' %
                     summary['leader_sparse_feature_bytes'])
        stream.write('- gt scope: `%s`\n' % summary['gt_scope'])
    print('Wrote LGCP RSU BEV training samples to %s' % args.output_dir)


if __name__ == '__main__':
    main()
