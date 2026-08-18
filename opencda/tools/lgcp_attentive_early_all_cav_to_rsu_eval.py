# -*- coding: utf-8 -*-
"""Evaluate all CAV point clouds uploaded to an RSU with attentive early weights.

This is a centralized-raw diagnostic for LGCP. Every managed CAV sends its
full raw LiDAR point cloud to the RSU/reference pose, all points are merged
into one RSU point cloud, and the migrated SGCP early-fusion PointPillar
checkpoint runs once as a single-agent detector.
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.lgcp_pointpillar_rsu_bev_fusion import (
    calculate_ap_safe,
    format_float,
    generate_gt,
    load_frame_for_reference,
    local_points_to_world,
    normalize_cav_key,
    postprocess_predictions,
    resolve_reference_pose,
    selected_timestamps,
    update_stats,
    world_points_to_reference,
    write_csv,
)
from opencda.tools.lgcp_where2comm_leader_feature_fusion import (
    preprocess_point_packets,
)
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Upload all CAV point clouds to RSU and evaluate the '
                    'SGCP attentive-derived early PointPillar detector.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='early')
    parser.add_argument('--coperception-yaml', required=True)
    parser.add_argument('--reference-cav-id', default='-1')
    parser.add_argument('--reference-z-override', type=float, default=2.0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=11)
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=0.05)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    return parser.parse_args()


def all_cav_ids(dataset, scenario_id):
    return [
        str(cav_id) for cav_id in dataset.scenarios[scenario_id]['cav_ids']
        if str(cav_id) != '-1'
    ]


def scenario_timestamps(dataset, scenario_id):
    return list(dataset.scenarios[scenario_id]['timestamps'])


def build_point_packets(frame, cav_ids, reference_pose):
    point_packets = []
    total_upload_bytes = 0
    total_upload_points = 0
    per_cav_rows = []
    for cav_id in cav_ids:
        cav = frame[normalize_cav_key(cav_id)]
        local_points = cav['lidar_np'].astype(np.float32)
        world_points = local_points_to_world(
            local_points,
            cav['params']['lidar_pose'])
        ref_points = world_points_to_reference(world_points, reference_pose)
        point_packets.append(ref_points.astype(np.float32))
        point_count = int(local_points.shape[0])
        upload_bytes = int(point_count * local_points.shape[1] * 4)
        total_upload_points += point_count
        total_upload_bytes += upload_bytes
        per_cav_rows.append(OrderedDict({
            'cav_id': str(cav_id),
            'points': point_count,
            'upload_bytes': upload_bytes,
        }))
    return point_packets, per_cav_rows, total_upload_points, total_upload_bytes


def run_early_detector(manager, merged_points):
    processed, valid_indices = preprocess_point_packets(
        manager,
        [merged_points])
    if processed is None:
        return None, valid_indices
    with torch.no_grad():
        output = manager.model({'processed_lidar': processed})
    return output, valid_indices


def build_gt(manager, dataset, scenario_id, timestamp, reference_pose,
             reference_cav_id):
    frame = load_frame_for_reference(
        dataset,
        scenario_id,
        timestamp,
        dataset.scenarios[scenario_id]['cav_ids'],
        reference_pose,
        reference_cav_id)
    reformat = manager.opencood_dataset.get_item_test(frame, reference_pose)
    batch = manager.opencood_dataset.collate_batch_test([reformat])
    return generate_gt(manager, manager.to_device(batch))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = OPV2VFrameDataset(args.dataset_root)
    cav_ids = all_cav_ids(dataset, args.scenario_id)
    timestamps = selected_timestamps(
        OrderedDict((timestamp, []) for timestamp in scenario_timestamps(
            dataset,
            args.scenario_id)),
        args.start_index,
        args.max_frames)

    params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    params['_dataset_root_override'] = args.dataset_root
    manager = OpenCOODManager(params)
    manager.opencood_dataset.max_cav = max(
        manager.opencood_dataset.max_cav,
        len(cav_ids) + 1)

    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }
    frame_rows = []
    cav_rows = []

    for frame_index, timestamp in enumerate(timestamps, start=1):
        reference_pose, _source = resolve_reference_pose(
            dataset,
            args.scenario_id,
            timestamp,
            args.reference_cav_id,
            None)
        if args.reference_z_override is not None:
            reference_pose = list(reference_pose)
            reference_pose[2] = float(args.reference_z_override)
        frame = load_frame_for_reference(
            dataset,
            args.scenario_id,
            timestamp,
            [args.reference_cav_id] + cav_ids,
            reference_pose,
            args.reference_cav_id)
        point_packets, per_cav_rows, upload_points, upload_bytes = (
            build_point_packets(frame, cav_ids, reference_pose))
        merged_points = (
            np.vstack(point_packets).astype(np.float32)
            if point_packets else np.empty((0, 4), dtype=np.float32))
        output, valid_indices = run_early_detector(manager, merged_points)
        gt_box_tensor = build_gt(
            manager,
            dataset,
            args.scenario_id,
            timestamp,
            reference_pose,
            args.reference_cav_id)
        if output is None:
            pred_box_tensor = gt_box_tensor.new_zeros((0, 8, 3))
            pred_score = gt_box_tensor.new_zeros((0,))
        else:
            pred_box_tensor, pred_score = postprocess_predictions(
                manager,
                output['psm'],
                output['rm'],
                args.postprocess_score_threshold)
        update_stats(result_stat, pred_box_tensor, pred_score, gt_box_tensor)

        for row in per_cav_rows:
            out = OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'frame_index': frame_index,
            })
            out.update(row)
            cav_rows.append(out)
        pred_count = 0 if pred_box_tensor is None else int(pred_box_tensor.shape[0])
        gt_count = 0 if gt_box_tensor is None else int(gt_box_tensor.shape[0])
        frame_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'frame_index': frame_index,
            'cav_count': len(cav_ids),
            'valid_packets': len(valid_indices),
            'upload_points': upload_points,
            'upload_bytes': upload_bytes,
            'pred_boxes': pred_count,
            'gt_boxes': gt_count,
        }))
        print('frame=%d/%d timestamp=%s valid=%d pred=%d gt=%d '
              'upload_kb=%.2f' % (
                  frame_index,
                  len(timestamps),
                  timestamp,
                  len(valid_indices),
                  pred_count,
                  gt_count,
                  upload_bytes / 1024.0))

    summary = OrderedDict({
        'frames': len(frame_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': args.fusion_method,
        'checkpoint_yaml': args.coperception_yaml,
        'cav_count': len(cav_ids),
        'valid_packets_mean': format_float(np.mean([
            float(row['valid_packets']) for row in frame_rows
        ])),
        'pred_samples': len(result_stat[0.5]['tp']),
        'gt_boxes': result_stat[0.5]['gt'],
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'upload_bytes_per_frame': format_float(np.mean([
            float(row['upload_bytes']) for row in frame_rows
        ])),
        'score_threshold': args.postprocess_score_threshold,
        'reference_cav_id': args.reference_cav_id,
        'reference_z_override': args.reference_z_override,
    })

    write_csv(os.path.join(args.output_dir, 'frame_summary.csv'),
              list(frame_rows[0].keys()), frame_rows)
    write_csv(os.path.join(args.output_dir, 'cav_uploads.csv'),
              list(cav_rows[0].keys()), cav_rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'),
              list(summary.keys()), [summary])
    print('Wrote all-CAV RSU attentive early evaluation to %s' %
          args.output_dir)
    print('AP@0.3=%s AP@0.5=%s AP@0.7=%s upload_bytes/frame=%s' % (
        summary['ap_03'],
        summary['ap_05'],
        summary['ap_07'],
        summary['upload_bytes_per_frame']))


if __name__ == '__main__':
    main()
