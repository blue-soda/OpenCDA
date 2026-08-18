# -*- coding: utf-8 -*-
"""
Evaluate LGCP area point-crop communication with native V2X-ViT inference.

This probe keeps the original OpenCOOD intermediate-fusion semantics:
RSU/reference is the ego slot, other inputs are real CAV agents, and the
checkpoint runs its normal VFE -> backbone -> compressor -> decoder ->
V2XTransformer -> detection heads path. LGCP only constrains the raw point
clouds before inference by cropping every agent to the planned area union.
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
from opencda.tools.lgcp_pointpillar_rsu_bev_fusion import (
    area_bounds_world,
    build_planned_area_bounds,
    calculate_ap_safe,
    filter_boxes_to_planned_areas,
    grouped_by_timestamp,
    local_points_to_world,
    parse_members,
    world_points_to_reference,
    read_csv,
    update_stats,
    write_csv,
)
from opencda.tools.offline_inference import load_coperception_params
from opencood.utils.transformation_utils import x1_to_x2


def parse_args():
    parser = argparse.ArgumentParser(
        description='LGCP point-crop + native V2X-ViT inference probe.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_v2xvit')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--ego-cav-id', default='-1',
                        help='RSU/reference ego id. Defaults to -1.')
    parser.add_argument('--reference-z-override', type=float, default=None,
                        help='Optional ego/reference z used for OpenCOOD '
                             'projection. Useful when a high RSU LiDAR would '
                             'put ground points outside a vehicle-trained '
                             'checkpoint z range.')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1)
    parser.add_argument('--max-areas-per-frame', type=int, default=5)
    parser.add_argument('--max-cavs', type=int, default=5,
                        help='Maximum OpenCOOD agents including ego.')
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--eval-scope', choices=['planned_areas', 'full'],
                        default='planned_areas')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=None)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    return parser.parse_args()


def format_float(value):
    return '%.6f' % float(value)


def normalize_cav_id(value):
    try:
        return int(value)
    except ValueError:
        return str(value)


def unique_strings(values):
    return list(OrderedDict((str(value), None) for value in values).keys())


def cav_sort_key(value):
    try:
        return (0, int(value))
    except ValueError:
        return (1, str(value))


def selected_timestamps(grouped, start_index, max_frames):
    timestamps = list(grouped.keys())[start_index:]
    if max_frames:
        timestamps = timestamps[:max_frames]
    return timestamps


def load_model_max_cav(model_dir):
    config_path = os.path.join(model_dir, 'config.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return int(config['model']['args'].get('max_cav', 5))


def candidate_cavs_from_plan(frame_plan, ego_cav_id, max_cavs):
    candidates = [str(ego_cav_id)]
    for row in frame_plan:
        candidates.append(str(row['leader_id']))
        candidates.extend(parse_members(row['group_members']))
    unique = unique_strings(candidates)
    ego = str(ego_cav_id)
    rest = [item for item in unique if item != ego]
    rest = sorted(rest, key=cav_sort_key)
    return [ego] + rest[:max(0, max_cavs - 1)]


def crop_local_to_area_union(points_local, lidar_pose, area_bounds):
    if points_local is None or points_local.size == 0:
        return np.empty((0, 4), dtype=np.float32), 0
    world_points = local_points_to_world(points_local, lidar_pose)
    keep = np.zeros((points_local.shape[0],), dtype=bool)
    for bounds in area_bounds:
        x0, x1, y0, y1 = bounds
        keep |= (
            (world_points[:, 0] >= x0) &
            (world_points[:, 0] < x1) &
            (world_points[:, 1] >= y0) &
            (world_points[:, 1] < y1))
    cropped = points_local[keep].astype(np.float32)
    return cropped, int(cropped.shape[0])


def apply_area_crop_to_frame(frame, frame_plan, ego_cav_id, grid_size_x,
                             grid_size_y, bytes_per_point):
    area_bounds = [
        area_bounds_world(row, grid_size_x, grid_size_y)
        for row in frame_plan
    ]
    rows = []
    total_upload_bytes = 0
    ego_area_bytes = 0
    for cav_id, cav_content in frame.items():
        original_points = int(cav_content['lidar_np'].shape[0])
        cropped, cropped_points = crop_local_to_area_union(
            cav_content['lidar_np'],
            cav_content['params']['lidar_pose'],
            area_bounds)
        cav_content['lidar_np'] = cropped
        area_bytes = int(cropped_points * bytes_per_point)
        if str(cav_id) == str(ego_cav_id):
            ego_area_bytes += area_bytes
        else:
            total_upload_bytes += area_bytes
        rows.append(OrderedDict({
            'cav_id': cav_id,
            'is_ego': str(cav_id) == str(ego_cav_id),
            'original_points': original_points,
            'area_points': cropped_points,
            'area_bytes': area_bytes,
        }))
    return rows, total_upload_bytes, ego_area_bytes


def adjust_frame_reference(frame, ego_cav_id, reference_pose):
    ego_key = normalize_cav_id(ego_cav_id)
    actual_ego_pose = frame[ego_key]['params']['lidar_pose']
    for cav_id, cav_content in frame.items():
        actual_pose = cav_content['params']['lidar_pose']
        if str(cav_id) == str(ego_cav_id):
            world_points = local_points_to_world(
                cav_content['lidar_np'],
                actual_ego_pose)
            cav_content['lidar_np'] = world_points_to_reference(
                world_points,
                reference_pose)
            cav_content['params']['lidar_pose'] = list(reference_pose)
        cav_content['ego'] = str(cav_id) == str(ego_cav_id)
        cav_content['params']['transformation_matrix'] = x1_to_x2(
            cav_content['params']['lidar_pose'],
            reference_pose)
        cav_content['params']['gt_transformation_matrix'] = (
            cav_content['params']['transformation_matrix'])
        cav_content['params']['spatial_correction_matrix'] = x1_to_x2(
            reference_pose,
            reference_pose)


def make_result_stat():
    return {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }


def run_native_inference(manager, frame, reference_pose):
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        reference_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    batch_data = manager.to_device(output_dict)
    pred_box_tensor, pred_score, gt_box_tensor = manager.inference(
        batch_data,
        with_stats=False)[:3]
    return pred_box_tensor, pred_score, gt_box_tensor, batch_data


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assignment_rows = read_csv(args.assignment_plan)
    grouped = grouped_by_timestamp(assignment_rows)
    timestamps = selected_timestamps(grouped, args.start_index,
                                     args.max_frames)

    dataset = OPV2VFrameDataset(args.dataset_root)
    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    checkpoint_dir = coperception_params['models'][args.fusion_method]
    model_max_cav = load_model_max_cav(checkpoint_dir)
    max_cavs = min(args.max_cavs, model_max_cav)
    manager = OpenCOODManager(coperception_params)
    post_processor = manager.opencood_dataset.post_processor
    original_threshold = post_processor.params['target_args'][
        'score_threshold']
    if args.postprocess_score_threshold is not None:
        post_processor.params['target_args']['score_threshold'] = (
            args.postprocess_score_threshold)

    frame_rows = []
    cav_rows = []
    result_stat = make_result_stat()
    try:
        for frame_index, timestamp in enumerate(timestamps, start=1):
            frame_plan = grouped[timestamp]
            if args.max_areas_per_frame:
                frame_plan = frame_plan[:args.max_areas_per_frame]
            cav_ids = candidate_cavs_from_plan(
                frame_plan,
                args.ego_cav_id,
                max_cavs)
            frame = dataset.load_frame(
                args.scenario_id,
                timestamp,
                ego_cav_id=args.ego_cav_id,
                cav_ids=cav_ids,
                add_transformation=True)
            ego_key = normalize_cav_id(args.ego_cav_id)
            if ego_key not in frame:
                raise ValueError('ego_cav_id %s missing at %s' % (
                    args.ego_cav_id,
                    timestamp))
            reference_pose = list(frame[ego_key]['params']['lidar_pose'])
            if args.reference_z_override is not None:
                reference_pose[2] = float(args.reference_z_override)
                adjust_frame_reference(frame, args.ego_cav_id, reference_pose)
            frame_cav_rows, upload_bytes, ego_area_bytes = (
                apply_area_crop_to_frame(
                    frame,
                    frame_plan,
                    args.ego_cav_id,
                    args.grid_size_x,
                    args.grid_size_y,
                    args.bytes_per_point))
            for row in frame_cav_rows:
                row.update({
                    'scenario_id': args.scenario_id,
                    'timestamp': timestamp,
                })
                cav_rows.append(row)

            pred_box_tensor, pred_score, gt_box_tensor, _batch = (
                run_native_inference(manager, frame, reference_pose))
            if args.eval_scope == 'planned_areas':
                planned_bounds = build_planned_area_bounds(
                    frame_plan,
                    args.grid_size_x,
                    args.grid_size_y)
                pred_box_tensor, pred_score = filter_boxes_to_planned_areas(
                    pred_box_tensor,
                    pred_score,
                    reference_pose,
                    planned_bounds)
                gt_box_tensor, _ = filter_boxes_to_planned_areas(
                    gt_box_tensor,
                    None,
                    reference_pose,
                    planned_bounds)
            update_stats(result_stat, pred_box_tensor, pred_score,
                         gt_box_tensor)
            pred_count = 0 if pred_box_tensor is None else int(
                pred_box_tensor.shape[0])
            score_count = 0 if pred_score is None else int(pred_score.shape[0])
            gt_count = 0 if gt_box_tensor is None else int(
                gt_box_tensor.shape[0])
            frame_rows.append(OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'frame_index': frame_index,
                'ego_cav_id': args.ego_cav_id,
                'cav_ids': ';'.join(str(item) for item in cav_ids),
                'planned_areas': len(frame_plan),
                'eval_scope': args.eval_scope,
                'pred_boxes': pred_count,
                'pred_scores': score_count,
                'gt_boxes': gt_count,
                'cav_upload_area_bytes': upload_bytes,
                'ego_area_bytes': ego_area_bytes,
                'total_area_point_bytes': upload_bytes + ego_area_bytes,
                'model_max_cav': model_max_cav,
                'used_max_cav': max_cavs,
                'reference_z_override': (
                    args.reference_z_override
                    if args.reference_z_override is not None else ''),
            }))
    finally:
        post_processor.params['target_args']['score_threshold'] = (
            original_threshold)

    summary = OrderedDict({
        'frames': len(frame_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': args.fusion_method,
        'ego_cav_id': args.ego_cav_id,
        'eval_scope': args.eval_scope,
        'score_threshold': (
            args.postprocess_score_threshold
            if args.postprocess_score_threshold is not None
            else original_threshold),
        'pred_samples': len(result_stat[0.5]['tp']),
        'gt_boxes': result_stat[0.5]['gt'],
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'cav_upload_area_bytes_total': sum(
            int(row['cav_upload_area_bytes']) for row in frame_rows),
        'cav_upload_area_bytes_per_frame': format_float(
            np.mean([int(row['cav_upload_area_bytes'])
                     for row in frame_rows])) if frame_rows else '',
        'ego_area_bytes_total': sum(
            int(row['ego_area_bytes']) for row in frame_rows),
        'total_area_point_bytes_per_frame': format_float(
            np.mean([int(row['total_area_point_bytes'])
                     for row in frame_rows])) if frame_rows else '',
    })
    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'frame_summary.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    if cav_rows:
        write_csv(os.path.join(args.output_dir, 'cav_area_points.csv'),
                  list(cav_rows[0].keys()),
                  cav_rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'),
              list(summary.keys()),
              [summary])
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)
    print('Wrote LGCP V2X-ViT area point-crop outputs to %s' %
          args.output_dir)
    print('AP@0.3=%s AP@0.5=%s AP@0.7=%s upload_bytes/frame=%s' % (
        summary['ap_03'],
        summary['ap_05'],
        summary['ap_07'],
        summary['cav_upload_area_bytes_per_frame']))


if __name__ == '__main__':
    main()
