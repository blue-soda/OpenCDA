# -*- coding: utf-8 -*-
"""
Evaluate LGCP area masks as an external Where2comm BEV-feature selector.

This probe keeps the Where2comm checkpoint's PointPillar backbone, affine
alignment, attentive fusion, and detection heads unchanged. LGCP only replaces
the internal objectness-based communication mask with a planned-area mask in
the RSU/ego reference BEV frame, or intersects the planned-area mask with the
checkpoint's own objectness communication mask.
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
    parse_members,
    update_stats,
    world_points_to_reference,
    write_csv,
)
from opencda.tools.lgcp_v2xvit_area_point_crop_eval import (
    adjust_frame_reference,
    apply_area_crop_to_frame,
    candidate_cavs_from_plan,
    load_model_max_cav,
    make_result_stat,
    normalize_cav_id,
    read_csv,
    selected_timestamps,
)
from opencda.tools.offline_inference import load_coperception_params
from opencood.tools import inference_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='LGCP area-mask Where2comm feature-selector probe.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_where2comm')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--ego-cav-id', default='-1',
                        help='RSU/reference ego id. Defaults to -1.')
    parser.add_argument('--reference-z-override', type=float, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1)
    parser.add_argument('--max-areas-per-frame', type=int, default=5)
    parser.add_argument('--max-cavs', type=int, default=5)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--mask-mode',
                        choices=['lgcp_area', 'lgcp_area_objectness', 'full',
                                 'none'],
                        default='lgcp_area')
    parser.add_argument('--mask-dilation-cells', type=int, default=0)
    parser.add_argument('--no-point-crop', action='store_true',
                        help='Keep full input point clouds and only mask '
                             'Where2comm feature communication.')
    parser.add_argument('--eval-scope', choices=['planned_areas', 'full'],
                        default='planned_areas')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=None)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    parser.add_argument('--feature-value-bits', type=int, default=16)
    parser.add_argument('--frame-rate-hz', type=float, default=10.0)
    return parser.parse_args()


def format_float(value):
    return '%.6f' % float(value)


def load_model_geometry(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    model_args = config['model']['args']
    lidar_range = model_args.get(
        'lidar_range',
        config.get('preprocess', {}).get('cav_lidar_range'))
    voxel_size = model_args.get(
        'voxel_size',
        config.get('preprocess', {}).get('args', {}).get('voxel_size'))
    feature_stride = config.get('postprocess', {}).get(
        'anchor_args', {}).get('feature_stride', 2)
    width = int(round((lidar_range[3] - lidar_range[0]) /
                      (voxel_size[0] * feature_stride)))
    height = int(round((lidar_range[4] - lidar_range[1]) /
                       (voxel_size[1] * feature_stride)))
    return {
        'lidar_range': [float(v) for v in lidar_range],
        'voxel_size': [float(v) for v in voxel_size],
        'feature_stride': int(feature_stride),
        'height': height,
        'width': width,
    }


def points_in_poly(xs, ys, polygon_xy):
    inside = np.zeros(xs.shape, dtype=bool)
    j = polygon_xy.shape[0] - 1
    for i in range(polygon_xy.shape[0]):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        denom = (yj - yi) if abs(yj - yi) > 1e-9 else 1e-9
        crosses = ((yi > ys) != (yj > ys)) & (
            xs < (xj - xi) * (ys - yi) / denom + xi)
        inside ^= crosses
        j = i
    return inside


def dilate_mask(mask, cells):
    if cells <= 0:
        return mask
    result = mask.astype(bool)
    for _ in range(cells):
        padded = np.pad(result, ((1, 1), (1, 1)), mode='constant')
        expanded = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy:dy + result.shape[0],
                                   dx:dx + result.shape[1]]
        result = expanded
    return result


def build_lgcp_area_mask(frame_plan, reference_pose, geometry, grid_size_x,
                         grid_size_y, dilation_cells=0):
    height = geometry['height']
    width = geometry['width']
    x_min, y_min = geometry['lidar_range'][0], geometry['lidar_range'][1]
    x_max, y_max = geometry['lidar_range'][3], geometry['lidar_range'][4]
    xs = x_min + (np.arange(width, dtype=np.float32) + 0.5) * (
        x_max - x_min) / float(width)
    ys = y_min + (np.arange(height, dtype=np.float32) + 0.5) * (
        y_max - y_min) / float(height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    mask = np.zeros((height, width), dtype=bool)
    for row in frame_plan:
        x0, x1, y0, y1 = area_bounds_world(row, grid_size_x, grid_size_y)
        corners_world = np.array([
            [x0, y0, reference_pose[2], 1.0],
            [x1, y0, reference_pose[2], 1.0],
            [x1, y1, reference_pose[2], 1.0],
            [x0, y1, reference_pose[2], 1.0],
        ], dtype=np.float32)
        corners_ref = world_points_to_reference(corners_world, reference_pose)
        mask |= points_in_poly(grid_x, grid_y, corners_ref[:, :2])
    return dilate_mask(mask, dilation_cells).astype(np.float32)


def enable_external_mask_semantics(model, external_mask_mode='replace'):
    touched = 0
    for module in model.modules():
        if hasattr(module, 'external_ego_full'):
            module.external_ego_full = True
            touched += 1
        if hasattr(module, 'external_rate_exclude_ego'):
            module.external_rate_exclude_ego = True
        if hasattr(module, 'external_mask_mode'):
            module.external_mask_mode = external_mask_mode
    return touched


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def estimate_feature_mask_bits(meta, value_bits):
    if not meta:
        return None
    # The default estimate matches Where2comm's non-ego CAV sender semantics.
    # leader_once is a lower-bound proxy for LGCP leader-to-RSU aggregated
    # feature packets over the same selected union cells.
    non_ego_agents = sum(max(int(v) - 1, 0) for v in meta.get(
        'record_len', []))
    leader_once_agents = 1 if non_ego_agents > 0 else 0
    total_bits = 0.0
    leader_once_total_bits = 0.0
    scale_rows = []
    for scale in meta.get('scales', []):
        comm_rate = to_float(scale.get('comm_rate', 0.0))
        height = int(scale.get('height', 0))
        width = int(scale.get('width', 0))
        channels = int(scale.get('payload_channels', 0))
        selected_cells = comm_rate * non_ego_agents * height * width
        leader_once_cells = comm_rate * leader_once_agents * height * width
        bits = selected_cells * channels * value_bits
        leader_once_bits = leader_once_cells * channels * value_bits
        total_bits += bits
        leader_once_total_bits += leader_once_bits
        scale_rows.append(OrderedDict({
            'scale': int(scale.get('scale', len(scale_rows))),
            'comm_rate': format_float(comm_rate),
            'height': height,
            'width': width,
            'payload_channels': channels,
            'selected_cells': format_float(selected_cells),
            'bits_per_frame': format_float(bits),
            'leader_once_cells': format_float(leader_once_cells),
            'leader_once_bits_per_frame': format_float(leader_once_bits),
        }))
    return total_bits, leader_once_total_bits, scale_rows


def run_masked_inference(manager, frame, reference_pose, mask_mode,
                         frame_plan, geometry, args):
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame, reference_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    record_len = int(output_dict['ego']['record_len'][0].item())
    mask_keep_ratio = ''
    if mask_mode != 'none':
        if mask_mode == 'full':
            base_mask = np.ones((geometry['height'], geometry['width']),
                                dtype=np.float32)
        else:
            base_mask = build_lgcp_area_mask(
                frame_plan,
                reference_pose,
                geometry,
                args.grid_size_x,
                args.grid_size_y,
                args.mask_dilation_cells)
        mask_keep_ratio = format_float(base_mask.mean())
        mask = np.repeat(base_mask[None, None, :, :], record_len, axis=0)
        output_dict['ego']['external_comm_mask'] = torch.from_numpy(mask)
    batch_data = manager.to_device(output_dict)
    with torch.no_grad():
        ret = inference_utils.inference_intermediate_fusion(
            batch_data,
            manager.model,
            manager.opencood_dataset,
            return_output=True,
            return_object_ids=False)
    model_output = ret[-1].get('ego', {})
    return ret[0], ret[1], ret[2], model_output, mask_keep_ratio


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
    coperception_params['_dataset_root_override'] = args.dataset_root
    checkpoint_dir = coperception_params['models'][args.fusion_method]
    geometry = load_model_geometry(checkpoint_dir)
    model_max_cav = load_model_max_cav(checkpoint_dir)
    max_cavs = min(args.max_cavs, model_max_cav)
    manager = OpenCOODManager(coperception_params)
    external_mask_mode = (
        'intersection'
        if args.mask_mode == 'lgcp_area_objectness'
        else 'replace')
    where2comm_modules = enable_external_mask_semantics(
        manager.model, external_mask_mode)

    post_processor = manager.opencood_dataset.post_processor
    original_threshold = post_processor.params['target_args'][
        'score_threshold']
    if args.postprocess_score_threshold is not None:
        post_processor.params['target_args']['score_threshold'] = (
            args.postprocess_score_threshold)

    result_stat = make_result_stat()
    frame_rows = []
    cav_rows = []
    scale_rows = []
    try:
        for frame_index, timestamp in enumerate(timestamps, start=1):
            frame_plan = grouped[timestamp]
            if args.max_areas_per_frame:
                frame_plan = frame_plan[:args.max_areas_per_frame]
            cav_ids = candidate_cavs_from_plan(
                frame_plan, args.ego_cav_id, max_cavs)
            frame = dataset.load_frame(
                args.scenario_id,
                timestamp,
                ego_cav_id=args.ego_cav_id,
                cav_ids=cav_ids,
                add_transformation=True)
            ego_key = normalize_cav_id(args.ego_cav_id)
            reference_pose = list(frame[ego_key]['params']['lidar_pose'])
            if args.reference_z_override is not None:
                reference_pose[2] = float(args.reference_z_override)
                adjust_frame_reference(frame, args.ego_cav_id, reference_pose)

            if args.no_point_crop:
                upload_bytes = 0
                ego_area_bytes = 0
            else:
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

            pred_box_tensor, pred_score, gt_box_tensor, model_output, \
                mask_keep_ratio = run_masked_inference(
                    manager,
                    frame,
                    reference_pose,
                    args.mask_mode,
                    frame_plan,
                    geometry,
                    args)

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

            comm_bits = ''
            leader_once_bits = ''
            comm_rate = ''
            comm_meta = model_output.get('comm_mbps_meta')
            if 'comm_rate' in model_output:
                comm_rate = format_float(to_float(model_output['comm_rate']))
            estimated = estimate_feature_mask_bits(
                comm_meta,
                args.feature_value_bits)
            if estimated is not None:
                comm_bits, leader_once_bits, current_scale_rows = estimated
                for row in current_scale_rows:
                    row.update({
                        'scenario_id': args.scenario_id,
                        'timestamp': timestamp,
                        'frame_index': frame_index,
                    })
                    scale_rows.append(row)
                comm_bits = format_float(comm_bits)
                leader_once_bits = format_float(leader_once_bits)

            frame_rows.append(OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'frame_index': frame_index,
                'ego_cav_id': args.ego_cav_id,
                'cav_ids': ';'.join(str(item) for item in cav_ids),
                'mask_mode': args.mask_mode,
                'mask_keep_ratio': mask_keep_ratio,
                'comm_rate': comm_rate,
                'second_hop_feature_bits': comm_bits,
                'second_hop_feature_bytes': (
                    '' if comm_bits == '' else
                    format_float(float(comm_bits) / 8.0)),
                'second_hop_leader_once_bits': leader_once_bits,
                'second_hop_leader_once_bytes': (
                    '' if leader_once_bits == '' else
                    format_float(float(leader_once_bits) / 8.0)),
                'cav_upload_area_bytes': upload_bytes,
                'ego_area_bytes': ego_area_bytes,
                'point_crop_enabled': not args.no_point_crop,
                'planned_areas': len(frame_plan),
                'pred_boxes': (
                    0 if pred_box_tensor is None else
                    int(pred_box_tensor.shape[0])),
                'gt_boxes': (
                    0 if gt_box_tensor is None else
                    int(gt_box_tensor.shape[0])),
            }))
    finally:
        post_processor.params['target_args']['score_threshold'] = (
            original_threshold)

    second_hop_bits = [
        float(row['second_hop_feature_bits'])
        for row in frame_rows
        if row['second_hop_feature_bits'] != ''
    ]
    second_hop_leader_once_bits = [
        float(row['second_hop_leader_once_bits'])
        for row in frame_rows
        if row['second_hop_leader_once_bits'] != ''
    ]
    summary = OrderedDict({
        'frames': len(frame_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': args.fusion_method,
        'checkpoint_dir': os.path.abspath(checkpoint_dir),
        'mask_mode': args.mask_mode,
        'external_mask_mode': external_mask_mode,
        'point_crop_enabled': not args.no_point_crop,
        'where2comm_modules': where2comm_modules,
        'feature_mask_height': geometry['height'],
        'feature_mask_width': geometry['width'],
        'score_threshold': (
            args.postprocess_score_threshold
            if args.postprocess_score_threshold is not None
            else original_threshold),
        'pred_samples': len(result_stat[0.5]['tp']),
        'gt_boxes': result_stat[0.5]['gt'],
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'avg_comm_rate': (
            format_float(np.mean([
                float(row['comm_rate']) for row in frame_rows
                if row['comm_rate'] != ''
            ])) if any(row['comm_rate'] != '' for row in frame_rows) else ''),
        'avg_second_hop_bits_per_frame': (
            format_float(np.mean(second_hop_bits))
            if second_hop_bits else ''),
        'avg_second_hop_mbps': (
            format_float(np.mean(second_hop_bits) * args.frame_rate_hz / 1e6)
            if second_hop_bits else ''),
        'avg_second_hop_leader_once_bits_per_frame': (
            format_float(np.mean(second_hop_leader_once_bits))
            if second_hop_leader_once_bits else ''),
        'avg_second_hop_leader_once_mbps': (
            format_float(
                np.mean(second_hop_leader_once_bits) *
                args.frame_rate_hz / 1e6)
            if second_hop_leader_once_bits else ''),
        'cav_upload_area_bytes_per_frame': (
            format_float(np.mean([
                int(row['cav_upload_area_bytes']) for row in frame_rows
            ])) if frame_rows else ''),
    })
    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'frame_summary.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    if cav_rows:
        write_csv(os.path.join(args.output_dir, 'cav_area_points.csv'),
                  list(cav_rows[0].keys()),
                  cav_rows)
    if scale_rows:
        write_csv(os.path.join(args.output_dir, 'feature_scale_summary.csv'),
                  list(scale_rows[0].keys()),
                  scale_rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'),
              list(summary.keys()),
              [summary])
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)
    print('Wrote LGCP Where2comm area-mask eval to %s' % args.output_dir)
    print('AP@0.3=%s AP@0.5=%s AP@0.7=%s comm_rate=%s mbps=%s '
          'leader_once_mbps=%s' % (
        summary['ap_03'],
        summary['ap_05'],
        summary['ap_07'],
        summary['avg_comm_rate'],
        summary['avg_second_hop_mbps'],
        summary['avg_second_hop_leader_once_mbps']))


if __name__ == '__main__':
    main()
