# -*- coding: utf-8 -*-
"""
Probe RSU detection from LGCP V2X-ViT compressed feature packets.

This is a smoke tool for the next LGCP model-level route:

area point slices -> V2X-ViT compressed latent packets -> RSU assembly
-> compressor decoder -> V2XTransformer -> detection heads -> AP.
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
    build_area_leader_points,
    build_gt_batch,
    build_planned_area_bounds,
    filter_boxes_to_planned_areas,
    grouped_by_timestamp,
    load_frame_for_reference,
    parse_members,
    postprocess_predictions,
    read_csv,
    resolve_reference_pose,
    selected_timestamps,
    unique_strings,
    update_stats,
    calculate_ap_safe,
    write_csv,
)
from opencda.tools.lgcp_v2xvit_feature_probe import (
    area_crop_indices,
    encode_backbone_features,
    shape_string,
)
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='LGCP V2X-ViT RSU compressed-feature detection probe.')
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
    parser.add_argument('--max-areas-per-frame', type=int, default=5)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--crop-halo-cells', type=int, default=1)
    parser.add_argument('--packet-mode', choices=['crop', 'full'],
                        default='crop')
    parser.add_argument('--query-mode', choices=['mean', 'zero', 'first'],
                        default='mean')
    parser.add_argument('--leader-query-selection',
                        choices=['plan_order', 'max_area_points',
                                 'max_member_upload', 'max_group_size'],
                        default='plan_order',
                        help='Reorder leader packets before fusion. This is '
                             'mainly a diagnostic for query-mode=first.')
    parser.add_argument('--eval-scope', choices=['planned_areas', 'full'],
                        default='planned_areas')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=0.05)
    parser.add_argument('--score-thresholds', default='',
                        help='Comma-separated postprocess score thresholds. '
                             'Overrides --postprocess-score-threshold when '
                             'set.')
    return parser.parse_args()


def load_lidar_range(model_dir):
    with open(os.path.join(model_dir, 'config.yaml'), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return config['model']['args']['lidar_range']


def score_thresholds(args):
    if args.score_thresholds:
        values = []
        for raw_value in args.score_thresholds.split(','):
            raw_value = raw_value.strip()
            if raw_value:
                values.append(float(raw_value))
        if not values:
            raise ValueError('--score-thresholds was set but empty.')
        return values
    return [args.postprocess_score_threshold]


def make_result_stat():
    return {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }


def assemble_latent_packets(args, compressed, valid_packets, reference_pose,
                            lidar_range):
    if args.packet_mode == 'full':
        return compressed

    assembled = torch.zeros_like(compressed)
    feature_h = int(compressed.shape[2])
    feature_w = int(compressed.shape[3])
    crop_cells = 0
    for local_index, (row, _packet) in enumerate(valid_packets):
        x0, x1, y0, y1, cells = area_crop_indices(
            row,
            reference_pose,
            args.grid_size_x,
            args.grid_size_y,
            lidar_range,
            feature_h,
            feature_w,
            args.crop_halo_cells)
        if cells > 0:
            assembled[local_index, :, y0:y1, x0:x1] = (
                compressed[local_index, :, y0:y1, x0:x1])
        crop_cells += cells
    return assembled, crop_cells


def leader_query_score(packet_pair, mode):
    _row, packet = packet_pair
    if mode == 'max_area_points':
        return int(packet['area_points_total'])
    if mode == 'max_member_upload':
        return int(packet['member_upload_bytes'])
    if mode == 'max_group_size':
        return len(packet['members'])
    return 0


def reorder_leader_packets(compressed, valid_packets, mode):
    if mode == 'plan_order':
        return compressed, valid_packets, ''
    order = sorted(
        range(len(valid_packets)),
        key=lambda index: leader_query_score(valid_packets[index], mode),
        reverse=True)
    if not order:
        return compressed, valid_packets, ''
    order_tensor = torch.as_tensor(
        order,
        dtype=torch.long,
        device=compressed.device)
    reordered_packets = [valid_packets[index] for index in order]
    query_row, query_packet = reordered_packets[0]
    query_label = '%s:%s:%s' % (
        mode,
        query_row.get('area_id', ''),
        query_packet.get('leader_id', ''))
    return compressed.index_select(0, order_tensor), reordered_packets, query_label


def make_query_stack(features, query_mode):
    if query_mode == 'first':
        return features
    if query_mode == 'mean':
        query = torch.mean(features, dim=0, keepdim=True)
    else:
        query = torch.zeros_like(features[:1])
    return torch.cat([query, features], dim=0)


def run_v2xvit_rsu(manager, decoded_features, query_mode):
    model = manager.model
    fusion_input = make_query_stack(decoded_features, query_mode)
    leaders = int(fusion_input.shape[0])
    mask = torch.ones((1, leaders), dtype=torch.bool, device=manager.device)
    prior_encoding = torch.zeros((1, leaders, 3), dtype=torch.float32,
                                 device=manager.device)
    # Heterogeneous attention expects types in {0, 1}; mark query as infra.
    if query_mode != 'first':
        prior_encoding[:, 0, 2] = 1.0
    spatial_correction_matrix = torch.eye(
        4, dtype=torch.float32, device=manager.device).view(1, 1, 4, 4)
    spatial_correction_matrix = spatial_correction_matrix.repeat(
        1, leaders, 1, 1)
    prior = prior_encoding.unsqueeze(-1).unsqueeze(-1).repeat(
        1, 1, 1, fusion_input.shape[2], fusion_input.shape[3])
    regroup_feature = torch.cat(
        [fusion_input.view(1, leaders, *fusion_input.shape[1:]), prior],
        dim=2)
    regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
    with torch.no_grad():
        fused_feature = model.fusion_net(
            regroup_feature,
            mask,
            spatial_correction_matrix)
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        psm = model.cls_head(fused_feature)
        rm = model.reg_head(fused_feature)
    return fusion_input, fused_feature, psm, rm


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
    lidar_range = load_lidar_range(
        coperception_params['models'][args.fusion_method])

    thresholds = score_thresholds(args)
    result_stats = {
        threshold: make_result_stat()
        for threshold in thresholds
    }
    frame_rows = []
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

        _scatter, _shrink, compressed, valid_indices = encode_backbone_features(
            manager,
            point_batches)
        if compressed is None:
            continue
        valid_packets = [leader_packets[index] for index in valid_indices]
        compressed, valid_packets, query_leader_label = reorder_leader_packets(
            compressed,
            valid_packets,
            args.leader_query_selection)
        assembled = assemble_latent_packets(
            args,
            compressed,
            valid_packets,
            reference_pose,
            lidar_range)
        if isinstance(assembled, tuple):
            assembled_latent, crop_cells = assembled
        else:
            assembled_latent = assembled
            crop_cells = int(assembled_latent.shape[0] *
                             assembled_latent.shape[2] *
                             assembled_latent.shape[3])
        with torch.no_grad():
            decoded = manager.model.naive_compressor.decoder(assembled_latent)
        fusion_input, fused_feature, psm, rm = run_v2xvit_rsu(
            manager,
            decoded,
            args.query_mode)
        gt_batch = build_gt_batch(
            manager,
            dataset,
            args.scenario_id,
            timestamp,
            reference_pose,
            args.reference_cav_id)
        gt_box_tensor = manager.opencood_dataset.post_processor.generate_gt_bbx(
            gt_batch)
        if args.eval_scope == 'planned_areas':
            planned_bounds = build_planned_area_bounds(
                frame_plan,
                args.grid_size_x,
                args.grid_size_y)
            gt_box_tensor, _ = filter_boxes_to_planned_areas(
                gt_box_tensor,
                None,
                reference_pose,
                planned_bounds)
        primary_pred_box_tensor = None
        primary_pred_score = None
        for threshold_index, threshold in enumerate(thresholds):
            pred_box_tensor, pred_score = postprocess_predictions(
                manager,
                psm,
                rm,
                threshold)
            if args.eval_scope == 'planned_areas':
                pred_box_tensor, pred_score = filter_boxes_to_planned_areas(
                    pred_box_tensor,
                    pred_score,
                    reference_pose,
                    planned_bounds)
            update_stats(
                result_stats[threshold],
                pred_box_tensor,
                pred_score,
                gt_box_tensor)
            if threshold_index == 0:
                primary_pred_box_tensor = pred_box_tensor
                primary_pred_score = pred_score
        pred_count = 0 if primary_pred_box_tensor is None else int(
            primary_pred_box_tensor.shape[0])
        score_count = 0 if primary_pred_score is None else int(
            primary_pred_score.shape[0])
        frame_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'reference': reference_label,
            'packet_mode': args.packet_mode,
            'query_mode': args.query_mode,
            'leader_query_selection': args.leader_query_selection,
            'query_leader_label': query_leader_label,
            'planned_areas': len(frame_plan),
            'valid_leader_features': len(valid_packets),
            'compressed_shape': shape_string(compressed),
            'assembled_latent_shape': shape_string(assembled_latent),
            'decoded_shape': shape_string(decoded),
            'fusion_input_shape': shape_string(fusion_input),
            'fused_feature_shape': shape_string(fused_feature),
            'psm_shape': shape_string(psm),
            'rm_shape': shape_string(rm),
            'crop_cells': crop_cells,
            'pred_boxes': pred_count,
            'pred_scores': score_count,
            'gt_boxes': int(gt_box_tensor.shape[0]),
        }))
        print('frame=%s/%s timestamp=%s mode=%s pred=%s gt=%s' % (
            frame_index,
            len(timestamps),
            timestamp,
            args.packet_mode,
            pred_count,
            int(gt_box_tensor.shape[0])))

    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'v2xvit_rsu_frame_rows.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    summary_rows = []
    for threshold in thresholds:
        result_stat = result_stats[threshold]
        summary_rows.append(OrderedDict({
            'frames': len(frame_rows),
            'scenario_id': args.scenario_id,
            'fusion_method': args.fusion_method,
        'packet_mode': args.packet_mode,
        'query_mode': args.query_mode,
        'leader_query_selection': args.leader_query_selection,
        'eval_scope': args.eval_scope,
            'score_threshold': threshold,
            'pred_samples': len(result_stat[0.5]['tp']),
            'gt_boxes': result_stat[0.5]['gt'],
            'ap_03': calculate_ap_safe(result_stat, 0.3),
            'ap_05': calculate_ap_safe(result_stat, 0.5),
            'ap_07': calculate_ap_safe(result_stat, 0.7),
        }))
    write_csv(os.path.join(args.output_dir, 'v2xvit_rsu_summary.csv'),
              list(summary_rows[0].keys()),
              summary_rows)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)
    print('Wrote LGCP V2X-ViT RSU detection probe outputs to %s' %
          args.output_dir)
    best_row = max(
        summary_rows,
        key=lambda row: float(row['ap_05']) if row['ap_05'] != '' else -1)
    print('best_threshold=%s ap_03=%s ap_05=%s ap_07=%s' % (
        best_row['score_threshold'],
        best_row['ap_03'],
        best_row['ap_05'],
        best_row['ap_07']))


if __name__ == '__main__':
    main()
