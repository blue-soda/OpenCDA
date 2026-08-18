# -*- coding: utf-8 -*-
"""Evaluate all CAV point clouds uploaded to an RSU with Where2comm.

This is an upper-bound diagnostic for LGCP: every managed CAV sends its full
raw LiDAR point cloud to the RSU/reference pose, then the RSU runs the
Where2comm intermediate feature fusion path once.
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
from opencda.tools.lgcp_where2comm_area_mask_eval import (
    enable_external_mask_semantics,
    estimate_feature_mask_bits,
)
from opencda.tools.lgcp_where2comm_leader_feature_fusion import (
    make_query_stack,
    preprocess_point_packets,
)
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Upload all CAV point clouds to RSU and evaluate '
                    'Where2comm intermediate fusion.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_where2comm')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--reference-cav-id', default='-1')
    parser.add_argument('--reference-z-override', type=float, default=2.0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=11)
    parser.add_argument('--query-mode',
                        choices=['mean', 'zero', 'first_leader'],
                        default='mean')
    parser.add_argument('--aggregation-mode',
                        choices=['per_cav_where2comm', 'centralized_raw'],
                        default='per_cav_where2comm',
                        help='per_cav_where2comm keeps one packet per CAV; '
                             'centralized_raw merges all uploaded points into '
                             'one RSU packet before detection.')
    parser.add_argument('--mask-mode',
                        choices=['none', 'full', 'objectness'],
                        default='objectness')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=0.05)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    parser.add_argument('--feature-value-bits', type=int, default=16)
    parser.add_argument('--frame-rate-hz', type=float, default=10.0)
    return parser.parse_args()


def shape_string(tensor):
    if tensor is None:
        return ''
    return 'x'.join(str(int(dim)) for dim in tensor.shape)


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


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


def identity_affine(batch_size, agent_count, device):
    matrix = torch.zeros(
        (batch_size, agent_count, agent_count, 2, 3),
        dtype=torch.float32,
        device=device)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1, 1] = 1.0
    return matrix


def full_masks(agent_count, height, width):
    return torch.ones(
        (agent_count, 1, height, width),
        dtype=torch.float32)


def run_fusion(manager, point_packets, args):
    processed, valid_indices = preprocess_point_packets(manager, point_packets)
    if processed is None:
        return None
    model = manager.model
    with torch.no_grad():
        batch_dict = model.pillar_vfe(processed)
        batch_dict = model.scatter(batch_dict)
        cav_scatter = batch_dict['spatial_features']
        fusion_input, query_count = make_query_stack(cav_scatter,
                                                     args.query_mode)
        record_len = torch.tensor([fusion_input.shape[0]],
                                  dtype=torch.int64,
                                  device=manager.device)
        backbone_dict = {
            'spatial_features': fusion_input,
            'record_len': record_len,
        }
        backbone_dict = model.backbone(backbone_dict)
        spatial_features = backbone_dict['spatial_features']
        spatial_features_2d_single = backbone_dict['spatial_features_2d']
        if model.shrink_flag:
            spatial_features_2d_single = model.shrink_conv(
                spatial_features_2d_single)
        psm_single = model.cls_head(spatial_features_2d_single)
        feature_list = model.backbone.get_multiscale_feature(spatial_features)
        affine = identity_affine(1, int(record_len[0].item()), manager.device)
        fused_feature_list = []
        communication_rates = []
        comm_mbps_scales = []
        for scale_index, fuse_module in enumerate(model.fusion_net):
            feature_i = feature_list[scale_index]
            payload_channels = feature_i.shape[1]
            if model.compression:
                payload_channels = max(
                    1,
                    payload_channels // model.compression_ratio)
                feature_i = model.naive_compressor_list[scale_index](
                    feature_i,
                    use_fp16=False)
            external_mask = None
            if args.mask_mode == 'full':
                external_mask = full_masks(
                    int(record_len[0].item()),
                    int(feature_i.shape[-2]),
                    int(feature_i.shape[-1])).to(manager.device)
            fused_i, comm_rate = fuse_module(
                feature_i,
                psm_single,
                record_len,
                affine,
                None,
                external_comm_mask=external_mask)
            fused_feature_list.append(fused_i)
            communication_rates.append(comm_rate)
            comm_mbps_scales.append(OrderedDict({
                'scale': scale_index,
                'comm_rate': format_float(to_float(comm_rate)),
                'payload_channels': int(payload_channels),
                'height': int(feature_i.shape[-2]),
                'width': int(feature_i.shape[-1]),
            }))

        fused_feature = model.backbone.decode_multiscale_feature(
            fused_feature_list)
        if model.shrink_flag:
            fused_feature = model.shrink_conv(fused_feature)
        psm = model.cls_head(fused_feature)
        rm = model.reg_head(fused_feature)
        comm_rate = (
            torch.stack([
                rate if torch.is_tensor(rate)
                else fused_feature.new_tensor(float(rate))
                for rate in communication_rates
            ]).mean() if communication_rates
            else fused_feature.new_tensor(0.0))
    return {
        'psm': psm,
        'rm': rm,
        'valid_indices': valid_indices,
        'cav_scatter': cav_scatter,
        'fusion_input': fusion_input,
        'record_len': record_len,
        'comm_rate': comm_rate,
        'comm_mbps_meta': {
            'record_len': record_len.detach().cpu().tolist(),
            'compression_ratio': int(model.compression_ratio),
            'scales': comm_mbps_scales,
        },
        'query_count': query_count,
    }


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


def update_scale_rows(scale_rows, fusion, scenario_id, timestamp, frame_index,
                      feature_value_bits, frame_rate_hz):
    if fusion is None:
        return
    estimated = estimate_feature_mask_bits(
        fusion['comm_mbps_meta'],
        feature_value_bits)
    if estimated is None:
        return
    _comm_bits, _leader_once_bits, current_scale_rows = estimated
    for row in current_scale_rows:
        row.update({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'frame_index': frame_index,
        })
        row['mbps'] = format_float(
            float(row['bits_per_frame']) * frame_rate_hz / 1e6)
        scale_rows.append(row)


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
    external_mode = 'replace' if args.mask_mode == 'full' else 'intersection'
    enable_external_mask_semantics(manager.model, external_mode)

    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }
    frame_rows = []
    cav_rows = []
    scale_rows = []

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
        if args.aggregation_mode == 'centralized_raw':
            merged_points = (
                np.vstack(point_packets).astype(np.float32)
                if point_packets else np.empty((0, 4), dtype=np.float32))
            point_packets = [merged_points]
        fusion = run_fusion(manager, point_packets, args)
        gt_box_tensor = build_gt(
            manager,
            dataset,
            args.scenario_id,
            timestamp,
            reference_pose,
            args.reference_cav_id)
        if fusion is None:
            pred_box_tensor = gt_box_tensor.new_zeros((0, 8, 3))
            pred_score = gt_box_tensor.new_zeros((0,))
        else:
            pred_box_tensor, pred_score = postprocess_predictions(
                manager,
                fusion['psm'],
                fusion['rm'],
                args.postprocess_score_threshold)
        update_stats(result_stat, pred_box_tensor, pred_score, gt_box_tensor)

        update_scale_rows(
            scale_rows,
            fusion,
            args.scenario_id,
            timestamp,
            frame_index,
            args.feature_value_bits,
            args.frame_rate_hz)
        second_hop_bits = sum(
            float(row['bits_per_frame']) for row in scale_rows
            if row['timestamp'] == timestamp)
        for row in per_cav_rows:
            out = OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'frame_index': frame_index,
            })
            out.update(row)
            cav_rows.append(out)
        valid_count = 0 if fusion is None else len(fusion['valid_indices'])
        pred_count = 0 if pred_box_tensor is None else int(pred_box_tensor.shape[0])
        gt_count = 0 if gt_box_tensor is None else int(gt_box_tensor.shape[0])
        frame_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'frame_index': frame_index,
            'cav_count': len(cav_ids),
            'valid_cav_packets': valid_count,
            'query_mode': args.query_mode,
            'aggregation_mode': args.aggregation_mode,
            'mask_mode': args.mask_mode,
            'upload_points': upload_points,
            'upload_bytes': upload_bytes,
            'second_hop_feature_bits': format_float(second_hop_bits),
            'second_hop_feature_mbps': format_float(
                second_hop_bits * args.frame_rate_hz / 1e6),
            'pred_boxes': pred_count,
            'gt_boxes': gt_count,
            'cav_scatter_shape': '' if fusion is None else shape_string(
                fusion['cav_scatter']),
            'fusion_input_shape': '' if fusion is None else shape_string(
                fusion['fusion_input']),
        }))
        print('frame=%d/%d timestamp=%s valid=%d pred=%d gt=%d '
              'upload_kb=%.2f second_mbps=%.6f' % (
                  frame_index,
                  len(timestamps),
                  timestamp,
                  valid_count,
                  pred_count,
                  gt_count,
                  upload_bytes / 1024.0,
                  second_hop_bits * args.frame_rate_hz / 1e6))

    summary = OrderedDict({
        'frames': len(frame_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': args.fusion_method,
        'query_mode': args.query_mode,
        'aggregation_mode': args.aggregation_mode,
        'mask_mode': args.mask_mode,
        'cav_count': len(cav_ids),
        'valid_cav_packets_mean': format_float(np.mean([
            float(row['valid_cav_packets']) for row in frame_rows
        ])),
        'pred_samples': len(result_stat[0.5]['tp']),
        'gt_boxes': result_stat[0.5]['gt'],
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'upload_bytes_per_frame': format_float(np.mean([
            float(row['upload_bytes']) for row in frame_rows
        ])),
        'second_hop_mbps': format_float(np.mean([
            float(row['second_hop_feature_mbps']) for row in frame_rows
        ])),
        'score_threshold': args.postprocess_score_threshold,
        'reference_cav_id': args.reference_cav_id,
        'reference_z_override': args.reference_z_override,
    })

    write_csv(os.path.join(args.output_dir, 'frame_summary.csv'),
              list(frame_rows[0].keys()), frame_rows)
    write_csv(os.path.join(args.output_dir, 'cav_uploads.csv'),
              list(cav_rows[0].keys()), cav_rows)
    write_csv(os.path.join(args.output_dir, 'feature_scale_summary.csv'),
              list(scale_rows[0].keys()), scale_rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'),
              list(summary.keys()), [summary])
    print('Wrote all-CAV RSU Where2comm evaluation to %s' % args.output_dir)
    print('AP@0.3=%s AP@0.5=%s AP@0.7=%s upload_bytes/frame=%s '
          'second_hop_mbps=%s' % (
              summary['ap_03'],
              summary['ap_05'],
              summary['ap_07'],
              summary['upload_bytes_per_frame'],
              summary['second_hop_mbps']))


if __name__ == '__main__':
    main()
