# -*- coding: utf-8 -*-
"""
Run LGCP leader-packet feature fusion with the Where2comm checkpoint.

Pipeline:
1. Each area-task leader collects member point-cloud slices inside that area.
2. The merged area points are encoded as one leader feature packet.
3. RSU fuses all leader packets at Where2comm multiscale feature level.
4. Detection heads run once on the RSU fused feature.

This tool intentionally stays separate from SGCP code paths.
"""

import argparse
import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.lgcp_pointpillar_rsu_bev_fusion import (
    build_area_leader_points,
    build_planned_area_bounds,
    calculate_ap_safe,
    filter_boxes_to_planned_areas,
    format_float,
    generate_gt,
    grouped_by_timestamp,
    load_frame_for_reference,
    postprocess_predictions,
    read_csv,
    resolve_reference_pose,
    selected_timestamps,
    shape_string,
    update_stats,
    write_csv,
)
from opencda.tools.lgcp_v2xvit_area_point_crop_eval import (
    candidate_cavs_from_plan,
)
from opencda.tools.lgcp_where2comm_area_mask_eval import (
    build_lgcp_area_mask,
    enable_external_mask_semantics,
    estimate_feature_mask_bits,
    load_model_geometry,
)
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='LGCP leader feature-packet RSU fusion with Where2comm.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_where2comm')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--reference-cav-id', default='-1')
    parser.add_argument('--reference-z-override', type=float, default=2.0)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1)
    parser.add_argument('--max-areas-per-frame', type=int, default=5)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--query-mode',
                        choices=['mean', 'zero', 'first_leader'],
                        default='mean')
    parser.add_argument('--packet-granularity',
                        choices=['area', 'leader'],
                        default='area',
                        help='area sends one feature packet per area-task; '
                             'leader merges all areas assigned to the same '
                             'leader into one RSU feature packet.')
    parser.add_argument('--mask-mode',
                        choices=['none', 'lgcp_area',
                                 'lgcp_area_objectness', 'full'],
                        default='lgcp_area_objectness')
    parser.add_argument('--mask-dilation-cells', type=int, default=1)
    parser.add_argument('--eval-scope', choices=['planned_areas', 'full'],
                        default='planned_areas')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=0.05)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    parser.add_argument('--feature-value-bits', type=int, default=16)
    parser.add_argument('--frame-rate-hz', type=float, default=10.0)
    return parser.parse_args()


def make_query_stack(spatial_features, query_mode):
    if query_mode == 'first_leader':
        return spatial_features, 0
    if query_mode == 'mean':
        query = torch.mean(spatial_features, dim=0, keepdim=True)
    elif query_mode == 'zero':
        query = torch.zeros_like(spatial_features[:1])
    else:
        raise ValueError('Unknown query_mode: %s' % query_mode)
    return torch.cat([query, spatial_features], dim=0), 1


def preprocess_point_packets(manager, point_packets):
    valid = []
    valid_indices = []
    preprocessor = manager.opencood_dataset.pre_processor
    for index, points in enumerate(point_packets):
        if points is None or points.shape[0] == 0:
            continue
        processed = preprocessor.preprocess(points.astype(np.float32))
        if processed['voxel_features'].shape[0] == 0:
            continue
        valid.append(processed)
        valid_indices.append(index)
    if not valid:
        return None, []
    collated = preprocessor.collate_batch(valid)
    return {
        'voxel_features': collated['voxel_features'].to(manager.device),
        'voxel_coords': collated['voxel_coords'].to(manager.device),
        'voxel_num_points': collated['voxel_num_points'].to(manager.device),
    }, valid_indices


def identity_affine(batch_size, agent_count, device):
    matrix = torch.zeros(
        (batch_size, agent_count, agent_count, 2, 3),
        dtype=torch.float32,
        device=device)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1, 1] = 1.0
    return matrix


def prepend_query_mask(mask, query_count):
    if query_count <= 0 or mask is None:
        return mask
    query_mask = torch.ones_like(mask[:query_count])
    return torch.cat([query_mask, mask], dim=0)


def build_packet_masks(packet_rows, reference_pose, geometry, args):
    if args.mask_mode == 'none':
        return None, ''
    if args.mask_mode == 'full':
        base = np.ones((len(packet_rows), 1, geometry['height'],
                        geometry['width']), dtype=np.float32)
        return torch.from_numpy(base), '1.000000'

    masks = []
    for row in packet_rows:
        mask_rows = row.get('_mask_rows', [row])
        mask = build_lgcp_area_mask(
            mask_rows,
            reference_pose,
            geometry,
            args.grid_size_x,
            args.grid_size_y,
            args.mask_dilation_cells)
        masks.append(mask[None, :, :])
    if not masks:
        return None, ''
    stacked = np.stack(masks, axis=0).astype(np.float32)
    return torch.from_numpy(stacked), format_float(stacked.mean())


def merge_leader_area_packets(packet_meta):
    grouped = OrderedDict()
    for row, packet in packet_meta:
        leader_id = str(row['leader_id'])
        if leader_id not in grouped:
            merged_row = OrderedDict(row)
            merged_row['_mask_rows'] = []
            merged_packet = {
                'leader_id': leader_id,
                'members': [],
                'points_reference': [],
                'total_local_points': 0,
                'area_points_total': 0,
                'member_upload_bytes': 0,
                'leader_own_bytes': 0,
                'missing_members': [],
                'area_ids': [],
            }
            grouped[leader_id] = [merged_row, merged_packet]
        merged_row, merged_packet = grouped[leader_id]
        merged_row['_mask_rows'].append(row)
        merged_packet['area_ids'].append(row['area_id'])
        merged_packet['total_local_points'] += int(packet['total_local_points'])
        merged_packet['area_points_total'] += int(packet['area_points_total'])
        merged_packet['member_upload_bytes'] += int(
            packet['member_upload_bytes'])
        merged_packet['leader_own_bytes'] += int(packet['leader_own_bytes'])
        merged_packet['missing_members'].extend(packet['missing_members'])
        merged_packet['members'].extend(packet['members'])
        if packet['points_reference'].shape[0] > 0:
            merged_packet['points_reference'].append(
                packet['points_reference'])

    merged = []
    for _leader_id, (row, packet) in grouped.items():
        row['area_id'] = ';'.join(packet['area_ids'])
        members = list(OrderedDict((str(member), None)
                                   for member in packet['members']).keys())
        packet['members'] = members
        packet['missing_members'] = list(OrderedDict(
            (str(member), None) for member in packet['missing_members']).keys())
        if packet['points_reference']:
            packet['points_reference'] = np.vstack(
                packet['points_reference']).astype(np.float32)
        else:
            packet['points_reference'] = np.empty((0, 4), dtype=np.float32)
        merged.append((row, packet))
    return merged


def run_feature_fusion(manager, point_packets, packet_rows, reference_pose,
                       geometry, args):
    processed, valid_indices = preprocess_point_packets(manager, point_packets)
    if processed is None:
        return None

    valid_rows = [packet_rows[index] for index in valid_indices]
    model = manager.model
    with torch.no_grad():
        batch_dict = model.pillar_vfe(processed)
        batch_dict = model.scatter(batch_dict)
        leader_scatter = batch_dict['spatial_features']
        fusion_input, query_count = make_query_stack(
            leader_scatter,
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

        external_mask = None
        mask_keep_ratio = ''
        if args.mask_mode != 'none':
            external_mask, mask_keep_ratio = build_packet_masks(
                valid_rows,
                reference_pose,
                geometry,
                args)
            external_mask = prepend_query_mask(external_mask, query_count)
            external_mask = external_mask.to(manager.device)

        feature_list = model.backbone.get_multiscale_feature(spatial_features)
        normalized_affine_matrix = identity_affine(
            1,
            int(record_len[0].item()),
            manager.device)
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
            fused_i, comm_rate = fuse_module(
                feature_i,
                psm_single,
                record_len,
                normalized_affine_matrix,
                None,
                external_comm_mask=external_mask)
            fused_feature_list.append(fused_i)
            communication_rates.append(comm_rate)
            comm_mbps_scales.append({
                'scale': scale_index,
                'comm_rate': comm_rate,
                'payload_channels': int(payload_channels),
                'height': int(feature_i.shape[-2]),
                'width': int(feature_i.shape[-1]),
            })

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
        ]).mean()
        if communication_rates else fused_feature.new_tensor(0.0))
    return {
        'psm': psm,
        'rm': rm,
        'leader_scatter': leader_scatter,
        'fusion_input': fusion_input,
        'valid_indices': valid_indices,
        'valid_rows': valid_rows,
        'query_count': query_count,
        'mask_keep_ratio': mask_keep_ratio,
        'comm_rate': comm_rate,
        'comm_mbps_meta': {
            'kind': 'lgcp_leader_packet_where2comm_multiscale',
            'record_len': record_len.detach().cpu().tolist(),
            'compression_ratio': int(model.compression_ratio),
            'scales': comm_mbps_scales,
        },
    }


def build_global_gt(manager, dataset, scenario_id, timestamp, reference_pose,
                    reference_cav_id, frame_plan):
    cav_ids = candidate_cavs_from_plan(
        frame_plan,
        reference_cav_id,
        manager.opencood_dataset.max_cav)
    frame = load_frame_for_reference(
        dataset,
        scenario_id,
        timestamp,
        cav_ids,
        reference_pose,
        reference_cav_id)
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        reference_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    gt_batch = manager.to_device(output_dict)
    return generate_gt(manager, gt_batch)


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


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
    coperception_params['_dataset_root_override'] = args.dataset_root
    checkpoint_dir = coperception_params['models'][args.fusion_method]
    geometry = load_model_geometry(checkpoint_dir)
    manager = OpenCOODManager(coperception_params)
    external_mode = (
        'intersection'
        if args.mask_mode == 'lgcp_area_objectness'
        else 'replace')
    where2comm_modules = enable_external_mask_semantics(
        manager.model,
        external_mode)

    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }
    frame_rows = []
    packet_rows_out = []
    scale_rows = []

    for frame_index, timestamp in enumerate(timestamps, start=1):
        frame_plan = grouped[timestamp]
        if args.max_areas_per_frame:
            frame_plan = frame_plan[:args.max_areas_per_frame]

        reference_pose, reference_source = resolve_reference_pose(
            dataset,
            args.scenario_id,
            timestamp,
            args.reference_cav_id,
            None)
        if args.reference_z_override is not None:
            reference_pose = list(reference_pose)
            reference_pose[2] = float(args.reference_z_override)

        cav_ids = sorted(set(
            [str(args.reference_cav_id)] +
            [str(row['leader_id']) for row in frame_plan] +
            [
                str(member)
                for row in frame_plan
                for member in str(row['group_members']).split(';')
                if str(member) != ''
            ]))
        frame = load_frame_for_reference(
            dataset,
            args.scenario_id,
            timestamp,
            cav_ids,
            reference_pose,
            args.reference_cav_id)

        point_packets = []
        packet_meta = []
        for row in frame_plan:
            packet = build_area_leader_points(
                frame,
                row,
                reference_pose,
                args.grid_size_x,
                args.grid_size_y)
            point_packets.append(packet['points_reference'])
            packet_meta.append((row, packet))

        if args.packet_granularity == 'leader':
            packet_meta = merge_leader_area_packets(packet_meta)
            point_packets = [
                packet['points_reference'] for _row, packet in packet_meta
            ]

        fusion = run_feature_fusion(
            manager,
            point_packets,
            [item[0] for item in packet_meta],
            reference_pose,
            geometry,
            args)
        if fusion is None:
            gt_box_tensor = build_global_gt(
                manager,
                dataset,
                args.scenario_id,
                timestamp,
                reference_pose,
                args.reference_cav_id,
                frame_plan)
            pred_box_tensor = None
            pred_score = None
            comm_rate = ''
            comm_bits = ''
            feature_mbps = ''
        else:
            pred_box_tensor, pred_score = postprocess_predictions(
                manager,
                fusion['psm'],
                fusion['rm'],
                args.postprocess_score_threshold)
            gt_box_tensor = build_global_gt(
                manager,
                dataset,
                args.scenario_id,
                timestamp,
                reference_pose,
                args.reference_cav_id,
                frame_plan)
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

            estimated = estimate_feature_mask_bits(
                fusion['comm_mbps_meta'],
                args.feature_value_bits)
            comm_bits = ''
            feature_mbps = ''
            if estimated is not None:
                comm_bits_value, _leader_once_bits, current_scale_rows = (
                    estimated)
                comm_bits = format_float(comm_bits_value)
                feature_mbps = format_float(
                    comm_bits_value * args.frame_rate_hz / 1e6)
                for row in current_scale_rows:
                    row.update({
                        'scenario_id': args.scenario_id,
                        'timestamp': timestamp,
                        'frame_index': frame_index,
                    })
                    scale_rows.append(row)
            comm_rate = format_float(to_float(fusion['comm_rate']))

        member_upload = sum(packet['member_upload_bytes']
                            for _row, packet in packet_meta)
        leader_own = sum(packet['leader_own_bytes']
                         for _row, packet in packet_meta)
        valid_packets = (
            0 if fusion is None else len(fusion['valid_indices']))
        for packet_index, (row, packet) in enumerate(packet_meta):
            packet_rows_out.append(OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'frame_index': frame_index,
                'packet_index': packet_index,
                'area_id': row['area_id'],
                'leader_id': row['leader_id'],
                'group_members': ';'.join(packet.get('members', [])),
                'area_points_total': packet['area_points_total'],
                'member_upload_bytes': packet['member_upload_bytes'],
                'leader_own_bytes': packet['leader_own_bytes'],
                'missing_members': ';'.join(packet['missing_members']),
            }))

        frame_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'frame_index': frame_index,
            'planned_areas': len(frame_plan),
            'valid_leader_packets': valid_packets,
            'query_mode': args.query_mode,
            'packet_granularity': args.packet_granularity,
            'mask_mode': args.mask_mode,
            'mask_keep_ratio': '' if fusion is None
            else fusion['mask_keep_ratio'],
            'comm_rate': comm_rate,
            'second_hop_feature_bits': comm_bits,
            'second_hop_feature_mbps': feature_mbps,
            'member_upload_bytes': member_upload,
            'leader_own_area_bytes': leader_own,
            'pred_boxes': 0 if pred_box_tensor is None
            else int(pred_box_tensor.shape[0]),
            'gt_boxes': 0 if gt_box_tensor is None
            else int(gt_box_tensor.shape[0]),
            'leader_scatter_shape': '' if fusion is None else shape_string(
                fusion['leader_scatter']),
            'fusion_input_shape': '' if fusion is None else shape_string(
                fusion['fusion_input']),
        }))
        print(
            'frame=%s/%s timestamp=%s areas=%s packets=%s pred=%s gt=%s '
            'comm_rate=%s mbps=%s' % (
                frame_index,
                len(timestamps),
                timestamp,
                len(frame_plan),
                valid_packets,
                frame_rows[-1]['pred_boxes'],
                frame_rows[-1]['gt_boxes'],
                comm_rate,
                feature_mbps))

    second_hop_bits = [
        float(row['second_hop_feature_bits'])
        for row in frame_rows
        if row['second_hop_feature_bits'] != ''
    ]
    summary = OrderedDict({
        'frames': len(frame_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': args.fusion_method,
        'checkpoint_dir': os.path.abspath(checkpoint_dir),
        'where2comm_modules': where2comm_modules,
        'query_mode': args.query_mode,
        'packet_granularity': args.packet_granularity,
        'mask_mode': args.mask_mode,
        'external_mask_mode': external_mode,
        'planned_areas_mean': format_float(np.mean([
            float(row['planned_areas']) for row in frame_rows
        ])) if frame_rows else '',
        'valid_leader_packets_mean': format_float(np.mean([
            float(row['valid_leader_packets']) for row in frame_rows
        ])) if frame_rows else '',
        'pred_samples': len(result_stat[0.5]['tp']),
        'gt_boxes': result_stat[0.5]['gt'],
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'avg_comm_rate': format_float(np.mean([
            float(row['comm_rate']) for row in frame_rows
            if row['comm_rate'] != ''
        ])) if any(row['comm_rate'] != '' for row in frame_rows) else '',
        'avg_second_hop_bits_per_frame': (
            format_float(np.mean(second_hop_bits))
            if second_hop_bits else ''),
        'avg_second_hop_mbps': (
            format_float(np.mean(second_hop_bits) * args.frame_rate_hz / 1e6)
            if second_hop_bits else ''),
        'member_upload_bytes_per_frame': format_float(np.mean([
            float(row['member_upload_bytes']) for row in frame_rows
        ])) if frame_rows else '',
        'leader_own_area_bytes_per_frame': format_float(np.mean([
            float(row['leader_own_area_bytes']) for row in frame_rows
        ])) if frame_rows else '',
        'score_threshold': args.postprocess_score_threshold,
        'reference_cav_id': args.reference_cav_id,
        'reference_z_override': args.reference_z_override,
    })

    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'frame_summary.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    if packet_rows_out:
        write_csv(os.path.join(args.output_dir, 'leader_packets.csv'),
                  list(packet_rows_out[0].keys()),
                  packet_rows_out)
    if scale_rows:
        write_csv(os.path.join(args.output_dir, 'feature_scale_summary.csv'),
                  list(scale_rows[0].keys()),
                  scale_rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'),
              list(summary.keys()),
              [summary])
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)

    print('Wrote LGCP Where2comm leader feature fusion to %s' %
          args.output_dir)
    print('AP@0.3=%s AP@0.5=%s AP@0.7=%s comm_rate=%s mbps=%s' % (
        summary['ap_03'],
        summary['ap_05'],
        summary['ap_07'],
        summary['avg_comm_rate'],
        summary['avg_second_hop_mbps']))


if __name__ == '__main__':
    main()
