# -*- coding: utf-8 -*-
"""
Probe a two-hop LGCP intermediate-feature route with Where2comm.

Pipeline:
1. Member CAVs crop their local point clouds to an LGCP area.
2. Each member encodes the cropped points to PointPillar BEV features.
3. The area leader runs Where2comm intermediate fusion over member features.
4. The leader uploads the fused feature packet to the RSU.
5. The RSU runs a second Where2comm intermediate fusion over leader packets.

This tool is intentionally standalone so existing SGCP and one-hop LGCP
experiments remain untouched.
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
    area_bounds_world,
    build_planned_area_bounds,
    calculate_ap_safe,
    filter_boxes_to_planned_areas,
    format_float,
    generate_gt,
    grouped_by_timestamp,
    load_frame_for_reference,
    local_points_to_world,
    parse_members,
    postprocess_predictions,
    read_csv,
    resolve_reference_pose,
    selected_timestamps,
    shape_string,
    update_stats,
    world_points_to_reference,
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
from opencood.models.fuse_modules.fusion_in_one import (
    AttFusion,
    warp_affine_simple,
)
from opencood.utils.transformation_utils import normalize_pairwise_tfm, x1_to_x2


def parse_args():
    parser = argparse.ArgumentParser(
        description='LGCP two-hop intermediate feature fusion with Where2comm.')
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
    parser.add_argument('--packet-granularity',
                        choices=['area', 'leader'],
                        default='area')
    parser.add_argument('--first-hop-projection',
                        choices=['project_first', 'project_full_first',
                                 'project_full_reference',
                                 'local_feature_warp'],
                        default='project_first',
                        help='project_first encodes member area points after '
                             'projection to leader coordinates; '
                             'project_full_first encodes full member points '
                             'after projection and only masks feature '
                             'communication outside the LGCP area; '
                             'project_full_reference encodes full member '
                             'points in the RSU/reference coordinates as a '
                             'coordinate-alignment diagnostic; '
                             'local_feature_warp encodes local member points '
                             'and relies on feature-level warping.')
    parser.add_argument('--area-merge-mode',
                        choices=['max', 'mean'],
                        default='max',
                        help='How to merge multiple area packets assigned to '
                             'the same leader before leader-to-RSU upload.')
    parser.add_argument('--query-mode',
                        choices=['mean', 'zero', 'first_leader'],
                        default='mean')
    parser.add_argument('--second-hop-pairwise-mode',
                        choices=['normal', 'inverse'],
                        default='normal',
                        help='Pairwise transform convention for leader-to-RSU '
                             'feature warping. normal uses x1_to_x2(src, dst); '
                             'inverse swaps the source/destination order as a '
                             'coordinate-alignment diagnostic.')
    parser.add_argument('--rsu-fusion-mode',
                        choices=['where2comm', 'direct_mean', 'direct_max',
                                 'direct_att'],
                        default='where2comm',
                        help='RSU aggregation over leader fused features. '
                             'where2comm keeps the original second-hop '
                             'selector; direct_* fuses leader fused features '
                             'without treating them as normal CAV features for '
                             'a second Where2comm selector.')
    parser.add_argument('--mask-mode',
                        choices=['none', 'lgcp_area',
                                 'lgcp_area_objectness', 'full'],
                        default='lgcp_area_objectness')
    parser.add_argument('--mask-dilation-cells', type=int, default=1)
    parser.add_argument('--eval-scope',
                        choices=['planned_areas', 'full'],
                        default='planned_areas')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=0.05)
    parser.add_argument('--bytes-per-point', type=int, default=16)
    parser.add_argument('--feature-value-bits', type=int, default=16)
    parser.add_argument('--frame-rate-hz', type=float, default=10.0)
    return parser.parse_args()


def unique_strings(values):
    return list(OrderedDict((str(value), None) for value in values).keys())


def normalize_cav_key(value):
    try:
        return int(value)
    except ValueError:
        return str(value)


def crop_world_to_area(points_local, lidar_pose, row, grid_size_x,
                       grid_size_y):
    if points_local is None or points_local.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    bounds = area_bounds_world(row, grid_size_x, grid_size_y)
    world_points = local_points_to_world(points_local, lidar_pose)
    x0, x1, y0, y1 = bounds
    keep = (
        (world_points[:, 0] >= x0) &
        (world_points[:, 0] < x1) &
        (world_points[:, 1] >= y0) &
        (world_points[:, 1] < y1))
    return world_points[keep].astype(np.float32)


def build_area_member_packets(frame, row, grid_size_x, grid_size_y,
                              bytes_per_point, first_hop_projection,
                              reference_pose=None):
    leader_id = str(row['leader_id'])
    members = parse_members(row['group_members'])
    if leader_id not in members:
        members = [leader_id] + members
    members = unique_strings(members)

    point_packets = []
    poses = []
    valid_member_ids = []
    member_rows = []
    raw_equiv_member_bytes = 0
    leader_own_raw_equiv_bytes = 0
    missing_members = []

    leader_pose = None
    leader_key = normalize_cav_key(leader_id)
    if leader_key in frame:
        leader_pose = frame[leader_key]['params']['lidar_pose']

    for member_id in members:
        key = normalize_cav_key(member_id)
        if key not in frame:
            missing_members.append(member_id)
            continue
        cav = frame[key]
        area_world = crop_world_to_area(
            cav['lidar_np'],
            cav['params']['lidar_pose'],
            row,
            grid_size_x,
            grid_size_y)
        cropped_points = int(area_world.shape[0])
        if first_hop_projection in ('project_first', 'project_full_first',
                                    'project_full_reference'):
            if first_hop_projection == 'project_full_reference':
                projection_pose = reference_pose
            else:
                projection_pose = leader_pose
            if projection_pose is None:
                missing_members.append(member_id)
                continue
            if first_hop_projection in ('project_full_first',
                                        'project_full_reference'):
                full_world = local_points_to_world(
                    cav['lidar_np'],
                    cav['params']['lidar_pose'])
                cropped = world_points_to_reference(full_world,
                                                    projection_pose)
            else:
                cropped = world_points_to_reference(area_world,
                                                    projection_pose)
            feature_pose = projection_pose
        else:
            # Keep the member-local points that correspond to the same world
            # area. This is a diagnostic mode because the checkpoint is mainly
            # exercised in projection-first semantics.
            if cropped_points:
                world_all = local_points_to_world(
                    cav['lidar_np'],
                    cav['params']['lidar_pose'])
                bounds = area_bounds_world(row, grid_size_x, grid_size_y)
                x0, x1, y0, y1 = bounds
                keep = (
                    (world_all[:, 0] >= x0) &
                    (world_all[:, 0] < x1) &
                    (world_all[:, 1] >= y0) &
                    (world_all[:, 1] < y1))
                cropped = cav['lidar_np'][keep].astype(np.float32)
            else:
                cropped = np.empty((0, 4), dtype=np.float32)
            feature_pose = cav['params']['lidar_pose']
        area_bytes = int(cropped_points * bytes_per_point)
        if str(member_id) == leader_id:
            leader_own_raw_equiv_bytes += area_bytes
        else:
            raw_equiv_member_bytes += area_bytes
        point_packets.append(cropped)
        poses.append(feature_pose)
        valid_member_ids.append(str(member_id))
        member_rows.append(OrderedDict({
            'member_id': str(member_id),
            'is_leader': str(member_id) == leader_id,
            'original_points': int(cav['lidar_np'].shape[0]),
            'area_points': cropped_points,
            'raw_equiv_bytes': area_bytes,
        }))

    return {
        'leader_id': leader_id,
        'members': valid_member_ids,
        'point_packets': point_packets,
        'poses': poses,
        'raw_equiv_member_bytes': raw_equiv_member_bytes,
        'leader_own_raw_equiv_bytes': leader_own_raw_equiv_bytes,
        'missing_members': missing_members,
        'member_rows': member_rows,
        'leader_pose': reference_pose
        if first_hop_projection == 'project_full_reference'
        else leader_pose,
    }


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


def build_pairwise_matrix(poses, device, mode='normal'):
    count = len(poses)
    matrix = np.zeros((1, count, count, 4, 4), dtype=np.float32)
    for i in range(count):
        for j in range(count):
            if i == j:
                matrix[0, i, j] = np.eye(4, dtype=np.float32)
            elif mode == 'inverse':
                matrix[0, i, j] = x1_to_x2(poses[j], poses[i]).astype(
                    np.float32)
            else:
                matrix[0, i, j] = x1_to_x2(poses[i], poses[j]).astype(
                    np.float32)
    return torch.from_numpy(matrix).to(device)


def repeat_area_mask(row, reference_pose, geometry, count, args):
    if args.mask_mode == 'none':
        return None, ''
    if args.mask_mode == 'full':
        mask = np.ones((count, 1, geometry['height'], geometry['width']),
                       dtype=np.float32)
        return torch.from_numpy(mask), '1.000000'
    area_mask = build_lgcp_area_mask(
        [row],
        reference_pose,
        geometry,
        args.grid_size_x,
        args.grid_size_y,
        args.mask_dilation_cells)
    masks = np.repeat(area_mask[None, None, :, :], count, axis=0).astype(
        np.float32)
    return torch.from_numpy(masks), format_float(masks.mean())


def build_packet_masks(packet_rows, reference_pose, geometry, args):
    if args.mask_mode == 'none':
        return None, ''
    if args.mask_mode == 'full':
        mask = np.ones((len(packet_rows), 1, geometry['height'],
                        geometry['width']), dtype=np.float32)
        return torch.from_numpy(mask), '1.000000'
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


def make_query_multiscale(packet_features, query_mode):
    if query_mode == 'first_leader':
        return packet_features, 0
    output = []
    for feature_i in packet_features:
        if query_mode == 'mean':
            query = torch.mean(feature_i, dim=0, keepdim=True)
        elif query_mode == 'zero':
            query = torch.zeros_like(feature_i[:1])
        else:
            raise ValueError('Unknown query_mode: %s' % query_mode)
        output.append(torch.cat([query, feature_i], dim=0))
    return output, 1


def prepend_query_psm(psm_stack, query_count, query_mode):
    if query_count <= 0:
        return psm_stack
    if query_mode == 'zero':
        query = torch.zeros_like(psm_stack[:1])
    else:
        query = torch.mean(psm_stack, dim=0, keepdim=True)
    return torch.cat([query, psm_stack], dim=0)


def prepend_query_mask(mask, query_count):
    if query_count <= 0 or mask is None:
        return mask
    query_mask = torch.ones_like(mask[:query_count])
    return torch.cat([query_mask, mask], dim=0)


def run_where2comm_multiscale(model, feature_list, psm_single, record_len,
                              normalized_affine_matrix, external_mask,
                              value_kind):
    fused_feature_list = []
    communication_rates = []
    comm_mbps_scales = []
    for scale_index, fuse_module in enumerate(model.fusion_net):
        feature_i = feature_list[scale_index]
        payload_channels = feature_i.shape[1]
        if model.compression:
            payload_channels = max(1,
                                   payload_channels //
                                   model.compression_ratio)
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
    comm_rate = (
        torch.stack([
            rate if torch.is_tensor(rate)
            else fused_feature_list[0].new_tensor(float(rate))
            for rate in communication_rates
        ]).mean()
        if communication_rates else
        fused_feature_list[0].new_tensor(0.0))
    return fused_feature_list, comm_rate, {
        'kind': value_kind,
        'record_len': record_len.detach().cpu().tolist(),
        'compression_ratio': int(model.compression_ratio),
        'scales': comm_mbps_scales,
    }


def direct_mean_or_max(feature_i, record_len, normalized_affine_matrix,
                       query_count, mode):
    _, _, height, width = feature_i.shape
    cav_count = int(record_len[0].item())
    tfm = normalized_affine_matrix[0][:cav_count, :cav_count, :, :]
    warped = warp_affine_simple(feature_i, tfm[0, :, :, :], (height, width))
    payload = warped[query_count:] if query_count > 0 else warped
    if payload.shape[0] == 0:
        payload = warped
    if mode == 'direct_max':
        return torch.max(payload, dim=0)[0].unsqueeze(0)
    if mode == 'direct_mean':
        return torch.mean(payload, dim=0, keepdim=True)
    raise ValueError('Unsupported direct mode: %s' % mode)


def run_direct_rsu_fusion(model, feature_list, record_len,
                          normalized_affine_matrix, query_count, mode):
    fused_feature_list = []
    comm_mbps_scales = []
    for scale_index, feature_i in enumerate(feature_list):
        payload_channels = feature_i.shape[1]
        transmitted_feature = feature_i
        if model.compression:
            payload_channels = max(1,
                                   payload_channels //
                                   model.compression_ratio)
            transmitted_feature = model.naive_compressor_list[scale_index](
                feature_i,
                use_fp16=False)
        if mode in ('direct_mean', 'direct_max'):
            fused_i = direct_mean_or_max(
                transmitted_feature,
                record_len,
                normalized_affine_matrix,
                query_count,
                mode)
        elif mode == 'direct_att':
            fused_i = AttFusion(transmitted_feature.shape[1]).to(
                transmitted_feature.device)(
                    transmitted_feature,
                    record_len,
                    normalized_affine_matrix,
                    use_warp_feature=True)
        else:
            raise ValueError('Unsupported direct RSU fusion mode: %s' % mode)
        fused_feature_list.append(fused_i)
        comm_mbps_scales.append({
            'scale': scale_index,
            'comm_rate': transmitted_feature.new_tensor(1.0),
            'payload_channels': int(payload_channels),
            'height': int(transmitted_feature.shape[-2]),
            'width': int(transmitted_feature.shape[-1]),
        })
    return fused_feature_list, feature_list[0].new_tensor(1.0), {
        'kind': 'lgcp_two_hop_direct_fused_feature_%s' % mode,
        'record_len': record_len.detach().cpu().tolist(),
        'compression_ratio': int(model.compression_ratio),
        'scales': comm_mbps_scales,
    }


def encode_area_to_leader_feature(manager, frame, row, geometry, args,
                                  reference_pose=None):
    model = manager.model
    member_packet = build_area_member_packets(
        frame,
        row,
        args.grid_size_x,
        args.grid_size_y,
        args.bytes_per_point,
        args.first_hop_projection,
        reference_pose)
    processed, valid_indices = preprocess_point_packets(
        manager,
        member_packet['point_packets'])
    if processed is None:
        return None, member_packet

    valid_poses = [member_packet['poses'][index] for index in valid_indices]
    valid_members = [member_packet['members'][index] for index in valid_indices]
    leader_id = str(row['leader_id'])
    leader_pose = None
    if member_packet.get('leader_pose') is not None:
        leader_pose = member_packet['leader_pose']
    elif leader_id in member_packet['members']:
        leader_pose = member_packet['poses'][
            member_packet['members'].index(leader_id)]
    elif valid_poses:
        leader_pose = valid_poses[0]
    else:
        return None, member_packet

    with torch.no_grad():
        batch_dict = model.pillar_vfe(processed)
        batch_dict = model.scatter(batch_dict)
        if valid_members[0] != leader_id:
            zero_leader_feature = torch.zeros_like(
                batch_dict['spatial_features'][:1])
            batch_dict['spatial_features'] = torch.cat(
                [zero_leader_feature, batch_dict['spatial_features']],
                dim=0)
            valid_poses = [leader_pose] + valid_poses
            valid_members = [leader_id] + valid_members
        batch_dict = model.backbone(batch_dict)
        spatial_features = batch_dict['spatial_features']
        spatial_features_2d_single = batch_dict['spatial_features_2d']
        if model.shrink_flag:
            spatial_features_2d_single = model.shrink_conv(
                spatial_features_2d_single)
        psm_single = model.cls_head(spatial_features_2d_single)
        feature_list = model.backbone.get_multiscale_feature(spatial_features)

        _, _, h0, w0 = spatial_features.shape
        pairwise_t_matrix = build_pairwise_matrix(
            valid_poses,
            manager.device)
        normalized_affine_matrix = normalize_pairwise_tfm(
            pairwise_t_matrix,
            h0,
            w0,
            model.voxel_size[0])
        record_len = torch.tensor([len(valid_poses)],
                                  dtype=torch.int64,
                                  device=manager.device)
        external_mask, mask_keep_ratio = repeat_area_mask(
            row,
            leader_pose,
            geometry,
            len(valid_poses),
            args)
        if external_mask is not None:
            external_mask = external_mask.to(manager.device)
        fused_list, comm_rate, comm_meta = run_where2comm_multiscale(
            model,
            feature_list,
            psm_single,
            record_len,
            normalized_affine_matrix,
            external_mask,
            'lgcp_two_hop_member_to_leader_where2comm')

        leader_decoded = model.backbone.decode_multiscale_feature(fused_list)
        if model.shrink_flag:
            leader_decoded = model.shrink_conv(leader_decoded)
        leader_psm = model.cls_head(leader_decoded)
        leader_rm = model.reg_head(leader_decoded)

    member_packet.update({
        'valid_members': valid_members,
        'valid_indices': valid_indices,
    })
    return {
        'row': row,
        'leader_id': leader_id,
        'leader_pose': leader_pose,
        'feature_list': fused_list,
        'psm': leader_psm,
        'rm': leader_rm,
        'comm_rate': comm_rate,
        'comm_mbps_meta': comm_meta,
        'mask_keep_ratio': mask_keep_ratio,
        'valid_members': valid_members,
    }, member_packet


def merge_area_features_by_leader(area_features, merge_mode):
    grouped = OrderedDict()
    for packet in area_features:
        leader_id = str(packet['leader_id'])
        grouped.setdefault(leader_id, []).append(packet)

    merged = []
    for leader_id, packets in grouped.items():
        if len(packets) == 1:
            merged.append(packets[0])
            continue
        feature_list = []
        for scale_index in range(len(packets[0]['feature_list'])):
            stack = torch.cat([
                packet['feature_list'][scale_index] for packet in packets
            ], dim=0)
            if merge_mode == 'max':
                merged_feature = torch.max(stack, dim=0)[0].unsqueeze(0)
            elif merge_mode == 'mean':
                merged_feature = torch.mean(stack, dim=0, keepdim=True)
            else:
                raise ValueError('Unknown area_merge_mode: %s' % merge_mode)
            feature_list.append(merged_feature)
        psm_stack = torch.cat([packet['psm'] for packet in packets], dim=0)
        if merge_mode == 'max':
            psm = torch.max(psm_stack, dim=0)[0].unsqueeze(0)
        else:
            psm = torch.mean(psm_stack, dim=0, keepdim=True)
        merged_row = OrderedDict(packets[0]['row'])
        merged_row['area_id'] = ';'.join(
            str(packet['row']['area_id']) for packet in packets)
        merged_row['_mask_rows'] = [packet['row'] for packet in packets]
        merged.append({
            'row': merged_row,
            'leader_id': leader_id,
            'leader_pose': packets[0]['leader_pose'],
            'feature_list': feature_list,
            'psm': psm,
            'comm_rate': '',
            'comm_mbps_meta': None,
            'mask_keep_ratio': '',
            'valid_members': unique_strings([
                member
                for packet in packets
                for member in packet.get('valid_members', [])
            ]),
        })
    return merged


def run_rsu_fusion(manager, leader_packets, reference_pose, geometry, args):
    if not leader_packets:
        return None
    model = manager.model
    with torch.no_grad():
        feature_list = []
        for scale_index in range(len(leader_packets[0]['feature_list'])):
            feature_list.append(torch.cat([
                packet['feature_list'][scale_index]
                for packet in leader_packets
            ], dim=0))
        psm_stack = torch.cat([packet['psm'] for packet in leader_packets],
                              dim=0)
        feature_list, query_count = make_query_multiscale(
            feature_list,
            args.query_mode)
        psm_single = prepend_query_psm(psm_stack, query_count,
                                       args.query_mode)
        poses = [reference_pose] * query_count + [
            packet['leader_pose'] for packet in leader_packets
        ]
        record_len = torch.tensor([len(poses)],
                                  dtype=torch.int64,
                                  device=manager.device)
        _, _, h0, w0 = feature_list[0].shape
        pairwise_t_matrix = build_pairwise_matrix(
            poses,
            manager.device,
            args.second_hop_pairwise_mode)
        normalized_affine_matrix = normalize_pairwise_tfm(
            pairwise_t_matrix,
            h0,
            w0,
            model.voxel_size[0])
        external_mask, mask_keep_ratio = build_packet_masks(
            [packet['row'] for packet in leader_packets],
            reference_pose,
            geometry,
            args)
        external_mask = prepend_query_mask(external_mask, query_count)
        if external_mask is not None:
            external_mask = external_mask.to(manager.device)
        if args.rsu_fusion_mode == 'where2comm':
            fused_list, comm_rate, comm_meta = run_where2comm_multiscale(
                model,
                feature_list,
                psm_single,
                record_len,
                normalized_affine_matrix,
                external_mask,
                'lgcp_two_hop_leader_to_rsu_where2comm')
        else:
            fused_list, comm_rate, comm_meta = run_direct_rsu_fusion(
                model,
                feature_list,
                record_len,
                normalized_affine_matrix,
                query_count,
                args.rsu_fusion_mode)
        fused_feature = model.backbone.decode_multiscale_feature(fused_list)
        if model.shrink_flag:
            fused_feature = model.shrink_conv(fused_feature)
        psm = model.cls_head(fused_feature)
        rm = model.reg_head(fused_feature)
    return {
        'psm': psm,
        'rm': rm,
        'comm_rate': comm_rate,
        'comm_mbps_meta': comm_meta,
        'mask_keep_ratio': mask_keep_ratio,
        'query_count': query_count,
        'feature_shapes': [
            shape_string(feature) for feature in feature_list
        ],
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


def evaluate_first_hop_leader_packet(manager, dataset, scenario_id, timestamp,
                                     feature_packet, args):
    row = feature_packet['row']
    leader_pose = feature_packet['leader_pose']
    leader_id = str(feature_packet['leader_id'])
    pred_box_tensor, pred_score = postprocess_predictions(
        manager,
        feature_packet['psm'],
        feature_packet['rm'],
        args.postprocess_score_threshold)
    gt_box_tensor = build_global_gt(
        manager,
        dataset,
        scenario_id,
        timestamp,
        leader_pose,
        leader_id,
        [row])
    planned_bounds = build_planned_area_bounds(
        [row],
        args.grid_size_x,
        args.grid_size_y)
    pred_box_tensor, pred_score = filter_boxes_to_planned_areas(
        pred_box_tensor,
        pred_score,
        leader_pose,
        planned_bounds)
    gt_box_tensor, _ = filter_boxes_to_planned_areas(
        gt_box_tensor,
        None,
        leader_pose,
        planned_bounds)
    return pred_box_tensor, pred_score, gt_box_tensor


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def bits_from_meta(meta, value_bits):
    estimated = estimate_feature_mask_bits(meta, value_bits)
    if estimated is None:
        return '', []
    bits, _leader_once_bits, scale_rows = estimated
    return format_float(bits), scale_rows


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
    first_hop_result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }
    frame_rows = []
    area_rows = []
    scale_rows = []

    for frame_index, timestamp in enumerate(timestamps, start=1):
        frame_plan = grouped[timestamp]
        if args.max_areas_per_frame:
            frame_plan = frame_plan[:args.max_areas_per_frame]

        reference_pose, _reference_source = resolve_reference_pose(
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

        area_features = []
        first_hop_bits_total = 0.0
        raw_equiv_member_bytes = 0
        leader_own_raw_equiv_bytes = 0

        for area_index, row in enumerate(frame_plan):
            feature_packet, member_packet = encode_area_to_leader_feature(
                manager,
                frame,
                row,
                geometry,
                args,
                reference_pose)
            raw_equiv_member_bytes += int(
                member_packet['raw_equiv_member_bytes'])
            leader_own_raw_equiv_bytes += int(
                member_packet['leader_own_raw_equiv_bytes'])
            first_bits = ''
            first_pred_boxes = 0
            first_gt_boxes = 0
            if feature_packet is not None:
                first_bits, current_scale_rows = bits_from_meta(
                    feature_packet['comm_mbps_meta'],
                    args.feature_value_bits)
                if first_bits != '':
                    first_hop_bits_total += float(first_bits)
                for scale_row in current_scale_rows:
                    scale_row.update({
                        'scenario_id': args.scenario_id,
                        'timestamp': timestamp,
                        'frame_index': frame_index,
                        'hop': 'member_to_leader',
                        'area_id': row['area_id'],
                        'leader_id': row['leader_id'],
                    })
                    scale_rows.append(scale_row)
                first_pred_box_tensor, first_pred_score, first_gt_box_tensor = (
                    evaluate_first_hop_leader_packet(
                        manager,
                        dataset,
                        args.scenario_id,
                        timestamp,
                        feature_packet,
                        args))
                update_stats(
                    first_hop_result_stat,
                    first_pred_box_tensor,
                    first_pred_score,
                    first_gt_box_tensor)
                first_pred_boxes = (
                    0 if first_pred_box_tensor is None
                    else int(first_pred_box_tensor.shape[0]))
                first_gt_boxes = (
                    0 if first_gt_box_tensor is None
                    else int(first_gt_box_tensor.shape[0]))
                area_features.append(feature_packet)
            area_rows.append(OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'frame_index': frame_index,
                'area_index': area_index,
                'area_id': row['area_id'],
                'leader_id': row['leader_id'],
                'planned_members': row['group_members'],
                'valid_members': '' if feature_packet is None else ';'.join(
                    feature_packet['valid_members']),
                'raw_equiv_member_bytes': member_packet[
                    'raw_equiv_member_bytes'],
                'leader_own_raw_equiv_bytes': member_packet[
                    'leader_own_raw_equiv_bytes'],
                'first_hop_feature_bits': first_bits,
                'first_hop_leader_pred_boxes': first_pred_boxes,
                'first_hop_leader_gt_boxes': first_gt_boxes,
                'missing_members': ';'.join(
                    member_packet['missing_members']),
            }))

        leader_packets = area_features
        if args.packet_granularity == 'leader':
            leader_packets = merge_area_features_by_leader(
                area_features,
                args.area_merge_mode)
        rsu_output = run_rsu_fusion(
            manager,
            leader_packets,
            reference_pose,
            geometry,
            args)

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
        second_hop_bits = ''
        second_hop_mbps = ''
        comm_rate = ''
        if rsu_output is not None:
            pred_box_tensor, pred_score = postprocess_predictions(
                manager,
                rsu_output['psm'],
                rsu_output['rm'],
                args.postprocess_score_threshold)
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
            second_hop_bits, current_scale_rows = bits_from_meta(
                rsu_output['comm_mbps_meta'],
                args.feature_value_bits)
            if second_hop_bits != '':
                second_hop_mbps = format_float(
                    float(second_hop_bits) * args.frame_rate_hz / 1e6)
            for scale_row in current_scale_rows:
                scale_row.update({
                    'scenario_id': args.scenario_id,
                    'timestamp': timestamp,
                    'frame_index': frame_index,
                    'hop': 'leader_to_rsu',
                    'area_id': '',
                    'leader_id': '',
                })
                scale_rows.append(scale_row)
            comm_rate = format_float(to_float(rsu_output['comm_rate']))

        first_hop_mbps = format_float(
            first_hop_bits_total * args.frame_rate_hz / 1e6)
        total_feature_bits = first_hop_bits_total + (
            0.0 if second_hop_bits == '' else float(second_hop_bits))
        frame_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'frame_index': frame_index,
            'planned_areas': len(frame_plan),
            'area_feature_packets': len(area_features),
            'leader_packets': len(leader_packets),
            'packet_granularity': args.packet_granularity,
            'first_hop_projection': args.first_hop_projection,
            'area_merge_mode': args.area_merge_mode,
            'query_mode': args.query_mode,
            'second_hop_pairwise_mode': args.second_hop_pairwise_mode,
            'rsu_fusion_mode': args.rsu_fusion_mode,
            'mask_mode': args.mask_mode,
            'rsu_mask_keep_ratio': '' if rsu_output is None else
            rsu_output['mask_keep_ratio'],
            'rsu_comm_rate': comm_rate,
            'first_hop_feature_bits': format_float(first_hop_bits_total),
            'first_hop_feature_mbps': first_hop_mbps,
            'second_hop_feature_bits': second_hop_bits,
            'second_hop_feature_mbps': second_hop_mbps,
            'total_feature_bits': format_float(total_feature_bits),
            'total_feature_mbps': format_float(
                total_feature_bits * args.frame_rate_hz / 1e6),
            'raw_equiv_member_bytes': raw_equiv_member_bytes,
            'leader_own_raw_equiv_bytes': leader_own_raw_equiv_bytes,
            'pred_boxes': 0 if pred_box_tensor is None
            else int(pred_box_tensor.shape[0]),
            'gt_boxes': 0 if gt_box_tensor is None
            else int(gt_box_tensor.shape[0]),
            'feature_shapes': '' if rsu_output is None else ';'.join(
                rsu_output['feature_shapes']),
        }))
        print(
            'frame=%s/%s timestamp=%s areas=%s leaders=%s pred=%s gt=%s '
            'first_mbps=%s second_mbps=%s total_mbps=%s' % (
                frame_index,
                len(timestamps),
                timestamp,
                len(frame_plan),
                len(leader_packets),
                frame_rows[-1]['pred_boxes'],
                frame_rows[-1]['gt_boxes'],
                frame_rows[-1]['first_hop_feature_mbps'],
                second_hop_mbps,
                frame_rows[-1]['total_feature_mbps']))

    first_hop_bits = [
        float(row['first_hop_feature_bits'])
        for row in frame_rows
        if row['first_hop_feature_bits'] != ''
    ]
    second_hop_bits_values = [
        float(row['second_hop_feature_bits'])
        for row in frame_rows
        if row['second_hop_feature_bits'] != ''
    ]
    total_bits = [
        float(row['total_feature_bits'])
        for row in frame_rows
        if row['total_feature_bits'] != ''
    ]
    summary = OrderedDict({
        'frames': len(frame_rows),
        'scenario_id': args.scenario_id,
        'fusion_method': args.fusion_method,
        'checkpoint_dir': os.path.abspath(checkpoint_dir),
        'where2comm_modules': where2comm_modules,
        'packet_granularity': args.packet_granularity,
        'first_hop_projection': args.first_hop_projection,
        'area_merge_mode': args.area_merge_mode,
        'query_mode': args.query_mode,
        'second_hop_pairwise_mode': args.second_hop_pairwise_mode,
        'rsu_fusion_mode': args.rsu_fusion_mode,
        'mask_mode': args.mask_mode,
        'external_mask_mode': external_mode,
        'planned_areas_mean': format_float(np.mean([
            float(row['planned_areas']) for row in frame_rows
        ])) if frame_rows else '',
        'leader_packets_mean': format_float(np.mean([
            float(row['leader_packets']) for row in frame_rows
        ])) if frame_rows else '',
        'pred_samples': len(result_stat[0.5]['tp']),
        'gt_boxes': result_stat[0.5]['gt'],
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'first_hop_leader_pred_samples': len(
            first_hop_result_stat[0.5]['tp']),
        'first_hop_leader_gt_boxes': first_hop_result_stat[0.5]['gt'],
        'first_hop_leader_ap_03': calculate_ap_safe(
            first_hop_result_stat,
            0.3),
        'first_hop_leader_ap_05': calculate_ap_safe(
            first_hop_result_stat,
            0.5),
        'first_hop_leader_ap_07': calculate_ap_safe(
            first_hop_result_stat,
            0.7),
        'avg_first_hop_bits_per_frame': format_float(np.mean(first_hop_bits))
        if first_hop_bits else '',
        'avg_first_hop_mbps': format_float(
            np.mean(first_hop_bits) * args.frame_rate_hz / 1e6)
        if first_hop_bits else '',
        'avg_second_hop_bits_per_frame': format_float(
            np.mean(second_hop_bits_values))
        if second_hop_bits_values else '',
        'avg_second_hop_mbps': format_float(
            np.mean(second_hop_bits_values) * args.frame_rate_hz / 1e6)
        if second_hop_bits_values else '',
        'avg_total_feature_bits_per_frame': format_float(np.mean(total_bits))
        if total_bits else '',
        'avg_total_feature_mbps': format_float(
            np.mean(total_bits) * args.frame_rate_hz / 1e6)
        if total_bits else '',
        'raw_equiv_member_bytes_per_frame': format_float(np.mean([
            float(row['raw_equiv_member_bytes']) for row in frame_rows
        ])) if frame_rows else '',
        'leader_own_raw_equiv_bytes_per_frame': format_float(np.mean([
            float(row['leader_own_raw_equiv_bytes']) for row in frame_rows
        ])) if frame_rows else '',
        'score_threshold': args.postprocess_score_threshold,
        'reference_cav_id': args.reference_cav_id,
        'reference_z_override': args.reference_z_override,
    })

    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'frame_summary.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    if area_rows:
        write_csv(os.path.join(args.output_dir, 'area_packets.csv'),
                  list(area_rows[0].keys()),
                  area_rows)
    if scale_rows:
        write_csv(os.path.join(args.output_dir, 'feature_scale_summary.csv'),
                  list(scale_rows[0].keys()),
                  scale_rows)
    write_csv(os.path.join(args.output_dir, 'summary.csv'),
              list(summary.keys()),
              [summary])
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)

    print('Wrote LGCP two-hop Where2comm feature fusion to %s' %
          args.output_dir)
    print('AP@0.3=%s AP@0.5=%s AP@0.7=%s first_mbps=%s '
          'second_mbps=%s total_mbps=%s' % (
              summary['ap_03'],
              summary['ap_05'],
              summary['ap_07'],
              summary['avg_first_hop_mbps'],
              summary['avg_second_hop_mbps'],
              summary['avg_total_feature_mbps']))


if __name__ == '__main__':
    main()
