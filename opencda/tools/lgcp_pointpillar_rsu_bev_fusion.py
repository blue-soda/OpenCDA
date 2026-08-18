# -*- coding: utf-8 -*-
"""
Run LGCP RSU-level BEV feature fusion with PointPillar attentive backbone.

This tool implements the current LGCP feature-fusion prototype:

1. For each LGCP area task, collect the leader and member point-cloud slices
   that fall inside the world-coordinate area cell.
2. Project those sliced points into a shared reference/RSU lidar frame.
3. Encode each area leader packet into a sparse scatter BEV canvas with the
   existing PointPillar VFE + scatter modules.
4. Stack all leader BEV canvases for one timestamp and feed them through the
   existing pointpillar_attentive_fusion backbone and detection heads.

The implementation is intentionally isolated from SGCP code paths.
"""

import argparse
import csv
import math
import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import load_coperception_params
from opencood.utils import eval_utils
from opencood.utils.transformation_utils import x1_to_x2, x_to_world


DEFAULT_LIDAR_RANGE = [-140.8, -40.0, -3.0, 140.8, 40.0, 1.0]
DEFAULT_VOXEL_SIZE = [0.4, 0.4, 4.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description='LGCP RSU BEV attentive fusion from area point slices.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_attentive')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--reference-cav-id', default='1',
                        help='Reference CAV used as RSU coordinate frame.')
    parser.add_argument('--reference-pose', nargs=6, type=float, default=None,
                        metavar=('X', 'Y', 'Z', 'ROLL', 'YAW', 'PITCH'),
                        help='Optional explicit RSU/reference lidar pose. '
                             'Overrides --reference-cav-id for coordinates.')
    parser.add_argument('--reference-z-override', type=float, default=None,
                        help='Optional z override for the resolved reference '
                             'pose. This keeps x/y/yaw from the RSU or CAV '
                             'while matching vehicle-trained checkpoint '
                             'lidar z-range assumptions.')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1,
                        help='Number of timestamps to run. Use 0 for all.')
    parser.add_argument('--max-areas-per-frame', type=int, default=0,
                        help='Optional smoke cap. 0 means all areas.')
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--query-mode',
                        choices=['first_leader', 'mean', 'zero'],
                        default='mean',
                        help='The attentive fusion module returns the first '
                             'agent query. mean/zero prepend a synthetic RSU '
                             'query canvas; first_leader preserves the native '
                             'OpenCOOD ordering assumption.')
    parser.add_argument('--packet-granularity',
                        choices=['area', 'leader'],
                        default='area',
                        help='area encodes one BEV packet per LGCP area-task; '
                             'leader merges all areas assigned to the same '
                             'leader into one BEV packet before RSU fusion.')
    parser.add_argument('--leader-feature-dtype',
                        choices=['float32', 'float16'],
                        default='float16',
                        help='Byte-accounting dtype for leader->RSU features.')
    parser.add_argument('--eval-scope', choices=['full', 'planned_areas'],
                        default='full',
                        help='full evaluates against all reference-frame GT; '
                             'planned_areas filters prediction/GT boxes by '
                             'the LGCP areas evaluated in this run.')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=None)
    parser.add_argument('--save-frame-tensors', action='store_true',
                        help='Save per-frame leader stacks and RSU outputs.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_by_timestamp(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['timestamp']].append(row)
    return OrderedDict((timestamp, grouped[timestamp])
                       for timestamp in sorted(grouped.keys()))


def selected_timestamps(grouped, start_index, max_frames):
    timestamps = list(grouped.keys())[start_index:]
    if max_frames:
        timestamps = timestamps[:max_frames]
    return timestamps


def parse_members(value):
    return [item for item in str(value).split(';') if item != '']


def unique_strings(values):
    return list(OrderedDict((str(value), None) for value in values).keys())


def normalize_cav_key(value):
    try:
        return int(value)
    except ValueError:
        return str(value)


def format_float(value):
    return '%.6f' % float(value)


def shape_string(tensor):
    if tensor is None:
        return ''
    return 'x'.join(str(int(item)) for item in tensor.shape)


def area_bounds_world(row, grid_size_x, grid_size_y):
    center_x = float(row['area_center_x'])
    center_y = float(row['area_center_y'])
    return (
        center_x - grid_size_x / 2.0,
        center_x + grid_size_x / 2.0,
        center_y - grid_size_y / 2.0,
        center_y + grid_size_y / 2.0,
    )


def project_points(points, matrix):
    if points is None or points.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    xyz1 = np.concatenate(
        [points[:, :3], np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=1)
    projected = np.dot(xyz1, matrix.T)[:, :3]
    output = points.copy()
    output[:, :3] = projected
    return output.astype(np.float32)


def local_points_to_world(points, lidar_pose):
    return project_points(points, x_to_world(lidar_pose))


def world_points_to_reference(points, reference_pose):
    return project_points(points, np.linalg.inv(x_to_world(reference_pose)))


def crop_world_area(points_world, bounds):
    if points_world is None or points_world.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    x0, x1, y0, y1 = bounds
    mask = (
        (points_world[:, 0] >= x0) &
        (points_world[:, 0] < x1) &
        (points_world[:, 1] >= y0) &
        (points_world[:, 1] < y1))
    return points_world[mask].astype(np.float32)


def load_frame_for_reference(dataset, scenario_id, timestamp, cav_ids,
                             reference_pose, reference_cav_id):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=reference_cav_id,
        cav_ids=cav_ids,
        add_transformation=False)
    for cav_id, cav_content in frame.items():
        cav_pose = cav_content['params']['lidar_pose']
        cav_content['ego'] = str(cav_id) == str(reference_cav_id)
        cav_content['params']['transformation_matrix'] = x1_to_x2(
            cav_pose,
            reference_pose)
        cav_content['params']['gt_transformation_matrix'] = (
            cav_content['params']['transformation_matrix'])
        cav_content['params']['spatial_correction_matrix'] = x1_to_x2(
            reference_pose,
            reference_pose)
    return frame


def resolve_reference_pose(dataset, scenario_id, timestamp, reference_cav_id,
                           reference_pose):
    if reference_pose is not None:
        return [float(item) for item in reference_pose], 'explicit'
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=reference_cav_id,
        cav_ids=[reference_cav_id],
        add_transformation=False)
    key = normalize_cav_key(reference_cav_id)
    if key not in frame:
        raise ValueError('reference_cav_id %s not found at %s' %
                         (reference_cav_id, timestamp))
    return frame[key]['params']['lidar_pose'], str(reference_cav_id)


def build_area_leader_points(frame, row, reference_pose, grid_size_x,
                             grid_size_y):
    leader_id = str(row['leader_id'])
    members = parse_members(row['group_members'])
    if leader_id not in members:
        members = [leader_id] + members
    members = unique_strings(members)
    bounds = area_bounds_world(row, grid_size_x, grid_size_y)

    reference_points = []
    member_rows = []
    total_local_points = 0
    area_points_total = 0
    member_upload_bytes = 0
    leader_own_bytes = 0
    missing_members = []

    for member_id in members:
        key = normalize_cav_key(member_id)
        if key not in frame:
            missing_members.append(member_id)
            continue
        cav = frame[key]
        local_points = cav['lidar_np']
        total_local_points += int(local_points.shape[0])
        world_points = local_points_to_world(
            local_points,
            cav['params']['lidar_pose'])
        area_world = crop_world_area(world_points, bounds)
        if area_world.size == 0:
            area_count = 0
            ref_points = np.empty((0, 4), dtype=np.float32)
        else:
            area_count = int(area_world.shape[0])
            ref_points = world_points_to_reference(area_world, reference_pose)
            reference_points.append(ref_points)
        area_points_total += area_count
        area_bytes = int(area_count * local_points.shape[1] * 4)
        if str(member_id) == leader_id:
            leader_own_bytes += area_bytes
        else:
            member_upload_bytes += area_bytes
        member_rows.append({
            'member_id': member_id,
            'local_points': int(local_points.shape[0]),
            'area_points': area_count,
            'area_bytes': area_bytes,
        })

    if reference_points:
        merged = np.vstack(reference_points).astype(np.float32)
    else:
        merged = np.empty((0, 4), dtype=np.float32)
    return {
        'leader_id': leader_id,
        'members': members,
        'points_reference': merged,
        'total_local_points': total_local_points,
        'area_points_total': area_points_total,
        'member_upload_bytes': member_upload_bytes,
        'leader_own_bytes': leader_own_bytes,
        'missing_members': missing_members,
        'member_rows': member_rows,
    }


def encode_scatter_features(manager, point_batches):
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
        return None, []

    collated = preprocessor.collate_batch(valid)
    batch_dict = {
        'voxel_features': collated['voxel_features'].to(manager.device),
        'voxel_coords': collated['voxel_coords'].to(manager.device),
        'voxel_num_points': collated['voxel_num_points'].to(manager.device),
    }
    with torch.no_grad():
        batch_dict = manager.model.pillar_vfe(batch_dict)
        batch_dict = manager.model.scatter(batch_dict)
    return batch_dict['spatial_features'], valid_indices


def make_query_stack(spatial_features, query_mode):
    if query_mode == 'first_leader':
        return spatial_features
    if query_mode == 'mean':
        query = torch.mean(spatial_features, dim=0, keepdim=True)
        return torch.cat([query, spatial_features], dim=0)
    if query_mode == 'zero':
        query = torch.zeros_like(spatial_features[:1])
        return torch.cat([query, spatial_features], dim=0)
    raise ValueError('Unknown query_mode: %s' % query_mode)


def run_rsu_backbone_and_heads(manager, leader_features, query_mode):
    fusion_input = make_query_stack(leader_features, query_mode)
    record_len = torch.tensor([fusion_input.shape[0]],
                              dtype=torch.int64,
                              device=manager.device)
    batch_dict = {
        'spatial_features': fusion_input,
        'record_len': record_len,
    }
    with torch.no_grad():
        batch_dict = manager.model.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        psm = manager.model.cls_head(spatial_features_2d)
        rm = manager.model.reg_head(spatial_features_2d)
    return fusion_input, spatial_features_2d, psm, rm


def postprocess_predictions(manager, psm, rm, score_threshold):
    post_processor = manager.opencood_dataset.post_processor
    original_threshold = post_processor.params['target_args'][
        'score_threshold']
    if score_threshold is not None:
        post_processor.params['target_args']['score_threshold'] = (
            score_threshold)
    try:
        anchor_box = post_processor.generate_anchor_box()
        device = psm.device
        data_dict = {
            'ego': {
                'anchor_box': torch.from_numpy(anchor_box).to(device),
                'transformation_matrix': torch.eye(4, device=device),
            }
        }
        output_dict = {'ego': {'psm': psm, 'rm': rm}}
        return post_processor.post_process(data_dict, output_dict)
    finally:
        post_processor.params['target_args']['score_threshold'] = (
            original_threshold)


def build_gt_batch(manager, dataset, scenario_id, timestamp, reference_pose,
                   reference_cav_id):
    cav_ids = dataset.scenarios[scenario_id]['cav_ids']
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
    return manager.to_device(output_dict)


def generate_gt(manager, gt_batch):
    return manager.opencood_dataset.post_processor.generate_gt_bbx(gt_batch)


def boxes_center_world_xy(box_tensor, reference_pose):
    if box_tensor is None or int(box_tensor.shape[0]) == 0:
        return np.empty((0, 2), dtype=np.float32)
    centers_ref = torch.mean(box_tensor[:, :, :2], dim=1).detach().cpu().numpy()
    matrix = x_to_world(reference_pose)
    xy1 = np.concatenate(
        [
            centers_ref,
            np.zeros((centers_ref.shape[0], 1), dtype=centers_ref.dtype),
            np.ones((centers_ref.shape[0], 1), dtype=centers_ref.dtype),
        ],
        axis=1)
    centers_world = np.dot(xy1, matrix.T)[:, :2]
    return centers_world.astype(np.float32)


def build_planned_area_bounds(frame_plan, grid_size_x, grid_size_y):
    return [
        (row['area_id'], area_bounds_world(row, grid_size_x, grid_size_y))
        for row in frame_plan
    ]


def merge_packets_by_leader(leader_packets):
    grouped = OrderedDict()
    for row, packet in leader_packets:
        leader_id = str(packet['leader_id'])
        if leader_id not in grouped:
            grouped[leader_id] = {
                'rows': [],
                'points': [],
                'members': OrderedDict(),
                'missing_members': OrderedDict(),
                'area_points_total': 0,
                'total_local_points': 0,
                'member_upload_bytes': 0,
                'leader_own_bytes': 0,
            }
        item = grouped[leader_id]
        item['rows'].append(row)
        if packet['points_reference'].shape[0] > 0:
            item['points'].append(packet['points_reference'])
        for member_id in packet['members']:
            item['members'][str(member_id)] = None
        for member_id in packet['missing_members']:
            item['missing_members'][str(member_id)] = None
        item['area_points_total'] += int(packet['area_points_total'])
        item['total_local_points'] += int(packet['total_local_points'])
        item['member_upload_bytes'] += int(packet['member_upload_bytes'])
        item['leader_own_bytes'] += int(packet['leader_own_bytes'])

    merged = []
    for leader_id, item in grouped.items():
        rows = item['rows']
        points = (
            np.vstack(item['points']).astype(np.float32)
            if item['points'] else np.empty((0, 4), dtype=np.float32))
        area_ids = [row['area_id'] for row in rows]
        synthetic_row = dict(rows[0])
        synthetic_row['area_id'] = ';'.join(str(area_id)
                                            for area_id in area_ids)
        synthetic_row['area_count'] = len(rows)
        packet = {
            'leader_id': leader_id,
            'members': list(item['members'].keys()),
            'points_reference': points,
            'total_local_points': item['total_local_points'],
            'area_points_total': item['area_points_total'],
            'member_upload_bytes': item['member_upload_bytes'],
            'leader_own_bytes': item['leader_own_bytes'],
            'missing_members': list(item['missing_members'].keys()),
            'member_rows': [],
        }
        merged.append((synthetic_row, packet))
    return merged


def area_membership_mask(box_tensor, reference_pose, planned_bounds):
    centers = boxes_center_world_xy(box_tensor, reference_pose)
    mask = np.zeros((centers.shape[0],), dtype=bool)
    for _area_id, bounds in planned_bounds:
        x0, x1, y0, y1 = bounds
        mask |= (
            (centers[:, 0] >= x0) &
            (centers[:, 0] < x1) &
            (centers[:, 1] >= y0) &
            (centers[:, 1] < y1))
    device = box_tensor.device if box_tensor is not None else torch.device('cpu')
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def filter_boxes_to_planned_areas(box_tensor, score_tensor, reference_pose,
                                  planned_bounds):
    if box_tensor is None:
        return None, None
    if int(box_tensor.shape[0]) == 0:
        return box_tensor, score_tensor
    mask = area_membership_mask(box_tensor, reference_pose, planned_bounds)
    filtered_boxes = box_tensor[mask]
    filtered_scores = None
    if score_tensor is not None:
        filtered_scores = score_tensor[mask.to(score_tensor.device)]
    return filtered_boxes, filtered_scores


def update_stats(result_stat, pred_box_tensor, pred_score, gt_box_tensor):
    if pred_box_tensor is None or pred_score is None:
        pred_box_tensor = gt_box_tensor.new_zeros((0, 8, 3))
        pred_score = gt_box_tensor.new_zeros((0,))
    for iou in (0.3, 0.5, 0.7):
        eval_utils.calculate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor,
                                   result_stat, iou)


def calculate_ap_safe(result_stat, iou):
    stat = {
        iou: {
            'tp': list(result_stat[iou]['tp']),
            'fp': list(result_stat[iou]['fp']),
            'gt': result_stat[iou]['gt'],
        }
    }
    if stat[iou]['gt'] == 0:
        return ''
    ap, _, _ = eval_utils.calculate_ap(stat, iou)
    return format_float(ap)


def nonzero_bev_cells(scatter_feature):
    # Count occupied BEV cells, independent of channel count.
    mask = torch.any(scatter_feature.detach() != 0, dim=1)
    return int(torch.count_nonzero(mask).item())


def save_frame_tensor(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def summarize_frame_tensors(leader_features, fusion_input, psm, rm,
                            spatial_features_2d):
    prob = torch.sigmoid(psm.detach())
    return {
        'leader_feature_stack_shape': shape_string(leader_features),
        'fusion_input_shape': shape_string(fusion_input),
        'spatial_features_2d_shape': shape_string(spatial_features_2d),
        'psm_shape': shape_string(psm),
        'rm_shape': shape_string(rm),
        'score_min': format_float(torch.min(prob).item()),
        'score_mean': format_float(torch.mean(prob).item()),
        'score_max': format_float(torch.max(prob).item()),
        'score_p95': format_float(torch.quantile(prob.flatten(), 0.95).item()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tensor_dir = os.path.join(args.output_dir, 'frame_tensors')

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

    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }
    area_rows = []
    frame_rows = []

    for frame_index, timestamp in enumerate(timestamps, start=1):
        reference_pose, reference_label = resolve_reference_pose(
            dataset,
            args.scenario_id,
            timestamp,
            args.reference_cav_id,
            args.reference_pose)
        if args.reference_z_override is not None:
            reference_pose = list(reference_pose)
            reference_pose[2] = float(args.reference_z_override)

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
        skipped_empty = 0
        for row in frame_plan:
            packet = build_area_leader_points(
                frame,
                row,
                reference_pose,
                args.grid_size_x,
                args.grid_size_y)
            if packet['points_reference'].shape[0] == 0:
                skipped_empty += 1
            leader_packets.append((row, packet))
            point_batches.append(packet['points_reference'])

        if args.packet_granularity == 'leader':
            leader_packets = merge_packets_by_leader(leader_packets)
            point_batches = [
                packet['points_reference'] for _row, packet in leader_packets
            ]

        leader_features, valid_indices = encode_scatter_features(
            manager,
            point_batches)
        if leader_features is None:
            gt_batch = build_gt_batch(
                manager,
                dataset,
                args.scenario_id,
                timestamp,
                reference_pose,
                args.reference_cav_id)
            gt_box_tensor = generate_gt(manager, gt_batch)
            empty_pred = gt_box_tensor.new_zeros((0, 8, 3))
            empty_score = gt_box_tensor.new_zeros((0,))
            update_stats(result_stat, empty_pred, empty_score, gt_box_tensor)
            frame_rows.append(empty_frame_row(
                args,
                timestamp,
                reference_label,
                frame_plan,
                skipped_empty,
                gt_box_tensor))
            continue

        valid_packets = [leader_packets[index] for index in valid_indices]
        fusion_input, spatial_features_2d, psm, rm = run_rsu_backbone_and_heads(
            manager,
            leader_features,
            args.query_mode)
        pred_box_tensor, pred_score = postprocess_predictions(
            manager,
            psm,
            rm,
            args.postprocess_score_threshold)
        gt_batch = build_gt_batch(
            manager,
            dataset,
            args.scenario_id,
            timestamp,
            reference_pose,
            args.reference_cav_id)
        gt_box_tensor = generate_gt(manager, gt_batch)
        raw_pred_box_tensor = pred_box_tensor
        raw_pred_score = pred_score
        raw_gt_box_tensor = gt_box_tensor
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
        update_stats(result_stat, pred_box_tensor, pred_score, gt_box_tensor)

        frame_tensor_summary = summarize_frame_tensors(
            leader_features,
            fusion_input,
            psm,
            rm,
            spatial_features_2d)
        bytes_per_feature = 2 if args.leader_feature_dtype == 'float16' else 4
        member_upload_bytes = 0
        leader_own_bytes = 0
        nonzero_cells_total = 0
        tensor_rel_path = ''
        tensor_bytes = 0

        for local_index, (row, packet) in enumerate(valid_packets):
            scatter = leader_features[local_index:local_index + 1]
            nz_cells = nonzero_bev_cells(scatter)
            nonzero_cells_total += nz_cells
            member_upload_bytes += int(packet['member_upload_bytes'])
            leader_own_bytes += int(packet['leader_own_bytes'])
            full_feature_bytes = int(scatter.numel() * bytes_per_feature)
            sparse_feature_bytes = int(nz_cells * scatter.shape[1] *
                                       bytes_per_feature)
            area_rows.append(OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'area_id': row['area_id'],
                'leader_id': packet['leader_id'],
                'group_members': ';'.join(packet['members']),
                'group_size': len(packet['members']),
                'area_center_x': row.get('area_center_x', ''),
                'area_center_y': row.get('area_center_y', ''),
                'reference': reference_label,
                'area_points_total': packet['area_points_total'],
                'total_local_points_seen': packet['total_local_points'],
                'member_upload_bytes': packet['member_upload_bytes'],
                'leader_own_area_bytes': packet['leader_own_bytes'],
                'missing_members': ';'.join(packet['missing_members']),
                'leader_scatter_shape': shape_string(scatter),
                'leader_nonzero_bev_cells': nz_cells,
                'leader_feature_full_bytes': full_feature_bytes,
                'leader_feature_sparse_cell_bytes': sparse_feature_bytes,
            }))

        if args.save_frame_tensors:
            tensor_rel_path = os.path.join(
                'frame_tensors',
                '%s_rsu_bev_attentive.npz' % timestamp)
            tensor_bytes = save_frame_tensor(
                os.path.join(args.output_dir, tensor_rel_path),
                {
                    'timestamp': np.asarray(timestamp),
                    'reference': np.asarray(reference_label),
                    'leader_features': leader_features.detach().cpu().numpy(),
                    'fusion_input': fusion_input.detach().cpu().numpy(),
                    'psm': psm.detach().cpu().numpy(),
                    'rm': rm.detach().cpu().numpy(),
                })

        raw_pred_count = 0 if raw_pred_box_tensor is None else int(
            raw_pred_box_tensor.shape[0])
        raw_gt_count = 0 if raw_gt_box_tensor is None else int(
            raw_gt_box_tensor.shape[0])
        pred_count = 0 if pred_box_tensor is None else int(
            pred_box_tensor.shape[0])
        score_count = 0 if pred_score is None else int(pred_score.shape[0])
        frame_row = OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'reference': reference_label,
            'reference_x': format_float(reference_pose[0]),
            'reference_y': format_float(reference_pose[1]),
            'reference_yaw': format_float(reference_pose[4]),
            'planned_areas': len(frame_plan),
            'valid_leader_features': len(valid_packets),
            'skipped_empty_area_packets': skipped_empty,
            'query_mode': args.query_mode,
            'packet_granularity': args.packet_granularity,
            'eval_scope': args.eval_scope,
            'member_upload_bytes': member_upload_bytes,
            'leader_own_area_bytes': leader_own_bytes,
            'leader_feature_full_bytes': int(
                leader_features.numel() * bytes_per_feature),
            'leader_feature_sparse_cell_bytes': int(
                nonzero_cells_total * leader_features.shape[1] *
                bytes_per_feature),
            'leader_nonzero_bev_cells': nonzero_cells_total,
            'raw_pred_boxes': raw_pred_count,
            'raw_gt_boxes': raw_gt_count,
            'eval_pred_boxes': pred_count,
            'eval_gt_boxes': int(gt_box_tensor.shape[0]),
            'pred_boxes': pred_count,
            'pred_scores': score_count,
            'gt_boxes': int(gt_box_tensor.shape[0]),
            'frame_tensor_file': tensor_rel_path.replace('\\', '/'),
            'frame_tensor_npz_bytes': tensor_bytes,
        })
        frame_row.update(frame_tensor_summary)
        frame_rows.append(frame_row)
        print('frame=%s/%s timestamp=%s leaders=%s pred=%s gt=%s' % (
            frame_index,
            len(timestamps),
            timestamp,
            len(valid_packets),
            pred_count,
            int(gt_box_tensor.shape[0])))

    summary_rows = [OrderedDict({
        'frames': len(frame_rows),
        'area_rows': len(area_rows),
        'fusion_method': coperception_params['fusion_method'],
        'query_mode': args.query_mode,
        'packet_granularity': args.packet_granularity,
        'eval_scope': args.eval_scope,
        'member_upload_bytes': sum(
            int(row['member_upload_bytes']) for row in frame_rows),
        'leader_feature_full_bytes': sum(
            int(row['leader_feature_full_bytes']) for row in frame_rows),
        'leader_feature_sparse_cell_bytes': sum(
            int(row['leader_feature_sparse_cell_bytes']) for row in frame_rows),
        'pred_boxes_mean': format_float(
            np.mean([float(row['pred_boxes']) for row in frame_rows])
            if frame_rows else 0.0),
        'gt_boxes_mean': format_float(
            np.mean([float(row['gt_boxes']) for row in frame_rows])
            if frame_rows else 0.0),
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'gt_total': result_stat[0.5]['gt'],
        'pred_samples': len(result_stat[0.5]['tp']),
    })]

    if area_rows:
        write_csv(os.path.join(args.output_dir, 'rsu_bev_area_manifest.csv'),
                  list(area_rows[0].keys()),
                  area_rows)
    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'rsu_bev_frame_summary.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    write_csv(os.path.join(args.output_dir, 'rsu_bev_eval_summary.csv'),
              list(summary_rows[0].keys()),
              summary_rows)
    write_run_metadata(args, coperception_params, timestamps, summary_rows[0])

    print('Wrote LGCP RSU BEV fusion outputs to %s' % args.output_dir)
    print('AP30=%s AP50=%s AP70=%s gt=%s pred_samples=%s' % (
        summary_rows[0]['ap_03'],
        summary_rows[0]['ap_05'],
        summary_rows[0]['ap_07'],
        summary_rows[0]['gt_total'],
        summary_rows[0]['pred_samples']))


def empty_frame_row(args, timestamp, reference_label, frame_plan,
                    skipped_empty, gt_box_tensor):
    return OrderedDict({
        'scenario_id': args.scenario_id,
        'timestamp': timestamp,
        'reference': reference_label,
        'reference_x': '',
        'reference_y': '',
        'reference_yaw': '',
        'planned_areas': len(frame_plan),
        'valid_leader_features': 0,
        'skipped_empty_area_packets': skipped_empty,
        'query_mode': args.query_mode,
        'packet_granularity': args.packet_granularity,
        'eval_scope': args.eval_scope,
        'member_upload_bytes': 0,
        'leader_own_area_bytes': 0,
        'leader_feature_full_bytes': 0,
        'leader_feature_sparse_cell_bytes': 0,
        'leader_nonzero_bev_cells': 0,
        'raw_pred_boxes': 0,
        'raw_gt_boxes': int(gt_box_tensor.shape[0]),
        'eval_pred_boxes': 0,
        'eval_gt_boxes': int(gt_box_tensor.shape[0]),
        'pred_boxes': 0,
        'pred_scores': 0,
        'gt_boxes': int(gt_box_tensor.shape[0]),
        'frame_tensor_file': '',
        'frame_tensor_npz_bytes': 0,
        'leader_feature_stack_shape': '',
        'fusion_input_shape': '',
        'spatial_features_2d_shape': '',
        'psm_shape': '',
        'rm_shape': '',
        'score_min': '',
        'score_mean': '',
        'score_max': '',
        'score_p95': '',
    })


def write_run_metadata(args, coperception_params, timestamps, summary):
    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'fusion_method': coperception_params['fusion_method'],
        'coperception_yaml': args.coperception_yaml,
        'reference_cav_id': args.reference_cav_id,
        'reference_pose': args.reference_pose,
        'reference_z_override': args.reference_z_override,
        'start_index': args.start_index,
        'max_frames': args.max_frames,
        'max_areas_per_frame': args.max_areas_per_frame,
        'grid_size_x': args.grid_size_x,
        'grid_size_y': args.grid_size_y,
        'query_mode': args.query_mode,
        'packet_granularity': args.packet_granularity,
        'leader_feature_dtype': args.leader_feature_dtype,
        'postprocess_score_threshold': args.postprocess_score_threshold,
        'eval_scope': args.eval_scope,
        'save_frame_tensors': args.save_frame_tensors,
        'timestamps': timestamps,
        'note': (
            'LGCP RSU BEV feature-fusion prototype. Member point slices are '
            'projected into a shared reference frame before PointPillar VFE; '
            'all leader scatter features are fused by the existing attentive '
            'BEV backbone.'),
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP RSU BEV Attentive Fusion\n\n')
        stream.write('This run crops member point clouds by LGCP area, ')
        stream.write('projects them into a shared reference/RSU lidar frame, ')
        stream.write('encodes one scatter BEV canvas per area leader, and ')
        stream.write('uses the existing PointPillar attentive backbone for ')
        stream.write('RSU-level feature fusion.\n\n')
        stream.write('- frames: `%s`\n' % summary['frames'])
        stream.write('- area rows: `%s`\n' % summary['area_rows'])
        stream.write('- query mode: `%s`\n' % summary['query_mode'])
        stream.write('- packet granularity: `%s`\n' %
                     summary['packet_granularity'])
        stream.write('- AP@0.3/AP@0.5/AP@0.7: `%s/%s/%s`\n' % (
            summary['ap_03'],
            summary['ap_05'],
            summary['ap_07']))
        stream.write('- member upload bytes: `%s`\n' %
                     summary['member_upload_bytes'])
        stream.write('- leader feature sparse-cell bytes: `%s`\n' %
                     summary['leader_feature_sparse_cell_bytes'])


if __name__ == '__main__':
    main()
