# -*- coding: utf-8 -*-
"""Associate SGCP uploaded points with missed GT object boxes.

This tool follows ``sgcp_head_box_diagnostics``. For dense target-grid misses,
it rebuilds the same SGCP constrained inputs and counts how many receiver/full
and uploaded/selected points actually fall inside the GT object's BEV box and
expanded neighborhoods. It helps distinguish "the selected grid contains
points" from "the detector received object-supporting points".
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
_OPENCOOD_ROOT = os.path.join(_REPO_ROOT, 'opencood')
if _OPENCOOD_ROOT not in sys.path:
    sys.path.insert(0, _OPENCOOD_ROOT)

from opencood.utils.transformation_utils import x_to_world

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.sensing.perception import sensor_transformation as st
from opencda.opencda_carla import Location, Rotation, Transform
from opencda.tools.offline_inference import (
    apply_sgcp_constraint,
    load_protocol,
    select_cav_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Count SGCP uploaded points inside missed GT boxes.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--failure-gt-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--resource-allocation',
                        default='perception_aware_potential_game')
    parser.add_argument('--clustering', default='coalition_game')
    parser.add_argument('--rho-th', type=float, default=3.0)
    parser.add_argument('--num-channels', type=int, default=10)
    parser.add_argument('--bandwidth-mhz', type=float, default=20.0)
    parser.add_argument('--head-rb-budget', type=int, default=2)
    parser.add_argument('--cav-count', type=int, default=None)
    parser.add_argument('--cav-ids', default=None)
    parser.add_argument('--object-ids', default=None)
    parser.add_argument('--max-rows', type=int, default=40)
    parser.add_argument('--min-nearest-head-points', type=int, default=30)
    parser.add_argument('--margins', default='0,1,2,4',
                        help='Comma-separated BEV expansion margins in m.')
    return parser.parse_args()


def read_failure_rows(path, object_ids=None, min_points=30, max_rows=40):
    wanted = None
    if object_ids:
        wanted = {item.strip() for item in object_ids.split(',')
                  if item.strip()}
    rows = []
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            if wanted is not None and row.get('object_id') not in wanted:
                continue
            if str(row.get('full_detected_method_missed', '0')) != '1':
                continue
            try:
                nearest_points = int(float(
                    row.get('nearest_head_covering_point_count', 0) or 0))
            except (TypeError, ValueError):
                nearest_points = 0
            if nearest_points < min_points:
                continue
            rows.append(row)
    rows = sorted(
        rows,
        key=lambda item: (
            item.get('timestamp', ''),
            item.get('object_id', ''),
            -int(float(item.get('nearest_head_covering_point_count', 0) or 0))))
    return rows[:max_rows] if max_rows > 0 else rows


def pose_to_transform(pose):
    return Transform(
        Location(pose[0], pose[1], pose[2]),
        Rotation(pitch=pose[5], yaw=pose[4], roll=pose[3]))


def local_to_world(points, lidar_pose):
    if points is None or points.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return st.lidar_local_to_global(points[:, :3], pose_to_transform(lidar_pose))


def bbox_center_world(vehicle):
    pose = vehicle_pose(vehicle)
    center = vehicle.get('center', [0.0, 0.0, 0.0])
    local = np.array([
        float(center[0]),
        float(center[1]),
        float(center[2]),
        1.0,
    ])
    return np.dot(x_to_world(pose), local)[:3]


def vehicle_pose(vehicle):
    location = vehicle.get('location', [0.0, 0.0, 0.0])
    angle = vehicle.get('angle', [0.0, 0.0, 0.0])
    return [
        float(location[0]),
        float(location[1]),
        float(location[2]),
        float(angle[0]),
        float(angle[1]),
        float(angle[2]),
    ]


def object_info(frame, object_id):
    vehicles = next(iter(frame.values()))['params'].get('vehicles', {})
    vehicle = vehicles.get(str(object_id))
    if vehicle is None:
        try:
            vehicle = vehicles.get(int(object_id))
        except (TypeError, ValueError):
            vehicle = None
    if vehicle is None:
        return None
    extent = vehicle.get('extent', [0.0, 0.0, 0.0])
    center = bbox_center_world(vehicle)
    pose = vehicle_pose(vehicle)
    return {
        'center': center,
        'yaw_rad': math.radians(float(pose[4])),
        'extent_x': float(extent[0]),
        'extent_y': float(extent[1]),
        'extent_z': float(extent[2]) if len(extent) > 2 else 2.0,
        'bp_id': vehicle.get('bp_id', ''),
    }


def count_points_in_box(points_world, info, margin=0.0, use_z=False):
    if points_world is None or points_world.size == 0 or info is None:
        return 0
    center = info['center']
    dx = points_world[:, 0] - center[0]
    dy = points_world[:, 1] - center[1]
    yaw = info['yaw_rad']
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    mask = (
        (np.abs(local_x) <= info['extent_x'] + margin) &
        (np.abs(local_y) <= info['extent_y'] + margin))
    if use_z and points_world.shape[1] >= 3:
        local_z = points_world[:, 2] - center[2]
        mask = mask & (np.abs(local_z) <= info['extent_z'] + margin)
    return int(np.count_nonzero(mask))


def point_counts_for_frame(args, dataset, protocol, scenario_id, timestamp,
                           miss_rows, margins):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=args.ego_cav_id,
        cav_ids=select_cav_ids(
            dataset,
            scenario_id,
            ego_cav_id=args.ego_cav_id,
            cav_count=args.cav_count,
            cav_ids=args.cav_ids))
    frame_items = apply_sgcp_constraint(
        frame,
        protocol,
        args.ego_cav_id,
        args.resource_allocation,
        'all-cluster-heads',
        t_min_stab=None,
        clustering=args.clustering,
        n_max=None,
        rho_th=args.rho_th,
        num_channels=args.num_channels,
        bandwidth_mhz=args.bandwidth_mhz,
        head_rb_budget=args.head_rb_budget)
    by_receiver = {
        int(metadata['receiver_id']): (eval_frame, metadata)
        for eval_frame, metadata in frame_items
    }
    raw_world_points = {
        int(cav_id): local_to_world(
            cav['lidar_np'],
            cav['params']['lidar_pose'])
        for cav_id, cav in frame.items()
    }
    full_reference_points = (
        np.concatenate(list(raw_world_points.values()), axis=0)
        if raw_world_points
        else np.empty((0, 3), dtype=np.float32))
    rows = []
    for miss in miss_rows:
        object_id = str(miss.get('object_id', ''))
        info = object_info(frame, object_id)
        if info is None:
            continue
        nearest_head = int(miss['nearest_head'])
        nearest_cav = int(miss['nearest_cav'])
        eval_frame, metadata = by_receiver.get(nearest_head, ({}, {}))

        receiver_points = np.empty((0, 3), dtype=np.float32)
        uploaded_points = []
        nearest_cav_uploaded_points = np.empty((0, 3), dtype=np.float32)
        for cav_id, cav in eval_frame.items():
            cav_id = int(cav_id)
            points_world = local_to_world(
                cav['lidar_np'],
                cav['params']['lidar_pose'])
            if cav_id == nearest_head:
                receiver_points = points_world
            else:
                uploaded_points.append(points_world)
                if cav_id == nearest_cav:
                    nearest_cav_uploaded_points = points_world
        if uploaded_points:
            uploaded_points = np.concatenate(uploaded_points, axis=0)
        else:
            uploaded_points = np.empty((0, 3), dtype=np.float32)
        total_points = (
            np.concatenate([receiver_points, uploaded_points], axis=0)
            if receiver_points.size or uploaded_points.size
            else np.empty((0, 3), dtype=np.float32))
        nearest_cav_raw = raw_world_points.get(
            nearest_cav,
            np.empty((0, 3), dtype=np.float32))

        base = {
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'object_id': object_id,
            'bp_id': info['bp_id'],
            'object_grid_id': miss.get('object_grid_id', ''),
            'nearest_head': nearest_head,
            'nearest_cav': nearest_cav,
            'nearest_head_covering_point_count': miss.get(
                'nearest_head_covering_point_count', ''),
            'nearest_cav_object_grid_points': miss.get(
                'nearest_cav_object_grid_points', ''),
            'source_cav_ids': ';'.join(
                str(item) for item in metadata.get('source_cav_ids', [])),
            'uploaded_source_ids': ';'.join(
                str(item) for item in metadata.get('source_cav_ids', [])[1:]),
            'communication_bytes': metadata.get('communication_bytes', ''),
            'receiver_total_points': int(receiver_points.shape[0]),
            'uploaded_total_points': int(uploaded_points.shape[0]),
            'nearest_cav_uploaded_total_points': int(
                nearest_cav_uploaded_points.shape[0]),
            'nearest_cav_raw_total_points': int(nearest_cav_raw.shape[0]),
            'full_reference_total_points': int(full_reference_points.shape[0]),
            'full_reference_best_iou': miss.get('full_reference_best_iou', ''),
            'method_best_iou': miss.get('method_best_iou', ''),
        }
        for margin in margins:
            suffix = ('m%s' % str(margin).replace('.', 'p'))
            receiver_box_points = count_points_in_box(
                receiver_points,
                info,
                margin=margin)
            uploaded_box_points = count_points_in_box(
                uploaded_points,
                info,
                margin=margin)
            total_box_points = count_points_in_box(
                total_points,
                info,
                margin=margin)
            nearest_cav_uploaded_box_points = count_points_in_box(
                nearest_cav_uploaded_points,
                info,
                margin=margin)
            nearest_cav_raw_box_points = count_points_in_box(
                nearest_cav_raw,
                info,
                margin=margin)
            full_reference_box_points = count_points_in_box(
                full_reference_points,
                info,
                margin=margin)
            raw_cav_box_counts = {
                cav_id: count_points_in_box(points, info, margin=margin)
                for cav_id, points in raw_world_points.items()
            }
            best_raw_cav_id, best_raw_cav_points = max(
                raw_cav_box_counts.items(),
                key=lambda item: item[1])
            sorted_raw_counts = sorted(
                raw_cav_box_counts.values(),
                reverse=True)
            nearest_cav_raw_rank = (
                1 + sum(
                    1 for value in sorted_raw_counts
                    if value > nearest_cav_raw_box_points))
            base['receiver_box_points_%s' % suffix] = receiver_box_points
            base['uploaded_box_points_%s' % suffix] = uploaded_box_points
            base['total_box_points_%s' % suffix] = total_box_points
            base['nearest_cav_uploaded_box_points_%s' % suffix] = (
                nearest_cav_uploaded_box_points)
            base['nearest_cav_raw_box_points_%s' % suffix] = (
                nearest_cav_raw_box_points)
            base['full_reference_box_points_%s' % suffix] = (
                full_reference_box_points)
            base['sgcp_full_box_point_ratio_%s' % suffix] = (
                round(float(total_box_points) / full_reference_box_points, 4)
                if full_reference_box_points > 0
                else '')
            base['best_raw_cav_id_%s' % suffix] = best_raw_cav_id
            base['best_raw_cav_box_points_%s' % suffix] = best_raw_cav_points
            base['nearest_cav_raw_rank_%s' % suffix] = nearest_cav_raw_rank
        rows.append(base)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        with open(path, 'w', newline='') as stream:
            stream.write('')
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    margins = [
        float(item.strip()) for item in args.margins.split(',')
        if item.strip()
    ]
    dataset = OPV2VFrameDataset(args.dataset_root)
    protocol = load_protocol(dataset, args.scenario_id)
    misses = read_failure_rows(
        args.failure_gt_csv,
        object_ids=args.object_ids,
        min_points=args.min_nearest_head_points,
        max_rows=args.max_rows)
    by_timestamp = defaultdict(list)
    for row in misses:
        by_timestamp[row['timestamp']].append(row)
    output_rows = []
    for timestamp, rows in sorted(by_timestamp.items()):
        output_rows.extend(point_counts_for_frame(
            args,
            dataset,
            protocol,
            args.scenario_id,
            timestamp,
            rows,
            margins))
        print('associated timestamp=%s misses=%d total_rows=%d' % (
            timestamp,
            len(rows),
            len(output_rows)))
    write_csv(args.output_csv, output_rows)
    if output_rows:
        zero_exact = sum(
            1 for row in output_rows
            if int(row.get('total_box_points_m0p0', 0)) == 0)
        zero_m2 = sum(
            1 for row in output_rows
            if int(row.get('total_box_points_m2p0', 0)) == 0)
        print('wrote %d rows to %s' % (len(output_rows), args.output_csv))
        print('zero_total_box_points_margin0=%d/%d margin2=%d/%d' % (
            zero_exact,
            len(output_rows),
            zero_m2,
            len(output_rows)))
    else:
        print('wrote 0 rows to %s' % args.output_csv)


if __name__ == '__main__':
    main()
