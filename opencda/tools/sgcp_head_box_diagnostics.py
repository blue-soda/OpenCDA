# -*- coding: utf-8 -*-
"""Diagnose SGCP missed objects before inter-cluster late fusion.

The failure diagnostics showed a secondary bucket where the nearest cluster
head receives dense points for a target grid, but the final late-fused result
still misses the object. This tool reruns the same SGCP constrained inference
for selected missed objects and reports, per cluster head, the best detector
box IoU/score before inter-cluster late fusion.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
_OPENCOOD_ROOT = os.path.join(_REPO_ROOT, 'opencood')
if _OPENCOOD_ROOT not in sys.path:
    sys.path.insert(0, _OPENCOOD_ROOT)

from opencood.utils import common_utils

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import (
    apply_sgcp_constraint,
    is_empty_pillar_error,
    load_coperception_params,
    load_protocol,
    run_opencood_inference,
    select_cav_ids,
    tensor_to_numpy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump per-head detector boxes for missed SGCP objects.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--failure-gt-csv', required=True,
                        help='gt_objects.csv from sgcp_failure_diagnostics.')
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
    parser.add_argument('--object-ids', default=None,
                        help='Optional comma-separated object ids.')
    parser.add_argument('--max-rows', type=int, default=40)
    parser.add_argument('--min-nearest-head-points', type=int, default=30)
    parser.add_argument('--iou-thresh', type=float, default=0.5)
    parser.add_argument('--sgcp-late-nms-thresh', type=float, default=0.15)
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


def best_iou_and_score(gt_box, pred_boxes, pred_scores):
    if gt_box is None or pred_boxes is None or pred_scores is None:
        return 0.0, ''
    pred_np = tensor_to_numpy(pred_boxes)
    score_np = tensor_to_numpy(pred_scores)
    if pred_np is None or score_np is None or pred_np.shape[0] == 0:
        return 0.0, ''
    gt_np = np.expand_dims(tensor_to_numpy(gt_box), axis=0)
    pred_polygons = list(common_utils.convert_format(pred_np))
    gt_polygons = list(common_utils.convert_format(gt_np))
    if not pred_polygons or not gt_polygons:
        return 0.0, ''
    best_iou = 0.0
    best_score = ''
    gt_polygon = gt_polygons[0]
    for pred_index, pred_polygon in enumerate(pred_polygons):
        ious = common_utils.compute_iou(pred_polygon, [gt_polygon])
        if len(ious) == 0:
            continue
        iou = float(ious[0])
        if iou > best_iou:
            best_iou = iou
            best_score = '%.6f' % float(score_np[pred_index])
    return best_iou, best_score


def gt_lookup_from_canonical(ret):
    gt_tensor = ret[2]
    object_ids = ret[3] if len(ret) > 3 else None
    if gt_tensor is None or object_ids is None:
        return {}
    gt_np = tensor_to_numpy(gt_tensor)
    if gt_np is None:
        return {}
    lookup = {}
    for index, object_id in enumerate(object_ids):
        if index >= gt_tensor.shape[0]:
            break
        lookup[str(object_id)] = gt_tensor[index]
    return lookup


def run_timestamp(args, dataset, protocol, manager, scenario_id, timestamp,
                  miss_rows):
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
    original_ego = next(cav for cav in frame.values() if cav['ego'])
    target_ego_lidar_pose = original_ego['params']['lidar_pose']
    canonical_ret = run_opencood_inference(
        manager,
        frame,
        target_ego_lidar_pose)
    gt_lookup = gt_lookup_from_canonical(canonical_ret)
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

    receiver_results = {}
    pred_tensors = []
    pred_scores = []
    for eval_frame, metadata in frame_items:
        receiver_id = int(metadata['receiver_id'])
        try:
            ret = run_opencood_inference(
                manager,
                eval_frame,
                target_ego_lidar_pose)
        except RuntimeError as error:
            if not is_empty_pillar_error(error):
                raise
            receiver_results[receiver_id] = {
                'pred_boxes': None,
                'pred_scores': None,
                'gt_boxes': None,
                'metadata': metadata,
                'skipped': 'empty_pillars',
            }
            continue
        pred_box_tensor, pred_score, gt_box_tensor = ret[0:3]
        receiver_results[receiver_id] = {
            'pred_boxes': pred_box_tensor,
            'pred_scores': pred_score,
            'gt_boxes': gt_box_tensor,
            'metadata': metadata,
            'skipped': '',
        }
        if pred_box_tensor is not None and pred_score is not None:
            pred_tensors.append(pred_box_tensor)
            pred_scores.append(pred_score)

    fused_pred, fused_score = manager.naive_late_fusion(
        pred_tensors,
        pred_scores,
        iou_threshold=args.sgcp_late_nms_thresh)

    output_rows = []
    for miss in miss_rows:
        object_id = str(miss.get('object_id', ''))
        gt_box = gt_lookup.get(object_id)
        if gt_box is None:
            continue
        fused_iou, fused_score_value = best_iou_and_score(
            gt_box,
            fused_pred,
            fused_score)
        full_iou, full_score = best_iou_and_score(
            gt_box,
            canonical_ret[0],
            canonical_ret[1])
        nearest_head = int(miss.get('nearest_head'))
        for receiver_id, result in sorted(receiver_results.items()):
            metadata = result['metadata']
            head_iou, head_score = best_iou_and_score(
                gt_box,
                result['pred_boxes'],
                result['pred_scores'])
            output_rows.append({
                'scenario_id': scenario_id,
                'timestamp': timestamp,
                'object_id': object_id,
                'bp_id': miss.get('bp_id', ''),
                'object_grid_id': miss.get('object_grid_id', ''),
                'nearest_head': nearest_head,
                'receiver_id': receiver_id,
                'receiver_is_nearest_head': int(receiver_id == nearest_head),
                'nearest_head_covering_point_count': miss.get(
                    'nearest_head_covering_point_count', ''),
                'scheduled_covering_links': miss.get(
                    'scheduled_covering_links', ''),
                'source_cav_ids': ';'.join(
                    str(item) for item in metadata.get(
                        'source_cav_ids', [])),
                'uploaded_source_ids': ';'.join(
                    str(item) for item in metadata.get(
                        'source_cav_ids', [])[1:]),
                'communication_bytes': metadata.get('communication_bytes', ''),
                'receiver_pred_boxes': (
                    0 if result['pred_boxes'] is None else
                    int(result['pred_boxes'].shape[0])),
                'receiver_gt_boxes': (
                    0 if result['gt_boxes'] is None else
                    int(result['gt_boxes'].shape[0])),
                'receiver_best_iou': '%.6f' % head_iou,
                'receiver_best_score': head_score,
                'receiver_matched': int(head_iou >= args.iou_thresh),
                'fused_best_iou': '%.6f' % fused_iou,
                'fused_best_score': fused_score_value,
                'fused_matched': int(fused_iou >= args.iou_thresh),
                'full_reference_best_iou': '%.6f' % full_iou,
                'full_reference_best_score': full_score,
                'skipped': result['skipped'],
            })
    return output_rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        'scenario_id',
        'timestamp',
        'object_id',
        'bp_id',
        'object_grid_id',
        'nearest_head',
        'receiver_id',
        'receiver_is_nearest_head',
        'nearest_head_covering_point_count',
        'scheduled_covering_links',
        'source_cav_ids',
        'uploaded_source_ids',
        'communication_bytes',
        'receiver_pred_boxes',
        'receiver_gt_boxes',
        'receiver_best_iou',
        'receiver_best_score',
        'receiver_matched',
        'fused_best_iou',
        'fused_best_score',
        'fused_matched',
        'full_reference_best_iou',
        'full_reference_best_score',
        'skipped',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)
    protocol = load_protocol(dataset, args.scenario_id)
    coperception_params = load_coperception_params(None)
    manager = OpenCOODManager(coperception_params)
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
        output_rows.extend(run_timestamp(
            args,
            dataset,
            protocol,
            manager,
            args.scenario_id,
            timestamp,
            rows))
        print('diagnosed timestamp=%s misses=%d total_rows=%d' % (
            timestamp,
            len(rows),
            len(output_rows)))
    write_csv(args.output_csv, output_rows)
    nearest_rows = [
        row for row in output_rows
        if int(row['receiver_is_nearest_head']) == 1
    ]
    nearest_matched = sum(
        int(row['receiver_matched']) for row in nearest_rows)
    fused_matched = sum(int(row['fused_matched']) for row in nearest_rows)
    print('wrote %d rows to %s' % (len(output_rows), args.output_csv))
    print('nearest_head_matched=%d/%d fused_matched=%d/%d' % (
        nearest_matched,
        len(nearest_rows),
        fused_matched,
        len(nearest_rows)))


if __name__ == '__main__':
    main()
