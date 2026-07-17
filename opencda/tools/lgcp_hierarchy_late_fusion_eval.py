# -*- coding: utf-8 -*-
"""
Evaluate LGCP box-level local-to-global hierarchy with OpenCOOD inference.

This is the first model-calling hierarchy adapter for LGCP. It differs from
lgcp_hierarchy_aggregation_eval.py: leader local results are produced by
running OpenCOOD on each area-task group, and RSU global results are produced
by late-fusing the area-filtered leader predictions in world coordinates.

It is still not neural feature slicing. The fusion unit here is detection
boxes, so the result should be reported as a box-level hierarchy ablation.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.lgcp_area_confidence_eval import (
    box_area_ids,
    corners_to_world,
    load_lgcp_config,
    slice_tensor_by_area,
)
from opencda.tools.offline_inference import load_coperception_params
from opencood.utils import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run LGCP box-level hierarchy late-fusion evaluation.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True,
                        help='area_assignment_plan.csv from hierarchy plan.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--lgcp-yaml',
                        default='opencda/scenario_testing/config_yaml/lgcp_carla.yaml')
    parser.add_argument('--fusion-method', default=None,
                        help='OpenCOOD fusion method. Defaults to coperception yaml.')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index in assignment-plan timestamp order.')
    parser.add_argument('--max-frames', type=int, default=1,
                        help='Number of frames to evaluate. Use 0 for all.')
    parser.add_argument('--max-areas-per-frame', type=int, default=0,
                        help='Optional smoke cap; 0 means all planned areas.')
    parser.add_argument('--late-nms-thresh', type=float, default=0.15)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value):
    if value is None:
        return ''
    return '%.6f' % float(value)


def parse_members(value):
    return [item for item in str(value).split(';') if item != '']


def group_assignment_rows(rows):
    by_timestamp = defaultdict(list)
    for row in rows:
        by_timestamp[row['timestamp']].append(row)
    return OrderedDict(
        (timestamp, by_timestamp[timestamp])
        for timestamp in sorted(by_timestamp.keys()))


def selected_timestamps(grouped, start_index, max_frames):
    timestamps = list(grouped.keys())
    timestamps = timestamps[start_index:]
    if max_frames:
        timestamps = timestamps[:max_frames]
    return timestamps


def run_inference(manager, dataset, scenario_id, timestamp, ego_id, cav_ids):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=ego_id,
        cav_ids=cav_ids)
    ego = next(cav for cav in frame.values() if cav['ego'])
    ego_lidar_pose = ego['params']['lidar_pose']
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        ego_lidar_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    batch_data = manager.to_device(output_dict)
    ret = manager.inference(
        batch_data,
        with_stats=False,
        return_object_ids=False)
    return {
        'pred_box_tensor': ret[0],
        'pred_score': ret[1],
        'gt_box_tensor': ret[2],
        'ego_lidar_pose': ego_lidar_pose,
    }


def tensor_to_world(tensor, lidar_pose):
    world = corners_to_world(tensor, lidar_pose)
    if world is None:
        return None
    return torch.as_tensor(world, dtype=torch.float32)


def filter_tensor_by_area_world(tensor_world, score, area_id, config):
    if tensor_world is None:
        return None, None
    area_ids = box_area_ids(
        tensor_world.detach().cpu().numpy(),
        config)
    area_tensor = slice_tensor_by_area(tensor_world, area_ids, area_id)
    if score is None or area_tensor is None:
        return area_tensor, None
    indices = [idx for idx, value in enumerate(area_ids) if value == area_id]
    area_score = score[torch.as_tensor(
        indices, dtype=torch.long, device=score.device)].detach().cpu()
    return area_tensor, area_score


def empty_like_box(template):
    if template is None:
        return torch.zeros((0, 8, 3), dtype=torch.float32)
    return template.new_zeros((0,) + tuple(template.shape[1:]))


def update_stats(result_stat, pred_box_tensor, pred_score, gt_box_tensor):
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


def safe_cat(tensors, template=None):
    tensors = [tensor for tensor in tensors if tensor is not None]
    tensors = [tensor for tensor in tensors if int(tensor.shape[0]) > 0]
    if not tensors:
        return empty_like_box(template)
    return torch.cat(tensors, dim=0)


def safe_score_cat(scores):
    scores = [score for score in scores if score is not None]
    scores = [score for score in scores if int(score.shape[0]) > 0]
    if not scores:
        return None
    return torch.cat(scores, dim=0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assignment_rows = read_csv(args.assignment_plan)
    grouped = group_assignment_rows(assignment_rows)
    timestamps = selected_timestamps(
        grouped,
        args.start_index,
        args.max_frames)

    lgcp_config = load_lgcp_config(args.lgcp_yaml)
    dataset = OPV2VFrameDataset(args.dataset_root)
    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)

    inference_cache = {}
    leader_rows = []
    frame_rows = []
    result_stat = {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }

    for frame_index, timestamp in enumerate(timestamps, start=1):
        frame_plan = grouped[timestamp]
        if args.max_areas_per_frame:
            frame_plan = frame_plan[:args.max_areas_per_frame]

        area_pred_tensors = []
        area_scores = []
        area_gt_tensors = []
        group_calls = set()
        box_template = None

        for row in frame_plan:
            area_id = row['area_id']
            leader_id = str(row['leader_id'])
            members = parse_members(row['group_members'])
            if leader_id not in members:
                members = [leader_id] + members
            members = list(OrderedDict((str(member), None)
                                       for member in members).keys())
            cache_key = (timestamp, leader_id, tuple(members))
            if cache_key not in inference_cache:
                inference_cache[cache_key] = run_inference(
                    manager,
                    dataset,
                    args.scenario_id,
                    timestamp,
                    leader_id,
                    members)
            group_calls.add(cache_key)
            ret = inference_cache[cache_key]

            pred_world = tensor_to_world(
                ret['pred_box_tensor'],
                ret['ego_lidar_pose'])
            gt_world = tensor_to_world(
                ret['gt_box_tensor'],
                ret['ego_lidar_pose'])
            if box_template is None:
                box_template = gt_world if gt_world is not None else pred_world

            area_pred, area_score = filter_tensor_by_area_world(
                pred_world,
                ret['pred_score'],
                area_id,
                lgcp_config)
            area_gt, _ = filter_tensor_by_area_world(
                gt_world,
                None,
                area_id,
                lgcp_config)
            if area_gt is None:
                area_gt = empty_like_box(gt_world)

            if area_pred is not None and int(area_pred.shape[0]) > 0:
                area_pred_tensors.append(area_pred)
                area_scores.append(area_score)
            if area_gt is not None and int(area_gt.shape[0]) > 0:
                area_gt_tensors.append(area_gt)

            leader_rows.append(OrderedDict({
                'scenario_id': args.scenario_id,
                'timestamp': timestamp,
                'area_id': area_id,
                'leader_id': leader_id,
                'group_members': ';'.join(members),
                'group_size': len(members),
                'group_confidence': row.get('group_confidence', ''),
                'area_pred_count': 0 if area_pred is None
                else int(area_pred.shape[0]),
                'area_gt_count': 0 if area_gt is None else int(area_gt.shape[0]),
            }))

        fused_pred = None
        fused_score = None
        if area_pred_tensors:
            fused_pred, fused_score = manager.naive_late_fusion(
                area_pred_tensors,
                area_scores,
                iou_threshold=args.late_nms_thresh)
        fused_gt = safe_cat(area_gt_tensors, box_template)
        if int(fused_gt.shape[0]) > 0:
            update_stats(result_stat, fused_pred, fused_score, fused_gt)

        frame_rows.append(OrderedDict({
            'scenario_id': args.scenario_id,
            'timestamp': timestamp,
            'planned_areas': len(frame_plan),
            'unique_group_inference_calls': len(group_calls),
            'leader_local_pred_boxes': sum(
                int(tensor.shape[0]) for tensor in area_pred_tensors),
            'leader_local_gt_boxes': sum(
                int(tensor.shape[0]) for tensor in area_gt_tensors),
            'rsu_fused_pred_boxes': 0 if fused_pred is None
            else int(fused_pred.shape[0]),
            'rsu_fused_gt_boxes': int(fused_gt.shape[0]),
        }))
        print('frame=%s/%s timestamp=%s areas=%s groups=%s pred=%s gt=%s' % (
            frame_index,
            len(timestamps),
            timestamp,
            len(frame_plan),
            len(group_calls),
            frame_rows[-1]['rsu_fused_pred_boxes'],
            frame_rows[-1]['rsu_fused_gt_boxes']))

    summary_rows = [OrderedDict({
        'frames': len(timestamps),
        'assignment_rows': len(leader_rows),
        'cached_group_inference_calls': len(inference_cache),
        'planned_areas_mean': format_float(
            np.mean([float(row['planned_areas']) for row in frame_rows])
            if frame_rows else 0.0),
        'rsu_fused_pred_boxes_mean': format_float(
            np.mean([float(row['rsu_fused_pred_boxes'])
                     for row in frame_rows]) if frame_rows else 0.0),
        'rsu_fused_gt_boxes_mean': format_float(
            np.mean([float(row['rsu_fused_gt_boxes'])
                     for row in frame_rows]) if frame_rows else 0.0),
        'ap_03': calculate_ap_safe(result_stat, 0.3),
        'ap_05': calculate_ap_safe(result_stat, 0.5),
        'ap_07': calculate_ap_safe(result_stat, 0.7),
        'gt_total': result_stat[0.5]['gt'],
        'pred_samples': len(result_stat[0.5]['tp']),
    })]

    write_csv(os.path.join(args.output_dir, 'leader_local_predictions.csv'),
              ['scenario_id', 'timestamp', 'area_id', 'leader_id',
               'group_members', 'group_size', 'group_confidence',
               'area_pred_count', 'area_gt_count'],
              leader_rows)
    write_csv(os.path.join(args.output_dir, 'rsu_global_frame_summary.csv'),
              ['scenario_id', 'timestamp', 'planned_areas',
               'unique_group_inference_calls', 'leader_local_pred_boxes',
               'leader_local_gt_boxes', 'rsu_fused_pred_boxes',
               'rsu_fused_gt_boxes'],
              frame_rows)
    write_csv(os.path.join(args.output_dir, 'rsu_global_eval_summary.csv'),
              ['frames', 'assignment_rows', 'cached_group_inference_calls',
               'planned_areas_mean', 'rsu_fused_pred_boxes_mean',
               'rsu_fused_gt_boxes_mean', 'ap_03', 'ap_05', 'ap_07',
               'gt_total', 'pred_samples'],
              summary_rows)

    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'lgcp_yaml': os.path.abspath(args.lgcp_yaml),
        'fusion_method': coperception_params['fusion_method'],
        'start_index': args.start_index,
        'max_frames': args.max_frames,
        'max_areas_per_frame': args.max_areas_per_frame,
        'late_nms_thresh': args.late_nms_thresh,
        'timestamps': timestamps,
        'note': (
            'Box-level hierarchy late fusion. This calls OpenCOOD but does '
            'not perform neural feature slicing.'),
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Box-Level Hierarchy Late Fusion\n\n')
        stream.write('This run evaluates LGCP hierarchy by running OpenCOOD ')
        stream.write('for each leader area-task group, slicing predictions ')
        stream.write('by LGCP area in world coordinates, and late-fusing ')
        stream.write('leader local predictions into an RSU global result.\n\n')
        stream.write('It is a model-calling box-level hierarchy ablation, ')
        stream.write('not neural feature tensor slicing.\n\n')
        stream.write('- frames: `%d`\n' % len(timestamps))
        stream.write('- assignment rows: `%d`\n' % len(leader_rows))
        stream.write('- cached group inference calls: `%d`\n' %
                     len(inference_cache))
        stream.write('- AP@0.5: `%s`\n' % summary_rows[0]['ap_05'])

    print('Wrote LGCP hierarchy late-fusion eval to %s' % args.output_dir)
    print('AP30=%s AP50=%s AP70=%s gt=%s pred_samples=%s' % (
        summary_rows[0]['ap_03'],
        summary_rows[0]['ap_05'],
        summary_rows[0]['ap_07'],
        summary_rows[0]['gt_total'],
        summary_rows[0]['pred_samples']))


if __name__ == '__main__':
    main()
