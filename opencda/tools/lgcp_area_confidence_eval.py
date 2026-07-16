# -*- coding: utf-8 -*-
"""
Export LGCP area-level confidence records from an OpenCDA data dump.

This first pass validates the offline area/grid data path:
  - transform dumped LiDAR points into world coordinates,
  - assign points and GT objects to LGCP ROI grids,
  - export per-agent density-based confidence records,
  - optionally slice OpenCOOD predictions by LGCP area and compute quality
    summaries.
"""

import argparse
import csv
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import load_coperception_params
from opencood.utils import eval_utils
from opencood.utils.transformation_utils import x_to_world


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export LGCP area confidence records from a CARLA dump.')
    parser.add_argument('--dataset-root', required=True,
                        help='Root folder containing scenario subfolders.')
    parser.add_argument('--scenario-id', default=None,
                        help='Scenario folder name. Defaults to the first one.')
    parser.add_argument('--lgcp-yaml',
                        default='opencda/scenario_testing/config_yaml/lgcp_carla.yaml',
                        help='Scenario YAML containing lgcp.roi metadata.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for CSV outputs and run notes.')
    parser.add_argument('--max-frames', type=int, default=3,
                        help='Number of frames to export. Use 0 for all frames.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from within the scenario.')
    parser.add_argument('--density-threshold', type=float, default=None,
                        help='Override density threshold. Defaults to YAML lidar threshold.')
    parser.add_argument('--grid-size-x', type=float, default=None,
                        help='Override LGCP ROI grid width in meters.')
    parser.add_argument('--grid-size-y', type=float, default=None,
                        help='Override LGCP ROI grid height in meters.')
    parser.add_argument('--include-empty', action='store_true',
                        help='Write all area-agent rows, including empty cells.')
    parser.add_argument('--with-inference', action='store_true',
                        help='Run OpenCOOD and export area_quality.csv.')
    parser.add_argument('--fusion-method', default=None,
                        help='Override coperception fusion method.')
    parser.add_argument('--coperception-yaml', default=None,
                        help='Path to enable_coperception.yaml.')
    parser.add_argument('--ego-cav-id', default='1',
                        help='Ego CAV id used as OpenCOOD reference.')
    return parser.parse_args()


def load_lgcp_config(path):
    with open(path, 'r') as stream:
        params = yaml.safe_load(stream)

    lgcp = params['lgcp']
    roi = lgcp['roi']
    threshold = params['vehicle_base']['sensing']['perception']['lidar'].get(
        'density_threshold', 2.0)
    return {
        'center': tuple(float(v) for v in roi['center']),
        'size': tuple(float(v) for v in roi['size']),
        'grid_size': tuple(float(v) for v in roi['grid_size']),
        'delta_g': float(lgcp.get('area_confidence', {}).get('delta_g', 0.0)),
        'density_threshold': float(threshold),
    }


def area_bounds(config):
    cx, cy = config['center']
    sx, sy = config['size']
    return cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0


def area_id_from_xy(x, y, config):
    x_min, x_max, y_min, y_max = area_bounds(config)
    if x < x_min or x >= x_max or y < y_min or y >= y_max:
        return None

    gx, gy = config['grid_size']
    ix = int(math.floor((x - x_min) / gx))
    iy = int(math.floor((y - y_min) / gy))
    return '%d_%d' % (ix, iy)


def area_center(area_id, config):
    ix_str, iy_str = area_id.split('_')
    ix, iy = int(ix_str), int(iy_str)
    x_min, _, y_min, _ = area_bounds(config)
    gx, gy = config['grid_size']
    return x_min + (ix + 0.5) * gx, y_min + (iy + 0.5) * gy


def list_area_ids(config):
    sx, sy = config['size']
    gx, gy = config['grid_size']
    nx = int(math.ceil(sx / gx))
    ny = int(math.ceil(sy / gy))
    return ['%d_%d' % (ix, iy) for ix in range(nx) for iy in range(ny)]


def pcd_to_world_xy(points, lidar_pose):
    x0, y0 = float(lidar_pose[0]), float(lidar_pose[1])
    yaw = math.radians(float(lidar_pose[4]))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    local_x = points[:, 0]
    local_y = points[:, 1]
    world_x = x0 + cos_yaw * local_x - sin_yaw * local_y
    world_y = y0 + sin_yaw * local_x + cos_yaw * local_y
    return world_x, world_y


def count_points_by_area(points, lidar_pose, config):
    world_x, world_y = pcd_to_world_xy(points, lidar_pose)
    x_min, x_max, y_min, y_max = area_bounds(config)
    gx, gy = config['grid_size']
    inside = (
        (world_x >= x_min) & (world_x < x_max) &
        (world_y >= y_min) & (world_y < y_max)
    )

    counts = Counter()
    if not np.any(inside):
        return counts

    ix = np.floor((world_x[inside] - x_min) / gx).astype(np.int32)
    iy = np.floor((world_y[inside] - y_min) / gy).astype(np.int32)
    for area in zip(ix, iy):
        counts['%d_%d' % area] += 1
    return counts


def count_gt_by_area(frame, config):
    ego = next(cav for cav in frame.values() if cav['ego'])
    vehicles = ego['params'].get('vehicles', {})
    counts = Counter()
    for vehicle in vehicles.values():
        location = vehicle.get('location')
        if not location:
            continue
        area_id = area_id_from_xy(float(location[0]), float(location[1]),
                                  config)
        if area_id is not None:
            counts[area_id] += 1
    return counts


def corners_to_world(corners, lidar_pose):
    if corners is None:
        return None
    if isinstance(corners, torch.Tensor):
        corners_np = corners.detach().cpu().numpy()
    else:
        corners_np = np.asarray(corners)
    if corners_np.size == 0:
        return corners_np.reshape(0, 8, 3)

    matrix = x_to_world(lidar_pose)
    flat = corners_np.reshape(-1, 3)
    flat_h = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    world = np.dot(matrix, flat_h.T).T[:, :3]
    return world.reshape(corners_np.shape)


def box_area_ids(corners_world, config):
    if corners_world is None or corners_world.shape[0] == 0:
        return []
    centers = np.mean(corners_world[:, :, :2], axis=1)
    return [area_id_from_xy(center[0], center[1], config)
            for center in centers]


def slice_tensor_by_area(tensor, area_ids, area_id):
    if tensor is None:
        return None
    indices = [idx for idx, value in enumerate(area_ids) if value == area_id]
    if not indices:
        return None
    return tensor[torch.as_tensor(indices, dtype=torch.long, device=tensor.device)]


def area_quality_rows(pred_box_tensor, pred_score, gt_box_tensor, ego_pose,
                      scenario_id, timestamp, area_ids, config, area_stats=None):
    pred_world = corners_to_world(pred_box_tensor, ego_pose)
    gt_world = corners_to_world(gt_box_tensor, ego_pose)
    pred_area_ids = box_area_ids(pred_world, config)
    gt_area_ids = box_area_ids(gt_world, config)
    rows = []

    for area_id in area_ids:
        area_pred = slice_tensor_by_area(pred_box_tensor, pred_area_ids, area_id)
        area_gt = slice_tensor_by_area(gt_box_tensor, gt_area_ids, area_id)
        if area_gt is None:
            area_gt = gt_box_tensor.new_zeros((0,) + tuple(gt_box_tensor.shape[1:]))
        if pred_score is None or area_pred is None:
            area_score = None
            score_values = np.asarray([], dtype=np.float64)
        else:
            pred_indices = [idx for idx, value in enumerate(pred_area_ids)
                            if value == area_id]
            area_score = pred_score[torch.as_tensor(
                pred_indices, dtype=torch.long, device=pred_score.device)]
            score_values = area_score.detach().cpu().numpy().astype(np.float64)

        gt_count = int(area_gt.shape[0])
        pred_count = 0 if area_pred is None else int(area_pred.shape[0])
        if gt_count == 0 and pred_count == 0:
            continue

        row = OrderedDict({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'area_id': area_id,
            'pred_count': pred_count,
            'gt_count': gt_count,
            'score_mean': '',
            'score_max': '',
            'score_top2_mean': '',
            'score_top3_mean': '',
        })
        if score_values.size > 0:
            sorted_scores = np.sort(score_values)[::-1]
            row['score_mean'] = format_float(float(np.mean(score_values)))
            row['score_max'] = format_float(float(np.max(score_values)))
            row['score_top2_mean'] = format_float(
                float(np.mean(sorted_scores[:min(2, sorted_scores.size)])))
            row['score_top3_mean'] = format_float(
                float(np.mean(sorted_scores[:min(3, sorted_scores.size)])))

        for iou in (0.3, 0.5, 0.7):
            stat = {iou: {'tp': [], 'fp': [], 'gt': 0}}
            eval_utils.calculate_tp_fp(area_pred, area_score, area_gt, stat, iou)
            if area_stats is not None:
                key = area_id
                if key not in area_stats:
                    area_stats[key] = {
                        0.3: {'tp': [], 'fp': [], 'gt': 0},
                        0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0},
                    }
                area_stats[key][iou]['tp'].extend(stat[iou]['tp'])
                area_stats[key][iou]['fp'].extend(stat[iou]['fp'])
                area_stats[key][iou]['gt'] += stat[iou]['gt']
            tp = int(sum(stat[iou]['tp']))
            fp = int(sum(stat[iou]['fp']))
            recall = float(tp) / gt_count if gt_count else ''
            precision = float(tp) / (tp + fp) if (tp + fp) else ''
            suffix = str(iou).replace('.', '')
            row['tp_%s' % suffix] = tp
            row['fp_%s' % suffix] = fp
            row['recall_%s' % suffix] = (
                '' if recall == '' else '%.6f' % recall)
            row['precision_%s' % suffix] = (
                '' if precision == '' else '%.6f' % precision)
        rows.append(row)

    return rows


def format_float(value):
    return '' if value == '' else '%.6f' % value


def safe_corr(xs, ys, rank=False):
    if len(xs) < 2:
        return ''
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if rank:
        x = np.argsort(np.argsort(x)).astype(np.float64)
        y = np.argsort(np.argsort(y)).astype(np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return ''
    return float(np.corrcoef(x, y)[0, 1])


def summarize_area_ap(area_stats):
    rows = []
    for area_id in sorted(area_stats.keys()):
        row = OrderedDict({'area_id': area_id})
        for iou in (0.3, 0.5, 0.7):
            stat = area_stats[area_id]
            suffix = str(iou).replace('.', '')
            gt_total = stat[iou]['gt']
            tp_total = sum(stat[iou]['tp'])
            fp_total = sum(stat[iou]['fp'])
            row['gt_%s' % suffix] = gt_total
            row['tp_%s' % suffix] = int(tp_total)
            row['fp_%s' % suffix] = int(fp_total)
            if gt_total > 0 and len(stat[iou]['tp']) > 0:
                # calculate_ap mutates the tp/fp lists, so pass a copy.
                ap_stat = {
                    iou: {
                        'tp': list(stat[iou]['tp']),
                        'fp': list(stat[iou]['fp']),
                        'gt': gt_total,
                    }
                }
                ap, _, _ = eval_utils.calculate_ap(ap_stat, iou)
                recall = float(tp_total) / gt_total
                precision = (
                    float(tp_total) / (tp_total + fp_total)
                    if (tp_total + fp_total) else '')
                row['ap_%s' % suffix] = format_float(ap)
                row['recall_%s' % suffix] = format_float(recall)
                row['precision_%s' % suffix] = format_float(precision)
            else:
                row['ap_%s' % suffix] = ''
                row['recall_%s' % suffix] = ''
                row['precision_%s' % suffix] = ''
        rows.append(row)
    return rows


def confidence_aggregates(records):
    grouped = OrderedDict()
    for row in records:
        key = (row['scenario_id'], row['timestamp'], row['area_id'])
        grouped.setdefault(key, {
            'density_linear': [],
            'density_distance': [],
        })
        grouped[key]['density_linear'].append(float(row['density_linear']))
        grouped[key]['density_distance'].append(float(row['density_distance']))

    aggregates = {}
    for key, values in grouped.items():
        linear = np.asarray(values['density_linear'], dtype=np.float64)
        density_distance = np.asarray(
            values['density_distance'], dtype=np.float64)
        aggregates[key] = {
            'confidence_mean': float(np.mean(linear)),
            'confidence_max': float(np.max(linear)),
            'confidence_noisy_or': float(1.0 - np.prod(1.0 - linear)),
            'density_distance_mean': float(np.mean(density_distance)),
            'density_distance_max': float(np.max(density_distance)),
        }
    return aggregates


def quality_by_key(quality_rows):
    quality = {}
    for row in quality_rows:
        key = (row['scenario_id'], row['timestamp'], row['area_id'])
        quality[key] = {
            'recall_03': row['recall_03'],
            'recall_05': row['recall_05'],
            'recall_07': row['recall_07'],
            'precision_03': row['precision_03'],
            'precision_05': row['precision_05'],
            'precision_07': row['precision_07'],
            'score_mean': row.get('score_mean', ''),
            'score_max': row.get('score_max', ''),
            'score_top2_mean': row.get('score_top2_mean', ''),
            'score_top3_mean': row.get('score_top3_mean', ''),
            'gt_count': row['gt_count'],
            'pred_count': row['pred_count'],
        }
    return quality


def build_confidence_quality_records(records, quality_rows):
    confidence = confidence_aggregates(records)
    quality = quality_by_key(quality_rows)
    rows = []
    for key, conf_values in confidence.items():
        if key not in quality:
            continue
        scenario_id, timestamp, area_id = key
        q = quality[key]
        row = OrderedDict({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'area_id': area_id,
            'gt_count': q['gt_count'],
            'pred_count': q['pred_count'],
        })
        for name, value in conf_values.items():
            row[name] = format_float(value)
        for name in (
                'score_mean', 'score_max', 'score_top2_mean',
                'score_top3_mean',
                'recall_03', 'recall_05', 'recall_07',
                'precision_03', 'precision_05', 'precision_07'):
            row[name] = q[name]
        rows.append(row)
    return rows


def summarize_correlations(conf_quality_rows, ap_rows):
    rows = []
    confidence_fields = [
        'confidence_mean',
        'confidence_max',
        'confidence_noisy_or',
        'density_distance_mean',
        'density_distance_max',
        'score_mean',
        'score_max',
        'score_top2_mean',
        'score_top3_mean',
    ]
    quality_fields = ['recall_03', 'recall_05', 'recall_07']
    for confidence_field in confidence_fields:
        for quality_field in quality_fields:
            xs, ys = [], []
            for row in conf_quality_rows:
                if row[confidence_field] == '' or row[quality_field] == '':
                    continue
                if int(row['gt_count']) <= 0:
                    continue
                xs.append(float(row[confidence_field]))
                ys.append(float(row[quality_field]))
            rows.append(OrderedDict({
                'scope': 'area_frame',
                'confidence': confidence_field,
                'quality': quality_field,
                'samples': len(xs),
                'pearson': format_float(safe_corr(xs, ys)),
                'spearman': format_float(safe_corr(xs, ys, rank=True)),
            }))

    # AP rows are area-level across all selected frames. Aggregate confidence
    # over timestamps for the same area before comparing with accumulated AP.
    by_area = OrderedDict()
    for row in conf_quality_rows:
        by_area.setdefault(row['area_id'], [])
        by_area[row['area_id']].append(row)
    ap_by_area = {row['area_id']: row for row in ap_rows}
    for confidence_field in confidence_fields:
        for quality_field in ('ap_03', 'ap_05', 'ap_07'):
            xs, ys = [], []
            for area_id, rows_for_area in by_area.items():
                if area_id not in ap_by_area:
                    continue
                ap_value = ap_by_area[area_id][quality_field]
                if ap_value == '':
                    continue
                values = [
                    float(row[confidence_field])
                    for row in rows_for_area
                    if row[confidence_field] != ''
                ]
                if not values:
                    continue
                xs.append(float(np.mean(values)))
                ys.append(float(ap_value))
            rows.append(OrderedDict({
                'scope': 'area_accumulated',
                'confidence': confidence_field,
                'quality': quality_field,
                'samples': len(xs),
                'pearson': format_float(safe_corr(xs, ys)),
                'spearman': format_float(safe_corr(xs, ys, rank=True)),
            }))
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_inference(manager, dataset, scenario_id, timestamp, ego_cav_id):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=ego_cav_id)
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
        return_object_ids=True)
    return ret[0], ret[1], ret[2], ego_lidar_pose


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_lgcp_config(args.lgcp_yaml)
    if args.density_threshold is not None:
        config['density_threshold'] = args.density_threshold
    if args.grid_size_x is not None or args.grid_size_y is not None:
        gx, gy = config['grid_size']
        config['grid_size'] = (
            float(args.grid_size_x) if args.grid_size_x is not None else gx,
            float(args.grid_size_y) if args.grid_size_y is not None else gy)

    dataset = OPV2VFrameDataset(args.dataset_root)
    scenario_id = args.scenario_id or next(iter(dataset.scenarios.keys()))
    timestamps = dataset.scenarios[scenario_id]['timestamps']
    if args.max_frames == 0:
        selected_timestamps = timestamps[args.start_index:]
    else:
        selected_timestamps = timestamps[
            args.start_index:args.start_index + args.max_frames]

    area_ids = list_area_ids(config)
    area_m2 = config['grid_size'][0] * config['grid_size'][1]
    rho_th = config['density_threshold']
    if args.with_inference:
        coperception_params = load_coperception_params(
            args.coperception_yaml,
            args.fusion_method)
        manager = OpenCOODManager(coperception_params)
    else:
        coperception_params = None
        manager = None

    records = []
    quality_rows = []
    area_stats = {}
    summary_rows = []
    for timestamp in selected_timestamps:
        frame = dataset.load_frame(
            scenario_id, timestamp, ego_cav_id=args.ego_cav_id)
        gt_counts = count_gt_by_area(frame, config)

        frame_confidence = []
        frame_gt_count = []
        for agent_id, cav in frame.items():
            point_counts = count_points_by_area(
                cav['lidar_np'], cav['params']['lidar_pose'], config)
            lidar_x, lidar_y = cav['params']['lidar_pose'][0:2]
            for area_id in area_ids:
                point_count = int(point_counts.get(area_id, 0))
                gt_count = int(gt_counts.get(area_id, 0))
                if not args.include_empty and point_count == 0 and gt_count == 0:
                    continue

                density = point_count / area_m2
                density_linear = min(density / rho_th, 1.0) if rho_th > 0 else 0
                center_x, center_y = area_center(area_id, config)
                distance = math.hypot(center_x - lidar_x, center_y - lidar_y)
                distance_decay = math.exp(-distance / 50.0)
                density_distance = density_linear * distance_decay

                row = OrderedDict({
                    'scenario_id': scenario_id,
                    'timestamp': timestamp,
                    'area_id': area_id,
                    'agent_id': agent_id,
                    'area_center_x': '%.3f' % center_x,
                    'area_center_y': '%.3f' % center_y,
                    'agent_x': '%.3f' % float(lidar_x),
                    'agent_y': '%.3f' % float(lidar_y),
                    'distance': '%.3f' % distance,
                    'point_count': point_count,
                    'density': '%.6f' % density,
                    'density_linear': '%.6f' % density_linear,
                    'distance_decay': '%.6f' % distance_decay,
                    'density_distance': '%.6f' % density_distance,
                    'gt_count': gt_count,
                    'gt_present': 1 if gt_count > 0 else 0,
                })
                records.append(row)
                frame_confidence.append(density_linear)
                frame_gt_count.append(gt_count)

        pearson = safe_corr(frame_confidence, frame_gt_count)
        spearman = safe_corr(frame_confidence, frame_gt_count, rank=True)
        quality_gt = ''
        quality_pred = ''
        quality_recall_50 = ''
        if args.with_inference:
            pred_box_tensor, pred_score, gt_box_tensor, ego_pose = run_inference(
                manager, dataset, scenario_id, timestamp, args.ego_cav_id)
            timestamp_quality_rows = area_quality_rows(
                pred_box_tensor, pred_score, gt_box_tensor, ego_pose,
                scenario_id, timestamp, area_ids, config, area_stats)
            quality_rows.extend(timestamp_quality_rows)
            quality_gt = sum(int(row['gt_count'])
                             for row in timestamp_quality_rows)
            quality_pred = sum(int(row['pred_count'])
                               for row in timestamp_quality_rows)
            tp_50 = sum(int(row['tp_05']) for row in timestamp_quality_rows)
            quality_recall_50 = (
                '' if quality_gt == 0 else '%.6f' % (float(tp_50) / quality_gt))
        summary_rows.append(OrderedDict({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'agent_area_rows': len(frame_confidence),
            'gt_area_count': sum(1 for count in gt_counts.values()
                                 if count > 0),
            'gt_object_count_in_roi': sum(gt_counts.values()),
            'density_gt_count_pearson': '' if pearson == '' else '%.6f' % pearson,
            'density_gt_count_spearman': '' if spearman == '' else '%.6f' % spearman,
            'quality_gt_count': quality_gt,
            'quality_pred_count': quality_pred,
            'quality_recall_05': quality_recall_50,
        }))

    area_fields = [
        'scenario_id', 'timestamp', 'area_id', 'agent_id',
        'area_center_x', 'area_center_y', 'agent_x', 'agent_y', 'distance',
        'point_count', 'density', 'density_linear', 'distance_decay',
        'density_distance', 'gt_count', 'gt_present',
    ]
    summary_fields = [
        'scenario_id', 'timestamp', 'agent_area_rows', 'gt_area_count',
        'gt_object_count_in_roi', 'density_gt_count_pearson',
        'density_gt_count_spearman', 'quality_gt_count',
        'quality_pred_count', 'quality_recall_05',
    ]
    write_csv(os.path.join(args.output_dir, 'area_records.csv'),
              area_fields, records)
    write_csv(os.path.join(args.output_dir, 'density_gt_summary.csv'),
              summary_fields, summary_rows)
    if args.with_inference:
        quality_fields = [
            'scenario_id', 'timestamp', 'area_id', 'pred_count', 'gt_count',
            'score_mean', 'score_max', 'score_top2_mean', 'score_top3_mean',
            'tp_03', 'fp_03', 'recall_03', 'precision_03',
            'tp_05', 'fp_05', 'recall_05', 'precision_05',
            'tp_07', 'fp_07', 'recall_07', 'precision_07',
        ]
        write_csv(os.path.join(args.output_dir, 'area_quality.csv'),
                  quality_fields, quality_rows)
        ap_rows = summarize_area_ap(area_stats)
        ap_fields = [
            'area_id',
            'gt_03', 'tp_03', 'fp_03', 'ap_03', 'recall_03', 'precision_03',
            'gt_05', 'tp_05', 'fp_05', 'ap_05', 'recall_05', 'precision_05',
            'gt_07', 'tp_07', 'fp_07', 'ap_07', 'recall_07', 'precision_07',
        ]
        write_csv(os.path.join(args.output_dir, 'area_ap_summary.csv'),
                  ap_fields, ap_rows)
        conf_quality_rows = build_confidence_quality_records(
            records, quality_rows)
        conf_quality_fields = [
            'scenario_id', 'timestamp', 'area_id', 'gt_count', 'pred_count',
            'confidence_mean', 'confidence_max', 'confidence_noisy_or',
            'density_distance_mean', 'density_distance_max',
            'score_mean', 'score_max', 'score_top2_mean', 'score_top3_mean',
            'recall_03', 'recall_05', 'recall_07',
            'precision_03', 'precision_05', 'precision_07',
        ]
        write_csv(os.path.join(args.output_dir, 'confidence_quality_records.csv'),
                  conf_quality_fields, conf_quality_rows)
        corr_rows = summarize_correlations(conf_quality_rows, ap_rows)
        corr_fields = [
            'scope', 'confidence', 'quality', 'samples',
            'pearson', 'spearman',
        ]
        write_csv(os.path.join(args.output_dir,
                               'confidence_quality_correlation.csv'),
                  corr_fields, corr_rows)
    else:
        ap_rows = []
        conf_quality_rows = []
        corr_rows = []

    config_snapshot = {
        'scenario_id': scenario_id,
        'dataset_root': os.path.abspath(args.dataset_root),
        'lgcp_yaml': os.path.abspath(args.lgcp_yaml),
        'selected_timestamps': selected_timestamps,
        'roi': {
            'center': config['center'],
            'size': config['size'],
            'grid_size': config['grid_size'],
        },
        'density_threshold': rho_th,
        'include_empty': args.include_empty,
        'with_inference': args.with_inference,
        'fusion_method': None if coperception_params is None
        else coperception_params['fusion_method'],
        'note': 'area_ap_summary and confidence_quality_correlation are generated when with_inference is true.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config_snapshot, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Area Confidence Smoke Export\n\n')
        stream.write('This run exports density-based area confidence records. ')
        if args.with_inference:
            stream.write('It also runs OpenCOOD inference and slices predictions ')
            stream.write('by LGCP area to compute per-frame recall/precision, ')
            stream.write('accumulated area AP, and confidence-quality correlations.\n\n')
        else:
            stream.write('It validates area/grid assignment but does not compute ')
            stream.write('area-level AP or recall yet.\n\n')
        stream.write('- scenario_id: `%s`\n' % scenario_id)
        stream.write('- frames: `%s`\n' % ', '.join(selected_timestamps))
        stream.write('- area_records: `%d`\n' % len(records))
        stream.write('- summary_rows: `%d`\n' % len(summary_rows))
        if args.with_inference:
            stream.write('- area_quality_rows: `%d`\n' % len(quality_rows))
            stream.write('- area_ap_rows: `%d`\n' % len(ap_rows))
            stream.write('- confidence_quality_rows: `%d`\n' %
                         len(conf_quality_rows))
            stream.write('- correlation_rows: `%d`\n' % len(corr_rows))

    print('Wrote %d area records to %s' % (len(records), args.output_dir))
    for row in summary_rows:
        print('timestamp=%s rows=%s gt_objects=%s pearson=%s spearman=%s recall05=%s' % (
            row['timestamp'],
            row['agent_area_rows'],
            row['gt_object_count_in_roi'],
            row['density_gt_count_pearson'],
            row['density_gt_count_spearman'],
            row['quality_recall_05']))


if __name__ == '__main__':
    main()
