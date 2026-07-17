# -*- coding: utf-8 -*-
"""
Evaluate offline LGCP leader-result and RSU global-aggregation proxies.

This consumes the hierarchy assignment plan exported by
lgcp_hierarchy_plan_eval.py plus area-level quality records from
lgcp_area_confidence_eval.py. It does not perform model-level feature fusion.
Instead, it materializes the missing hierarchy data products:

area assignment -> leader local result proxy -> RSU global aggregation proxy.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP offline hierarchy aggregation proxies.')
    parser.add_argument('--assignment-plan', required=True,
                        help='area_assignment_plan.csv from hierarchy plan.')
    parser.add_argument('--area-quality', required=True,
                        help='area_quality.csv from area confidence eval.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--quality-field', default='recall_05',
                        help='Area quality field used for aggregation proxy.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_or_zero(value):
    if value in (None, ''):
        return 0.0
    return float(value)


def load_quality(rows, quality_field):
    by_key = {}
    by_timestamp = defaultdict(dict)
    for row in rows:
        key = (row['timestamp'], row['area_id'])
        quality = float_or_zero(row.get(quality_field))
        item = {
            'scenario_id': row.get('scenario_id', ''),
            'quality': quality,
            'gt_count': float_or_zero(row.get('gt_count')),
            'pred_count': float_or_zero(row.get('pred_count')),
            'recall_03': float_or_zero(row.get('recall_03')),
            'recall_05': float_or_zero(row.get('recall_05')),
            'recall_07': float_or_zero(row.get('recall_07')),
            'precision_05': float_or_zero(row.get('precision_05')),
        }
        by_key[key] = item
        by_timestamp[row['timestamp']][row['area_id']] = item
    return by_key, by_timestamp


def build_leader_results(assignments, quality_by_key):
    rows = []
    for row in assignments:
        key = (row['timestamp'], row['area_id'])
        quality = quality_by_key.get(key, {})
        group_conf = float_or_zero(row.get('group_confidence'))
        area_quality = float_or_zero(quality.get('quality'))
        gt_count = float_or_zero(quality.get('gt_count'))
        pred_count = float_or_zero(quality.get('pred_count'))
        members = [item for item in row['group_members'].split(';') if item]
        rows.append(OrderedDict({
            'timestamp': row['timestamp'],
            'area_id': row['area_id'],
            'leader_id': row['leader_id'],
            'group_members': row['group_members'],
            'group_size': row['group_size'],
            'group_confidence': '%.6f' % group_conf,
            'area_quality': '%.6f' % area_quality,
            'confidence_weighted_quality': '%.6f' % (
                group_conf * area_quality),
            'gt_count': '%.6f' % gt_count,
            'pred_count': '%.6f' % pred_count,
            'member_count_check': len(members),
        }))
    return rows


def summarize_frames(leader_rows, quality_by_timestamp):
    by_timestamp = defaultdict(list)
    for row in leader_rows:
        by_timestamp[row['timestamp']].append(row)

    rows = []
    for timestamp in sorted(quality_by_timestamp.keys()):
        frame_quality = quality_by_timestamp[timestamp]
        selected = by_timestamp.get(timestamp, [])
        selected_area_ids = {row['area_id'] for row in selected}
        total_areas = len(frame_quality)
        total_gt = sum(item['gt_count'] for item in frame_quality.values())
        selected_gt = sum(
            frame_quality[area_id]['gt_count']
            for area_id in selected_area_ids
            if area_id in frame_quality)
        selected_quality = [
            float_or_zero(row['area_quality'])
            for row in selected]
        selected_weighted_quality = [
            float_or_zero(row['confidence_weighted_quality'])
            for row in selected]
        leader_loads = defaultdict(int)
        for row in selected:
            leader_loads[row['leader_id']] += int(row['group_size'])

        rows.append(OrderedDict({
            'timestamp': timestamp,
            'total_quality_areas': total_areas,
            'selected_areas': len(selected),
            'selected_area_ratio': '%.6f' % (
                len(selected) / float(total_areas or 1)),
            'total_gt_count': '%.6f' % total_gt,
            'selected_gt_count': '%.6f' % selected_gt,
            'selected_gt_ratio': '%.6f' % (
                selected_gt / float(total_gt or 1.0)),
            'mean_area_quality': '%.6f' % (
                float(np.mean(selected_quality)) if selected_quality else 0.0),
            'mean_confidence_weighted_quality': '%.6f' % (
                float(np.mean(selected_weighted_quality))
                if selected_weighted_quality else 0.0),
            'leader_count': len(leader_loads),
            'leader_max_load': max(leader_loads.values())
            if leader_loads else 0,
            'leader_loads': ';'.join(
                '%s:%s' % item for item in sorted(leader_loads.items())),
        }))
    return rows


def summarize_global(frame_rows):
    if not frame_rows:
        return []
    fields = [
        'total_quality_areas',
        'selected_areas',
        'selected_area_ratio',
        'total_gt_count',
        'selected_gt_count',
        'selected_gt_ratio',
        'mean_area_quality',
        'mean_confidence_weighted_quality',
        'leader_count',
        'leader_max_load',
    ]
    row = OrderedDict({'frames': len(frame_rows)})
    for field in fields:
        values = np.asarray([float(item[field]) for item in frame_rows],
                            dtype=np.float64)
        row[field + '_mean'] = '%.6f' % float(np.mean(values))
        row[field + '_min'] = '%.6f' % float(np.min(values))
        row[field + '_max'] = '%.6f' % float(np.max(values))
    return [row]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assignment_rows = read_csv(args.assignment_plan)
    quality_rows = read_csv(args.area_quality)
    quality_by_key, quality_by_timestamp = load_quality(
        quality_rows, args.quality_field)

    leader_rows = build_leader_results(assignment_rows, quality_by_key)
    frame_rows = summarize_frames(leader_rows, quality_by_timestamp)
    global_rows = summarize_global(frame_rows)

    write_csv(os.path.join(args.output_dir, 'leader_local_results.csv'),
              ['timestamp', 'area_id', 'leader_id', 'group_members',
               'group_size', 'group_confidence', 'area_quality',
               'confidence_weighted_quality', 'gt_count', 'pred_count',
               'member_count_check'],
              leader_rows)
    write_csv(os.path.join(args.output_dir, 'rsu_global_frame_summary.csv'),
              ['timestamp', 'total_quality_areas', 'selected_areas',
               'selected_area_ratio', 'total_gt_count', 'selected_gt_count',
               'selected_gt_ratio', 'mean_area_quality',
               'mean_confidence_weighted_quality', 'leader_count',
               'leader_max_load', 'leader_loads'],
              frame_rows)
    write_csv(os.path.join(args.output_dir, 'rsu_global_summary.csv'),
              ['frames'] + [
                  field + suffix
                  for field in [
                      'total_quality_areas', 'selected_areas',
                      'selected_area_ratio', 'total_gt_count',
                      'selected_gt_count', 'selected_gt_ratio',
                      'mean_area_quality',
                      'mean_confidence_weighted_quality',
                      'leader_count', 'leader_max_load']
                  for suffix in ('_mean', '_min', '_max')
              ],
              global_rows)

    config = {
        'assignment_plan': os.path.abspath(args.assignment_plan),
        'area_quality': os.path.abspath(args.area_quality),
        'quality_field': args.quality_field,
        'note': 'Offline proxy only; no model-level feature fusion.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Hierarchy Aggregation Proxy\n\n')
        stream.write('This run materializes leader local result and RSU ')
        stream.write('global aggregation proxy records from the hierarchy ')
        stream.write('assignment plan. It does not perform real feature ')
        stream.write('slicing or OpenCOOD model-level leader fusion.\n\n')
        stream.write('- leader_result_rows: `%d`\n' % len(leader_rows))
        stream.write('- frame_summary_rows: `%d`\n' % len(frame_rows))

    print('Wrote LGCP hierarchy aggregation proxy to %s' % args.output_dir)
    if global_rows:
        row = global_rows[0]
        print('frames=%s selected_gt_ratio_mean=%s '
              'mean_area_quality=%s leader_max_load_mean=%s' % (
                  row['frames'],
                  row['selected_gt_ratio_mean'],
                  row['mean_area_quality_mean'],
                  row['leader_max_load_mean']))


if __name__ == '__main__':
    main()
