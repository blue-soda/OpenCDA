# -*- coding: utf-8 -*-
"""
Export an offline LGCP hierarchy plan from area confidence records.

This is a control-plane prototype for the LGCP pipeline:
RSU area assignment -> area-task groups -> leader selection ->
member-to-leader uploads -> leader-to-RSU uploads.

It does not perform real feature slicing, leader local fusion, or RSU global
aggregation yet. The output plan is meant to make those stages explicit and
replayable for later online / NS3 implementation.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export offline LGCP RSU hierarchy assignment plans.')
    parser.add_argument('--area-records', required=True)
    parser.add_argument('--area-quality', default=None,
                        help='Optional area_quality.csv for area priority.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--confidence-field', default='density_distance')
    parser.add_argument('--delta-g', type=float, default=0.075)
    parser.add_argument('--max-group-size', type=int, default=4)
    parser.add_argument('--max-areas', type=int, default=0,
                        help='Limit areas per frame. 0 means all areas.')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Limit frames. 0 means all frames.')
    parser.add_argument('--min-area-confidence', type=float, default=0.0,
                        help='Skip areas whose best CAV confidence is lower.')
    parser.add_argument('--feature-packet-bytes', type=int, default=10000)
    parser.add_argument('--leader-result-bytes', type=int, default=2000)
    parser.add_argument('--assignment-bytes', type=int, default=64,
                        help='Control bytes for one area assignment entry.')
    parser.add_argument('--broadcast-bytes', type=int, default=2000,
                        help='RSU global-view broadcast byte proxy per frame.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def noisy_or(values):
    if not values:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - values))


def group_confidence(area_conf, group):
    return noisy_or([area_conf.get(agent_id, 0.0) for agent_id in group])


def greedy_group(area_conf, delta_g, max_group_size):
    selected = []
    current = 0.0
    ranked_agents = sorted(
        area_conf.keys(),
        key=lambda agent_id: area_conf.get(agent_id, 0.0),
        reverse=True)
    for agent_id in ranked_agents:
        if len(selected) >= max_group_size:
            break
        new_conf = group_confidence(area_conf, selected + [agent_id])
        if new_conf - current >= delta_g:
            selected.append(agent_id)
            current = new_conf
    if not selected and ranked_agents:
        selected = [ranked_agents[0]]
        current = group_confidence(area_conf, selected)
    return selected, current


def build_area_confidence(area_records, confidence_field):
    by_timestamp_area = defaultdict(lambda: defaultdict(dict))
    area_centers = {}
    for row in area_records:
        agent_id = str(row['agent_id'])
        if agent_id == '-1':
            continue
        timestamp = row['timestamp']
        area_id = row['area_id']
        by_timestamp_area[timestamp][area_id][agent_id] = float(
            row[confidence_field])
        area_centers[area_id] = (
            row.get('area_center_x', ''),
            row.get('area_center_y', ''))
    return by_timestamp_area, area_centers


def build_area_priority(area_quality):
    priority = defaultdict(lambda: defaultdict(float))
    if not area_quality:
        return priority
    for row in area_quality:
        timestamp = row['timestamp']
        area_id = row['area_id']
        gt_count = float(row.get('gt_count', 0) or 0)
        pred_count = float(row.get('pred_count', 0) or 0)
        priority[timestamp][area_id] = max(
            priority[timestamp][area_id],
            1.0 + gt_count + 0.1 * pred_count)
    return priority


def choose_leader(group, leader_loads, area_conf):
    return min(
        group,
        key=lambda agent_id: (
            leader_loads[agent_id],
            -area_conf.get(agent_id, 0.0),
            agent_id))


def select_areas(timestamp_area_conf, timestamp_priority, max_areas,
                 min_area_confidence):
    candidates = []
    for area_id, area_conf in timestamp_area_conf.items():
        best_conf = max(area_conf.values()) if area_conf else 0.0
        if best_conf < min_area_confidence:
            continue
        candidates.append((
            area_id,
            timestamp_priority.get(area_id, 0.0),
            best_conf))
    candidates.sort(key=lambda item: (item[1], item[2], item[0]),
                    reverse=True)
    if max_areas > 0:
        candidates = candidates[:max_areas]
    return [item[0] for item in candidates]


def export_plans(area_conf_by_timestamp, area_priority, area_centers, args):
    timestamps = sorted(area_conf_by_timestamp.keys())
    if args.max_frames > 0:
        timestamps = timestamps[:args.max_frames]

    assignment_rows = []
    upload_rows = []
    summary_rows = []

    for timestamp in timestamps:
        timestamp_area_conf = area_conf_by_timestamp[timestamp]
        timestamp_priority = area_priority.get(timestamp, {})
        area_ids = select_areas(
            timestamp_area_conf,
            timestamp_priority,
            args.max_areas,
            args.min_area_confidence)
        leader_loads = defaultdict(int)
        covered_areas = 0
        local_uploads = 0
        leader_uploads = 0
        group_size_total = 0
        group_conf_total = 0.0

        for area_id in area_ids:
            area_conf = timestamp_area_conf[area_id]
            group, confidence = greedy_group(
                area_conf,
                args.delta_g,
                args.max_group_size)
            if not group:
                continue
            leader = choose_leader(group, leader_loads, area_conf)
            member_uploads = max(0, len(group) - 1)
            leader_loads[leader] += len(group)
            covered_areas += 1
            local_uploads += member_uploads
            leader_uploads += 1
            group_size_total += len(group)
            group_conf_total += confidence
            center_x, center_y = area_centers.get(area_id, ('', ''))

            assignment_rows.append(OrderedDict({
                'timestamp': timestamp,
                'area_id': area_id,
                'area_center_x': center_x,
                'area_center_y': center_y,
                'priority': '%.6f' % timestamp_priority.get(area_id, 0.0),
                'group_members': ';'.join(group),
                'group_size': len(group),
                'leader_id': leader,
                'group_confidence': '%.6f' % confidence,
                'member_uploads': member_uploads,
                'leader_uploads': 1,
            }))
            for member in group:
                if member == leader:
                    continue
                upload_rows.append(OrderedDict({
                    'timestamp': timestamp,
                    'area_id': area_id,
                    'source_id': member,
                    'target_id': leader,
                    'upload_type': 'member_to_leader',
                    'bytes': args.feature_packet_bytes,
                }))
            upload_rows.append(OrderedDict({
                'timestamp': timestamp,
                'area_id': area_id,
                'source_id': leader,
                'target_id': 'RSU',
                'upload_type': 'leader_to_rsu',
                'bytes': args.leader_result_bytes,
            }))

        assignment_bytes = covered_areas * args.assignment_bytes
        local_upload_bytes = local_uploads * args.feature_packet_bytes
        leader_upload_bytes = leader_uploads * args.leader_result_bytes
        broadcast_bytes = args.broadcast_bytes if covered_areas else 0
        total_bytes = (
            assignment_bytes + local_upload_bytes +
            leader_upload_bytes + broadcast_bytes)
        summary_rows.append(OrderedDict({
            'timestamp': timestamp,
            'area_count': len(area_ids),
            'covered_area_count': covered_areas,
            'avg_group_size': '%.6f' % (
                group_size_total / float(covered_areas or 1)),
            'avg_group_confidence': '%.6f' % (
                group_conf_total / float(covered_areas or 1)),
            'local_upload_packets': local_uploads,
            'leader_upload_packets': leader_uploads,
            'control_assignment_bytes': assignment_bytes,
            'local_upload_bytes': local_upload_bytes,
            'leader_upload_bytes': leader_upload_bytes,
            'broadcast_bytes': broadcast_bytes,
            'total_byte_proxy': total_bytes,
            'leader_count': len([load for load in leader_loads.values()
                                 if load > 0]),
            'leader_max_load': max(leader_loads.values())
            if leader_loads else 0,
            'leader_loads': ';'.join(
                '%s:%s' % item for item in sorted(leader_loads.items())),
        }))

    return assignment_rows, upload_rows, summary_rows


def summarize(summary_rows):
    if not summary_rows:
        return []
    numeric_fields = [
        'area_count',
        'covered_area_count',
        'avg_group_size',
        'avg_group_confidence',
        'local_upload_packets',
        'leader_upload_packets',
        'control_assignment_bytes',
        'local_upload_bytes',
        'leader_upload_bytes',
        'broadcast_bytes',
        'total_byte_proxy',
        'leader_count',
        'leader_max_load',
    ]
    row = OrderedDict({'frames': len(summary_rows)})
    for field in numeric_fields:
        values = np.asarray([float(item[field]) for item in summary_rows],
                            dtype=np.float64)
        row[field + '_mean'] = '%.6f' % float(np.mean(values))
        row[field + '_max'] = '%.6f' % float(np.max(values))
    return [row]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    area_records = read_csv(args.area_records)
    area_quality = read_csv(args.area_quality) if args.area_quality else []
    area_conf, area_centers = build_area_confidence(
        area_records,
        args.confidence_field)
    area_priority = build_area_priority(area_quality)
    assignment_rows, upload_rows, summary_rows = export_plans(
        area_conf,
        area_priority,
        area_centers,
        args)
    aggregate_rows = summarize(summary_rows)

    write_csv(os.path.join(args.output_dir, 'area_assignment_plan.csv'),
              ['timestamp', 'area_id', 'area_center_x', 'area_center_y',
               'priority', 'group_members', 'group_size', 'leader_id',
               'group_confidence', 'member_uploads', 'leader_uploads'],
              assignment_rows)
    write_csv(os.path.join(args.output_dir, 'upload_plan.csv'),
              ['timestamp', 'area_id', 'source_id', 'target_id',
               'upload_type', 'bytes'],
              upload_rows)
    write_csv(os.path.join(args.output_dir, 'hierarchy_frame_summary.csv'),
              ['timestamp', 'area_count', 'covered_area_count',
               'avg_group_size', 'avg_group_confidence',
               'local_upload_packets', 'leader_upload_packets',
               'control_assignment_bytes', 'local_upload_bytes',
               'leader_upload_bytes', 'broadcast_bytes', 'total_byte_proxy',
               'leader_count', 'leader_max_load', 'leader_loads'],
              summary_rows)
    write_csv(os.path.join(args.output_dir, 'hierarchy_summary.csv'),
              ['frames'] + [
                  field + suffix
                  for field in [
                      'area_count', 'covered_area_count', 'avg_group_size',
                      'avg_group_confidence', 'local_upload_packets',
                      'leader_upload_packets', 'control_assignment_bytes',
                      'local_upload_bytes', 'leader_upload_bytes',
                      'broadcast_bytes', 'total_byte_proxy',
                      'leader_count', 'leader_max_load']
                  for suffix in ('_mean', '_max')
              ],
              aggregate_rows)

    config = {
        'area_records': os.path.abspath(args.area_records),
        'area_quality': None if args.area_quality is None
        else os.path.abspath(args.area_quality),
        'confidence_field': args.confidence_field,
        'delta_g': args.delta_g,
        'max_group_size': args.max_group_size,
        'max_areas': args.max_areas,
        'max_frames': args.max_frames,
        'min_area_confidence': args.min_area_confidence,
        'feature_packet_bytes': args.feature_packet_bytes,
        'leader_result_bytes': args.leader_result_bytes,
        'assignment_bytes': args.assignment_bytes,
        'broadcast_bytes': args.broadcast_bytes,
        'note': 'Offline control-plane hierarchy plan; no real feature fusion.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Offline Hierarchy Plan\n\n')
        stream.write('This run exports RSU area assignment, area-task groups, ')
        stream.write('leader selection, member-to-leader uploads, and ')
        stream.write('leader-to-RSU uploads from area confidence records. ')
        stream.write('It does not perform real feature fusion yet.\n\n')
        stream.write('- assignment_rows: `%d`\n' % len(assignment_rows))
        stream.write('- upload_rows: `%d`\n' % len(upload_rows))
        stream.write('- summary_rows: `%d`\n' % len(summary_rows))

    print('Wrote LGCP hierarchy plan to %s' % args.output_dir)
    if aggregate_rows:
        row = aggregate_rows[0]
        print('frames=%s area_mean=%s local_packets_mean=%s '
              'leader_packets_mean=%s total_bytes_mean=%s '
              'leader_max_load_mean=%s' % (
                  row['frames'],
                  row['covered_area_count_mean'],
                  row['local_upload_packets_mean'],
                  row['leader_upload_packets_mean'],
                  row['total_byte_proxy_mean'],
                  row['leader_max_load_mean']))


if __name__ == '__main__':
    main()
