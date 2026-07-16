# -*- coding: utf-8 -*-
"""
Estimate LGCP control-plane overhead from offline hierarchy artifacts.

The estimator separates CAV pose reports, area-confidence reports, RSU
assignment messages, RSU global-view broadcast, and planned data uploads. It is
intended for rebuttal / paper tables, so every byte assumption is exposed as a
CLI argument and written back to config.yaml.
"""

import argparse
import csv
import os
from collections import OrderedDict, defaultdict

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate LGCP control-plane overhead from CSV artifacts.')
    parser.add_argument('--area-records', required=True)
    parser.add_argument('--area-assignment-plan', required=True)
    parser.add_argument('--upload-plan', required=True)
    parser.add_argument('--hierarchy-frame-summary', default=None,
                        help='Optional hierarchy_frame_summary.csv.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--confidence-field', default='density_distance')
    parser.add_argument('--min-confidence-report', type=float, default=0.0,
                        help='Report CAV-area confidence only above this value.')
    parser.add_argument('--pose-report-bytes', type=int, default=32,
                        help='Per-CAV pose/direction/speed report bytes.')
    parser.add_argument('--confidence-entry-bytes', type=int, default=16,
                        help='Per reported CAV-area confidence entry bytes.')
    parser.add_argument('--assignment-entry-bytes', type=int, default=64,
                        help='Fallback bytes for one RSU area assignment entry.')
    parser.add_argument('--global-view-bytes', type=int, default=2000,
                        help='Fallback RSU global-view broadcast bytes/frame.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_area_reports(rows, confidence_field, min_confidence):
    agents_by_frame = defaultdict(set)
    confidence_entries = defaultdict(int)
    for row in rows:
        agent_id = str(row.get('agent_id', ''))
        if not agent_id.lstrip('-').isdigit() or int(agent_id) <= 0:
            continue
        timestamp = row['timestamp']
        agents_by_frame[timestamp].add(agent_id)
        if as_float(row.get(confidence_field)) > min_confidence:
            confidence_entries[timestamp] += 1
    return agents_by_frame, confidence_entries


def collect_assignments(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row['timestamp']] += 1
    return counts


def collect_uploads(rows):
    packets = defaultdict(int)
    bytes_by_frame = defaultdict(int)
    type_packets = defaultdict(lambda: defaultdict(int))
    type_bytes = defaultdict(lambda: defaultdict(int))
    for row in rows:
        timestamp = row['timestamp']
        upload_type = row.get('upload_type', 'unknown')
        packet_bytes = int(float(row.get('bytes', 0) or 0))
        packets[timestamp] += 1
        bytes_by_frame[timestamp] += packet_bytes
        type_packets[timestamp][upload_type] += 1
        type_bytes[timestamp][upload_type] += packet_bytes
    return packets, bytes_by_frame, type_packets, type_bytes


def collect_frame_summary(rows):
    summary = {}
    for row in rows:
        summary[row['timestamp']] = row
    return summary


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def max_or_zero(values):
    values = list(values)
    if not values:
        return 0.0
    return max(values)


def build_rows(args):
    area_rows = read_csv(args.area_records)
    assignment_rows = read_csv(args.area_assignment_plan)
    upload_rows = read_csv(args.upload_plan)
    frame_summary_rows = (
        read_csv(args.hierarchy_frame_summary)
        if args.hierarchy_frame_summary else [])

    agents_by_frame, confidence_entries = collect_area_reports(
        area_rows, args.confidence_field, args.min_confidence_report)
    assignments = collect_assignments(assignment_rows)
    upload_packets, upload_bytes, type_packets, type_bytes = collect_uploads(
        upload_rows)
    frame_summary = collect_frame_summary(frame_summary_rows)

    timestamps = sorted(set(agents_by_frame) | set(assignments) |
                        set(upload_packets) | set(frame_summary))
    rows = []
    for timestamp in timestamps:
        active_cavs = len(agents_by_frame.get(timestamp, set()))
        pose_bytes = active_cavs * args.pose_report_bytes
        confidence_bytes = (
            confidence_entries.get(timestamp, 0) *
            args.confidence_entry_bytes)

        summary = frame_summary.get(timestamp, {})
        assignment_bytes = int(float(
            summary.get('control_assignment_bytes', 0) or 0))
        if not assignment_bytes:
            assignment_bytes = (
                assignments.get(timestamp, 0) *
                args.assignment_entry_bytes)
        global_view_bytes = int(float(
            summary.get('broadcast_bytes', 0) or 0))
        if not global_view_bytes:
            global_view_bytes = args.global_view_bytes

        control_bytes = (
            pose_bytes + confidence_bytes + assignment_bytes +
            global_view_bytes)
        planned_data_bytes = upload_bytes.get(timestamp, 0)
        total_bytes = control_bytes + planned_data_bytes
        control_ratio = (
            control_bytes / float(total_bytes) if total_bytes else 0.0)

        rows.append(OrderedDict({
            'timestamp': timestamp,
            'active_cavs': active_cavs,
            'confidence_entries': confidence_entries.get(timestamp, 0),
            'assignment_entries': assignments.get(timestamp, 0),
            'pose_report_bytes': pose_bytes,
            'confidence_report_bytes': confidence_bytes,
            'assignment_bytes': assignment_bytes,
            'global_view_bytes': global_view_bytes,
            'control_plane_bytes': control_bytes,
            'member_to_leader_packets': type_packets[timestamp].get(
                'member_to_leader', 0),
            'leader_to_rsu_packets': type_packets[timestamp].get(
                'leader_to_rsu', 0),
            'member_to_leader_bytes': type_bytes[timestamp].get(
                'member_to_leader', 0),
            'leader_to_rsu_bytes': type_bytes[timestamp].get(
                'leader_to_rsu', 0),
            'planned_data_bytes': planned_data_bytes,
            'total_bytes_with_control': total_bytes,
            'control_plane_ratio': '%.6f' % control_ratio,
        }))
    return rows


def summarize(rows):
    numeric_fields = [
        'active_cavs', 'confidence_entries', 'assignment_entries',
        'pose_report_bytes', 'confidence_report_bytes', 'assignment_bytes',
        'global_view_bytes', 'control_plane_bytes',
        'member_to_leader_packets', 'leader_to_rsu_packets',
        'member_to_leader_bytes', 'leader_to_rsu_bytes',
        'planned_data_bytes', 'total_bytes_with_control',
        'control_plane_ratio',
    ]
    summary = OrderedDict({'frames': len(rows)})
    for field in numeric_fields:
        values = [as_float(row[field]) for row in rows]
        summary[field + '_mean'] = '%.6f' % mean(values)
        summary[field + '_max'] = '%.6f' % max_or_zero(values)
    return summary


def write_notes(path, rows, summary):
    with open(path, 'w') as stream:
        stream.write('# LGCP Control-Plane Overhead\n\n')
        stream.write('This run estimates per-frame control-plane overhead from ')
        stream.write('offline LGCP hierarchy artifacts.\n\n')
        stream.write('- frames: `%s`\n' % summary['frames'])
        stream.write('- control_plane_bytes_mean: `%s`\n' %
                     summary['control_plane_bytes_mean'])
        stream.write('- planned_data_bytes_mean: `%s`\n' %
                     summary['planned_data_bytes_mean'])
        stream.write('- control_plane_ratio_mean: `%s`\n' %
                     summary['control_plane_ratio_mean'])
        stream.write('\nOnly positive CAV ids are counted as CAV reports; ')
        stream.write('non-positive ids are treated as RSU/reference records.\n')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = build_rows(args)
    summary = summarize(rows)

    frame_fields = list(rows[0].keys()) if rows else [
        'timestamp', 'active_cavs', 'confidence_entries',
        'assignment_entries', 'pose_report_bytes',
        'confidence_report_bytes', 'assignment_bytes',
        'global_view_bytes', 'control_plane_bytes',
        'member_to_leader_packets', 'leader_to_rsu_packets',
        'member_to_leader_bytes', 'leader_to_rsu_bytes',
        'planned_data_bytes', 'total_bytes_with_control',
        'control_plane_ratio']
    write_csv(os.path.join(args.output_dir, 'control_overhead_by_frame.csv'),
              frame_fields, rows)
    write_csv(os.path.join(args.output_dir, 'control_overhead_summary.csv'),
              list(summary.keys()), [summary])

    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump({
            'area_records': os.path.abspath(args.area_records),
            'area_assignment_plan': os.path.abspath(
                args.area_assignment_plan),
            'upload_plan': os.path.abspath(args.upload_plan),
            'hierarchy_frame_summary': os.path.abspath(
                args.hierarchy_frame_summary)
            if args.hierarchy_frame_summary else '',
            'confidence_field': args.confidence_field,
            'min_confidence_report': args.min_confidence_report,
            'pose_report_bytes': args.pose_report_bytes,
            'confidence_entry_bytes': args.confidence_entry_bytes,
            'assignment_entry_bytes': args.assignment_entry_bytes,
            'global_view_bytes': args.global_view_bytes,
            'note': 'Offline LGCP control-plane overhead estimate.',
        }, stream, sort_keys=False)
    write_notes(os.path.join(args.output_dir, 'notes.md'), rows, summary)

    print('frames=%d control_plane_bytes_mean=%s '
          'planned_data_bytes_mean=%s control_plane_ratio_mean=%s' % (
              len(rows),
              summary['control_plane_bytes_mean'],
              summary['planned_data_bytes_mean'],
              summary['control_plane_ratio_mean']))


if __name__ == '__main__':
    main()
