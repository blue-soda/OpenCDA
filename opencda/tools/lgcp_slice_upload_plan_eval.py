# -*- coding: utf-8 -*-
"""
Replace fixed LGCP member-upload packet sizes with area-slice byte proxies.

The hierarchy planner exports upload_plan.csv with fixed member_to_leader
packet bytes. lgcp_feature_slice_manifest.py exports data-dependent raw
LiDAR slice bytes for each (timestamp, area, member, leader). This tool joins
both products and writes an upload plan that can be replayed by
offline_ns3_replay.py without changing the replay code.
"""

import argparse
import csv
import os
from collections import OrderedDict, defaultdict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build raw-slice-aware LGCP upload plans.')
    parser.add_argument('--upload-plan', required=True,
                        help='Fixed-byte upload_plan.csv from hierarchy plan.')
    parser.add_argument('--feature-slice-manifest', required=True,
                        help='feature_slice_manifest.csv from slice export.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--min-member-bytes', type=int, default=0,
                        help='Optional floor for member_to_leader byte sizes.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_slice_index(slice_rows):
    index = {}
    duplicates = defaultdict(int)
    for row in slice_rows:
        if row.get('upload_type') != 'member_to_leader':
            continue
        key = (
            row['timestamp'],
            row['area_id'],
            str(row['agent_id']),
            str(row['leader_id']),
        )
        duplicates[key] += 1
        index[key] = row
    return index, duplicates


def rewrite_upload_plan(upload_rows, slice_index, min_member_bytes):
    rows = []
    unmatched = []
    for row in upload_rows:
        original_bytes = int(float(row['bytes']))
        new_bytes = original_bytes
        byte_source = 'fixed_original'
        raw_point_count = ''
        slice_point_count = ''
        slice_ratio = ''

        if row.get('upload_type') == 'member_to_leader':
            key = (
                row['timestamp'],
                row['area_id'],
                str(row['source_id']),
                str(row['target_id']),
            )
            slice_row = slice_index.get(key)
            if slice_row is None:
                unmatched.append(key)
                byte_source = 'missing_slice_fallback'
            else:
                new_bytes = max(
                    min_member_bytes,
                    int(float(slice_row['byte_proxy'])))
                byte_source = 'raw_lidar_slice'
                raw_point_count = slice_row.get('raw_point_count', '')
                slice_point_count = slice_row.get('slice_point_count', '')
                slice_ratio = slice_row.get('slice_ratio', '')

        rows.append(OrderedDict({
            'timestamp': row['timestamp'],
            'area_id': row['area_id'],
            'source_id': row['source_id'],
            'target_id': row['target_id'],
            'upload_type': row.get('upload_type', ''),
            'bytes': new_bytes,
            'original_bytes': original_bytes,
            'byte_source': byte_source,
            'raw_point_count': raw_point_count,
            'slice_point_count': slice_point_count,
            'slice_ratio': slice_ratio,
        }))
    return rows, unmatched


def summarize(rows, unmatched_count):
    if not rows:
        return []
    by_type = defaultdict(list)
    for row in rows:
        by_type[row['upload_type']].append(row)

    summary_rows = []
    all_rows = list(rows)
    for upload_type in sorted(by_type.keys()) + ['all']:
        group = all_rows if upload_type == 'all' else by_type[upload_type]
        bytes_values = np.asarray([float(row['bytes']) for row in group])
        original_values = np.asarray(
            [float(row['original_bytes']) for row in group])
        summary_rows.append(OrderedDict({
            'upload_type': upload_type,
            'requests': len(group),
            'bytes_total': '%.6f' % float(np.sum(bytes_values)),
            'bytes_mean': '%.6f' % float(np.mean(bytes_values)),
            'original_bytes_total': '%.6f' % float(np.sum(original_values)),
            'original_bytes_mean': '%.6f' % float(np.mean(original_values)),
            'byte_ratio_vs_original': '%.6f' % (
                float(np.sum(bytes_values)) /
                max(float(np.sum(original_values)), 1e-9)),
            'unmatched_member_rows': unmatched_count
            if upload_type == 'member_to_leader' else '',
        }))
    return summary_rows


def summarize_frames(rows):
    by_timestamp = defaultdict(list)
    for row in rows:
        by_timestamp[row['timestamp']].append(row)

    frame_rows = []
    for timestamp in sorted(by_timestamp.keys()):
        group = by_timestamp[timestamp]
        member_rows = [
            row for row in group
            if row['upload_type'] == 'member_to_leader']
        leader_rows = [
            row for row in group
            if row['upload_type'] == 'leader_to_rsu']
        frame_rows.append(OrderedDict({
            'timestamp': timestamp,
            'requests': len(group),
            'member_requests': len(member_rows),
            'leader_requests': len(leader_rows),
            'bytes_total': sum(int(row['bytes']) for row in group),
            'member_bytes': sum(int(row['bytes']) for row in member_rows),
            'leader_bytes': sum(int(row['bytes']) for row in leader_rows),
            'original_bytes_total': sum(
                int(row['original_bytes']) for row in group),
        }))
    return frame_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    upload_rows = read_csv(args.upload_plan)
    slice_rows = read_csv(args.feature_slice_manifest)
    slice_index, duplicates = build_slice_index(slice_rows)
    rewritten_rows, unmatched = rewrite_upload_plan(
        upload_rows, slice_index, args.min_member_bytes)
    summary_rows = summarize(rewritten_rows, len(unmatched))
    frame_rows = summarize_frames(rewritten_rows)

    write_csv(os.path.join(args.output_dir, 'raw_slice_upload_plan.csv'),
              ['timestamp', 'area_id', 'source_id', 'target_id',
               'upload_type', 'bytes', 'original_bytes', 'byte_source',
               'raw_point_count', 'slice_point_count', 'slice_ratio'],
              rewritten_rows)
    write_csv(os.path.join(args.output_dir, 'raw_slice_upload_summary.csv'),
              ['upload_type', 'requests', 'bytes_total', 'bytes_mean',
               'original_bytes_total', 'original_bytes_mean',
               'byte_ratio_vs_original', 'unmatched_member_rows'],
              summary_rows)
    write_csv(os.path.join(args.output_dir,
                           'raw_slice_upload_frame_summary.csv'),
              ['timestamp', 'requests', 'member_requests', 'leader_requests',
               'bytes_total', 'member_bytes', 'leader_bytes',
               'original_bytes_total'],
              frame_rows)

    config = {
        'upload_plan': os.path.abspath(args.upload_plan),
        'feature_slice_manifest': os.path.abspath(
            args.feature_slice_manifest),
        'min_member_bytes': args.min_member_bytes,
        'slice_member_rows': len(slice_index),
        'duplicate_slice_keys': sum(
            1 for count in duplicates.values() if count > 1),
        'unmatched_member_rows': len(unmatched),
        'note': 'member_to_leader bytes use raw LiDAR area-slice byte_proxy; leader_to_rsu bytes remain unchanged.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Raw-Slice Upload Plan\n\n')
        stream.write('This run rewrites hierarchy member-to-leader upload ')
        stream.write('bytes using raw LiDAR area-slice byte proxies. ')
        stream.write('Leader-to-RSU result bytes remain unchanged.\n\n')
        stream.write('- upload_rows: `%d`\n' % len(rewritten_rows))
        stream.write('- unmatched_member_rows: `%d`\n' % len(unmatched))

    print('Wrote raw-slice-aware upload plan to %s' % args.output_dir)
    for row in summary_rows:
        print('%s requests=%s bytes=%s ratio=%s unmatched=%s' % (
            row['upload_type'],
            row['requests'],
            row['bytes_total'],
            row['byte_ratio_vs_original'],
            row['unmatched_member_rows']))


if __name__ == '__main__':
    main()
