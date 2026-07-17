"""Summarize SGCP CAV coverage from receiver-level protocol traces.

This diagnostic answers a different question from AP: which CAVs are cluster
heads, which members actually upload, and which members are left unfused by the
PPS schedule.  It is useful when two variants have similar payload but different
late-fusion GT coverage.
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SGCP trace coverage by CAV and by frame.')
    parser.add_argument('--trace-csv', required=True,
                        help='Receiver-level SGCP trace CSV.')
    parser.add_argument('--output-cav-csv', required=True,
                        help='Per-CAV summary CSV to write.')
    parser.add_argument('--output-frame-csv', default='',
                        help='Optional per-frame coverage summary CSV.')
    parser.add_argument('--label', default='',
                        help='Experiment label recorded in output CSVs.')
    return parser.parse_args()


def parse_int(value, default=0):
    if value is None or value == '':
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_id_list(value):
    if not value:
        return []
    result = []
    for item in str(value).split(';'):
        item = item.strip()
        if not item:
            continue
        try:
            result.append(int(item))
        except ValueError:
            continue
    return result


def parse_json_int_map(value):
    if not value:
        return {}
    try:
        raw = json.loads(value)
    except (TypeError, ValueError):
        return {}
    result = {}
    for key, item in raw.items():
        try:
            result[int(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result


def read_rows(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def ensure_parent(path):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def empty_cav_stat():
    return {
        'frames_present': set(),
        'head_frames': set(),
        'member_frames': set(),
        'uploaded_frames': set(),
        'fused_frames': set(),
        'unscheduled_member_frames': set(),
        'selected_grids': 0,
        'uploaded_points': 0,
        'local_points': 0,
        'communication_bytes': 0,
        'pred_boxes_when_head': 0,
        'gt_boxes_when_head': 0,
    }


def add_frame(stat, key, field):
    stat[field].add(key)
    stat['frames_present'].add(key)


def summarize(label, rows):
    cav_stats = defaultdict(empty_cav_stat)
    frame_stats = {}
    all_frame_keys = sorted({
        (row.get('scenario_id', ''), row.get('timestamp', ''))
        for row in rows
    })

    for row in rows:
        frame_key = (row.get('scenario_id', ''), row.get('timestamp', ''))
        frame_id = '%s/%s' % frame_key
        receiver_id = parse_int(row.get('receiver_id'))
        member_ids = set(parse_id_list(row.get('cluster_member_ids')))
        uploaded_ids = set(parse_id_list(row.get('uploaded_source_ids')))
        selected_grids = parse_json_int_map(
            row.get('selected_grid_counts_json'))
        point_counts = parse_json_int_map(row.get('point_counts_json'))
        communication_bytes = parse_int(row.get('communication_bytes'))

        frame = frame_stats.setdefault(frame_key, {
            'heads': set(),
            'members': set(),
            'uploaded': set(),
            'fused': set(),
            'unscheduled_members': set(),
            'selected_grids': 0,
            'uploaded_points': 0,
            'communication_bytes': 0,
        })

        frame['heads'].add(receiver_id)
        frame['members'].update(member_ids)
        frame['uploaded'].update(uploaded_ids)
        frame['fused'].add(receiver_id)
        frame['fused'].update(uploaded_ids)
        frame['selected_grids'] += sum(selected_grids.values())
        frame['communication_bytes'] += communication_bytes

        add_frame(cav_stats[receiver_id], frame_id, 'head_frames')
        add_frame(cav_stats[receiver_id], frame_id, 'fused_frames')
        cav_stats[receiver_id]['local_points'] += point_counts.get(
            receiver_id, 0)
        cav_stats[receiver_id]['pred_boxes_when_head'] += parse_int(
            row.get('pred_boxes'))
        cav_stats[receiver_id]['gt_boxes_when_head'] += parse_int(
            row.get('gt_boxes'))

        for cav_id in member_ids:
            add_frame(cav_stats[cav_id], frame_id, 'member_frames')
            if cav_id != receiver_id and cav_id not in uploaded_ids:
                add_frame(cav_stats[cav_id], frame_id,
                          'unscheduled_member_frames')
                frame['unscheduled_members'].add(cav_id)

        for cav_id in uploaded_ids:
            add_frame(cav_stats[cav_id], frame_id, 'uploaded_frames')
            add_frame(cav_stats[cav_id], frame_id, 'fused_frames')
            cav_stats[cav_id]['selected_grids'] += selected_grids.get(
                cav_id, 0)
            cav_stats[cav_id]['uploaded_points'] += point_counts.get(
                cav_id, 0)
            cav_stats[cav_id]['communication_bytes'] += (
                communication_bytes if len(uploaded_ids) == 1 else 0)
            frame['uploaded_points'] += point_counts.get(cav_id, 0)

    cav_rows = []
    frame_count = max(1, len(all_frame_keys))
    for cav_id in sorted(cav_stats.keys()):
        stat = cav_stats[cav_id]
        uploaded_count = len(stat['uploaded_frames'])
        member_count = len(stat['member_frames'])
        fused_count = len(stat['fused_frames'])
        cav_rows.append({
            'label': label,
            'cav_id': cav_id,
            'frames': frame_count,
            'head_frames': len(stat['head_frames']),
            'member_frames': member_count,
            'uploaded_frames': uploaded_count,
            'fused_frames': fused_count,
            'unscheduled_member_frames': len(
                stat['unscheduled_member_frames']),
            'upload_rate_when_member': (
                uploaded_count / float(member_count) if member_count else 0.0),
            'fused_rate': fused_count / float(frame_count),
            'selected_grids': stat['selected_grids'],
            'uploaded_points': stat['uploaded_points'],
            'local_points_when_head': stat['local_points'],
            'pred_boxes_when_head': stat['pred_boxes_when_head'],
            'gt_boxes_when_head': stat['gt_boxes_when_head'],
        })

    frame_rows = []
    for frame_key in all_frame_keys:
        frame = frame_stats.get(frame_key, {})
        frame_rows.append({
            'label': label,
            'scenario_id': frame_key[0],
            'timestamp': frame_key[1],
            'head_count': len(frame.get('heads', set())),
            'member_count': len(frame.get('members', set())),
            'uploaded_count': len(frame.get('uploaded', set())),
            'fused_count': len(frame.get('fused', set())),
            'unscheduled_member_count': len(
                frame.get('unscheduled_members', set())),
            'selected_grids': frame.get('selected_grids', 0),
            'uploaded_points': frame.get('uploaded_points', 0),
            'communication_bytes': frame.get('communication_bytes', 0),
        })

    return cav_rows, frame_rows


def write_csv(path, rows, fieldnames):
    ensure_parent(path)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(rows, field):
    if not rows:
        return 0.0
    return sum(float(row[field]) for row in rows) / len(rows)


def main():
    args = parse_args()
    rows = read_rows(args.trace_csv)
    cav_rows, frame_rows = summarize(args.label, rows)

    cav_fields = [
        'label',
        'cav_id',
        'frames',
        'head_frames',
        'member_frames',
        'uploaded_frames',
        'fused_frames',
        'unscheduled_member_frames',
        'upload_rate_when_member',
        'fused_rate',
        'selected_grids',
        'uploaded_points',
        'local_points_when_head',
        'pred_boxes_when_head',
        'gt_boxes_when_head',
    ]
    frame_fields = [
        'label',
        'scenario_id',
        'timestamp',
        'head_count',
        'member_count',
        'uploaded_count',
        'fused_count',
        'unscheduled_member_count',
        'selected_grids',
        'uploaded_points',
        'communication_bytes',
    ]
    write_csv(args.output_cav_csv, cav_rows, cav_fields)
    if args.output_frame_csv:
        write_csv(args.output_frame_csv, frame_rows, frame_fields)

    print(
        'label=%s frames=%d cavs=%d avg_fused_count=%.2f '
        'avg_uploaded_count=%.2f avg_unscheduled_members=%.2f '
        'avg_selected_grids=%.2f output=%s' % (
            args.label,
            len(frame_rows),
            len(cav_rows),
            mean(frame_rows, 'fused_count'),
            mean(frame_rows, 'uploaded_count'),
            mean(frame_rows, 'unscheduled_member_count'),
            mean(frame_rows, 'selected_grids'),
            args.output_cav_csv))


if __name__ == '__main__':
    main()
