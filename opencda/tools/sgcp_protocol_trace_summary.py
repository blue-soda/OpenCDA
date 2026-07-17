"""Summarize SGCP receiver-level protocol traces by frame.

The input CSV is produced by:

    python -m opencda.tools.offline_inference --sgcp-trace-output ...

Each input row represents one receiver / cluster-head inference sample.  This
tool keeps the experiment read-only and aggregates those rows into a frame-level
view that is easier to audit beside AP and payload tables.
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Aggregate SGCP protocol trace rows by frame.')
    parser.add_argument('--trace-csv', required=True,
                        help='Receiver-level SGCP trace CSV.')
    parser.add_argument('--output-csv', required=True,
                        help='Frame-level summary CSV to write.')
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


def split_links(value):
    if not value:
        return []
    return [item for item in str(value).split(';') if item]


def summarize_frame(rows):
    scenario_id = rows[0].get('scenario_id', '')
    timestamp = rows[0].get('timestamp', '')
    receiver_ids = []
    uploaded_sources = set()
    fused_cav_ids = set()
    cluster_members = []
    selected_grid_total = 0
    uploaded_point_total = 0
    local_point_total = 0
    communication_bytes = 0
    missing_channel_rows = 0
    missing_channel_sources = set()
    channel_links = set()
    pred_counts = []
    gt_counts = []
    skipped_rows = 0

    for row in rows:
        receiver_id = parse_int(row.get('receiver_id'))
        receiver_ids.append(receiver_id)
        fused_cav_ids.add(receiver_id)

        member_ids = parse_id_list(row.get('cluster_member_ids'))
        cluster_members.append('%s:%s' % (
            receiver_id, ';'.join(str(item) for item in member_ids)))

        row_uploaded = parse_id_list(row.get('uploaded_source_ids'))
        uploaded_sources.update(row_uploaded)
        fused_cav_ids.update(row_uploaded)

        selected_grids = parse_json_int_map(
            row.get('selected_grid_counts_json'))
        selected_grid_total += sum(selected_grids.values())

        point_counts = parse_json_int_map(row.get('point_counts_json'))
        for cav_id, point_count in point_counts.items():
            if cav_id == receiver_id:
                local_point_total += point_count
            elif cav_id in row_uploaded:
                uploaded_point_total += point_count

        communication_bytes += parse_int(row.get('communication_bytes'))

        missing = parse_id_list(row.get('missing_channel_sources'))
        if missing:
            missing_channel_rows += 1
            missing_channel_sources.update(missing)

        channel_links.update(split_links(row.get('channel_allocation')))

        if row.get('pred_boxes') != '':
            pred_counts.append(parse_int(row.get('pred_boxes')))
        if row.get('gt_boxes') != '':
            gt_counts.append(parse_int(row.get('gt_boxes')))
        if row.get('skipped'):
            skipped_rows += 1

    def min_or_blank(values):
        return '' if not values else min(values)

    def max_or_blank(values):
        return '' if not values else max(values)

    return {
        'scenario_id': scenario_id,
        'timestamp': timestamp,
        'receiver_count': len(rows),
        'receiver_ids': ';'.join(str(item) for item in sorted(receiver_ids)),
        'cluster_member_sets': '|'.join(cluster_members),
        'fused_cav_ids': ';'.join(str(item) for item in sorted(fused_cav_ids)),
        'uploaded_source_ids': ';'.join(
            str(item) for item in sorted(uploaded_sources)),
        'uploaded_source_count': len(uploaded_sources),
        'total_communication_bytes': communication_bytes,
        'total_selected_grids': selected_grid_total,
        'total_uploaded_points': uploaded_point_total,
        'total_local_points': local_point_total,
        'channel_link_count': len(channel_links),
        'channel_allocation_links': ';'.join(sorted(channel_links)),
        'missing_channel_rows': missing_channel_rows,
        'missing_channel_sources': ';'.join(
            str(item) for item in sorted(missing_channel_sources)),
        'pred_boxes_sum': sum(pred_counts),
        'pred_boxes_min': min_or_blank(pred_counts),
        'pred_boxes_max': max_or_blank(pred_counts),
        'gt_boxes_sum': sum(gt_counts),
        'gt_boxes_min': min_or_blank(gt_counts),
        'gt_boxes_max': max_or_blank(gt_counts),
        'rows_with_zero_pred': sum(1 for item in pred_counts if item == 0),
        'rows_with_zero_gt': sum(1 for item in gt_counts if item == 0),
        'skipped_rows': skipped_rows,
        'ap_note': 'global_only',
    }


def read_rows(path):
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        return list(reader)


def write_rows(path, rows):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fieldnames = [
        'scenario_id',
        'timestamp',
        'receiver_count',
        'receiver_ids',
        'cluster_member_sets',
        'fused_cav_ids',
        'uploaded_source_ids',
        'uploaded_source_count',
        'total_communication_bytes',
        'total_selected_grids',
        'total_uploaded_points',
        'total_local_points',
        'channel_link_count',
        'channel_allocation_links',
        'missing_channel_rows',
        'missing_channel_sources',
        'pred_boxes_sum',
        'pred_boxes_min',
        'pred_boxes_max',
        'gt_boxes_sum',
        'gt_boxes_min',
        'gt_boxes_max',
        'rows_with_zero_pred',
        'rows_with_zero_gt',
        'skipped_rows',
        'ap_note',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    rows = read_rows(args.trace_csv)
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get('scenario_id', ''), row.get('timestamp', ''))
        grouped[key].append(row)

    summaries = [
        summarize_frame(grouped[key])
        for key in sorted(grouped.keys())
    ]
    write_rows(args.output_csv, summaries)

    total_payload = sum(
        parse_int(row['total_communication_bytes']) for row in summaries)
    missing_rows = sum(parse_int(row['missing_channel_rows'])
                       for row in summaries)
    print('frames=%d trace_rows=%d total_payload_bytes=%d '
          'missing_channel_rows=%d output=%s' % (
              len(summaries), len(rows), total_payload, missing_rows,
              args.output_csv))


if __name__ == '__main__':
    main()
