"""Summarize detector-quality proxies from SGCP protocol traces.

The SGCP trace is receiver-level: each row represents one cluster head after
early fusion.  This tool aggregates those rows into simple quality proxies for
future object-aware fallback design:

* per receiver/source row: pred boxes, GT boxes, pred/GT ratio, payload
* per uploaded CAV: quality of the receiver rows in which it participates

It does not compute AP.  AP remains the OpenCOOD global evaluator output.
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SGCP detector-quality proxy metrics.')
    parser.add_argument('--trace-csv', required=True,
                        help='Receiver-level SGCP trace CSV.')
    parser.add_argument('--output-receiver-csv', required=True,
                        help='Per receiver/source-row summary CSV.')
    parser.add_argument('--output-cav-csv', required=True,
                        help='Per uploaded-CAV quality summary CSV.')
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


def pred_gt_ratio(pred_boxes, gt_boxes):
    if gt_boxes <= 0:
        return 0.0
    return pred_boxes / float(gt_boxes)


def summarize(label, rows):
    receiver_rows = []
    cav_stats = defaultdict(lambda: {
        'upload_rows': 0,
        'receiver_pred_sum': 0,
        'receiver_gt_sum': 0,
        'zero_pred_rows': 0,
        'low_ratio_rows': 0,
        'selected_grids': 0,
        'uploaded_points': 0,
        'communication_bytes': 0,
    })

    for row in rows:
        receiver_id = parse_int(row.get('receiver_id'))
        pred_boxes = parse_int(row.get('pred_boxes'))
        gt_boxes = parse_int(row.get('gt_boxes'))
        ratio = pred_gt_ratio(pred_boxes, gt_boxes)
        uploaded_sources = parse_id_list(row.get('uploaded_source_ids'))
        selected_grids = parse_json_int_map(
            row.get('selected_grid_counts_json'))
        point_counts = parse_json_int_map(row.get('point_counts_json'))
        communication_bytes = parse_int(row.get('communication_bytes'))
        total_selected_grids = sum(selected_grids.values())
        total_uploaded_points = sum(
            point_counts.get(source_id, 0) for source_id in uploaded_sources)
        bytes_per_uploaded = (
            communication_bytes / float(len(uploaded_sources))
            if uploaded_sources else 0.0)

        receiver_rows.append({
            'label': label,
            'scenario_id': row.get('scenario_id', ''),
            'timestamp': row.get('timestamp', ''),
            'receiver_id': receiver_id,
            'uploaded_source_ids': ';'.join(
                str(item) for item in uploaded_sources),
            'uploaded_source_count': len(uploaded_sources),
            'selected_grid_total': total_selected_grids,
            'uploaded_point_total': total_uploaded_points,
            'communication_bytes': communication_bytes,
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes,
            'pred_gt_ratio': ratio,
            'zero_pred': 1 if pred_boxes == 0 else 0,
            'low_pred_gt_ratio': 1 if gt_boxes > 0 and ratio < 0.15 else 0,
        })

        for source_id in uploaded_sources:
            stat = cav_stats[source_id]
            stat['upload_rows'] += 1
            stat['receiver_pred_sum'] += pred_boxes
            stat['receiver_gt_sum'] += gt_boxes
            stat['zero_pred_rows'] += 1 if pred_boxes == 0 else 0
            stat['low_ratio_rows'] += (
                1 if gt_boxes > 0 and ratio < 0.15 else 0)
            stat['selected_grids'] += selected_grids.get(source_id, 0)
            stat['uploaded_points'] += point_counts.get(source_id, 0)
            stat['communication_bytes'] += bytes_per_uploaded

    cav_rows = []
    for cav_id in sorted(cav_stats.keys()):
        stat = cav_stats[cav_id]
        upload_rows = stat['upload_rows']
        cav_rows.append({
            'label': label,
            'cav_id': cav_id,
            'upload_rows': upload_rows,
            'avg_receiver_pred_boxes': (
                stat['receiver_pred_sum'] / float(upload_rows)
                if upload_rows else 0.0),
            'avg_receiver_gt_boxes': (
                stat['receiver_gt_sum'] / float(upload_rows)
                if upload_rows else 0.0),
            'avg_pred_gt_ratio': pred_gt_ratio(
                stat['receiver_pred_sum'],
                stat['receiver_gt_sum']),
            'zero_pred_rows': stat['zero_pred_rows'],
            'low_ratio_rows': stat['low_ratio_rows'],
            'selected_grids': stat['selected_grids'],
            'uploaded_points': stat['uploaded_points'],
            'approx_communication_bytes': int(stat['communication_bytes']),
        })

    return receiver_rows, cav_rows


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
    receiver_rows, cav_rows = summarize(args.label, rows)

    receiver_fields = [
        'label',
        'scenario_id',
        'timestamp',
        'receiver_id',
        'uploaded_source_ids',
        'uploaded_source_count',
        'selected_grid_total',
        'uploaded_point_total',
        'communication_bytes',
        'pred_boxes',
        'gt_boxes',
        'pred_gt_ratio',
        'zero_pred',
        'low_pred_gt_ratio',
    ]
    cav_fields = [
        'label',
        'cav_id',
        'upload_rows',
        'avg_receiver_pred_boxes',
        'avg_receiver_gt_boxes',
        'avg_pred_gt_ratio',
        'zero_pred_rows',
        'low_ratio_rows',
        'selected_grids',
        'uploaded_points',
        'approx_communication_bytes',
    ]
    write_csv(args.output_receiver_csv, receiver_rows, receiver_fields)
    write_csv(args.output_cav_csv, cav_rows, cav_fields)

    print(
        'label=%s receiver_rows=%d cavs=%d avg_pred_gt_ratio=%.4f '
        'zero_pred_rows=%d low_ratio_rows=%d output=%s' % (
            args.label,
            len(receiver_rows),
            len(cav_rows),
            mean(receiver_rows, 'pred_gt_ratio'),
            sum(parse_int(row['zero_pred']) for row in receiver_rows),
            sum(parse_int(row['low_pred_gt_ratio'])
                for row in receiver_rows),
            args.output_receiver_csv))


if __name__ == '__main__':
    main()
