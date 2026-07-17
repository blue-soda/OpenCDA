"""Summarize SGCP late-fusion box counts from offline inference logs.

The offline SGCP evaluator prints one line per cluster-head source and one
line after inter-cluster late fusion.  This tool turns those stdout logs into a
frame-level CSV so AP changes can be interpreted beside prediction counts,
late-fusion suppression, ground-truth deduplication, and payload.
"""

import argparse
import csv
import os
import re
from collections import defaultdict


SOURCE_RE = re.compile(
    r'frame=(?P<frame>\d+)/(?P<total>\d+)\s+'
    r'late_source=(?P<source>\d+)/(?P<source_total>\d+)\s+'
    r'scenario=(?P<scenario>\S+)\s+'
    r'timestamp=(?P<timestamp>\S+)\s+'
    r'receiver=(?P<receiver>\d+)\s+'
    r'cavs=(?P<cavs>\[[^\]]*\])\s+'
    r'pred_boxes=(?P<pred>\d+)\s+'
    r'gt_boxes=(?P<gt>\d+)\s+'
    r'comm_bytes=(?P<comm>\d+)')

FUSION_RE = re.compile(
    r'sgcp_late_fusion\s+'
    r'frame=(?P<frame>\d+)/(?P<total>\d+)\s+'
    r'scenario=(?P<scenario>\S+)\s+'
    r'timestamp=(?P<timestamp>\S+)\s+'
    r'sources=(?P<sources>\d+)\s+'
    r'fused_pred_boxes=(?P<fused_pred>\d+)\s+'
    r'fused_gt_boxes=(?P<fused_gt>\d+)')

AP_RE = re.compile(
    r'Average Precision at IOU 0\.3 is (?P<ap30>[0-9.]+).*'
    r'Average Precision at IOU 0\.5 is (?P<ap50>[0-9.]+).*'
    r'Average Precision at IOU 0\.7 is (?P<ap70>[0-9.]+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SGCP late-fusion stdout logs by frame.')
    parser.add_argument('--log', required=True,
                        help='offline_inference stdout log.')
    parser.add_argument('--output-csv', required=True,
                        help='Frame-level summary CSV to write.')
    parser.add_argument('--label', default='',
                        help='Experiment label recorded in the output CSV.')
    return parser.parse_args()


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_log(path):
    sources_by_key = defaultdict(list)
    fused_by_key = {}
    ap = {'ap30': '', 'ap50': '', 'ap70': ''}

    with open(path, 'r', errors='replace') as stream:
        for line in stream:
            source_match = SOURCE_RE.search(line)
            if source_match:
                item = source_match.groupdict()
                key = (parse_int(item['frame']), item['scenario'],
                       item['timestamp'])
                sources_by_key[key].append(item)
                continue

            fusion_match = FUSION_RE.search(line)
            if fusion_match:
                item = fusion_match.groupdict()
                key = (parse_int(item['frame']), item['scenario'],
                       item['timestamp'])
                fused_by_key[key] = item
                continue

            ap_match = AP_RE.search(line)
            if ap_match:
                ap = ap_match.groupdict()

    return sources_by_key, fused_by_key, ap


def summarize(label, sources_by_key, fused_by_key, ap):
    rows = []
    all_keys = sorted(set(sources_by_key.keys()) | set(fused_by_key.keys()))
    for frame_index, scenario, timestamp in all_keys:
        sources = sources_by_key.get((frame_index, scenario, timestamp), [])
        fused = fused_by_key.get((frame_index, scenario, timestamp), {})

        source_pred_sum = sum(parse_int(item.get('pred')) for item in sources)
        source_gt_sum = sum(parse_int(item.get('gt')) for item in sources)
        total_comm = sum(parse_int(item.get('comm')) for item in sources)
        fused_pred = parse_int(fused.get('fused_pred'))
        fused_gt = parse_int(fused.get('fused_gt'))
        expected_sources = parse_int(fused.get('sources'))

        receiver_ids = sorted(parse_int(item.get('receiver'))
                              for item in sources)
        rows.append({
            'label': label,
            'frame_index': frame_index,
            'scenario_id': scenario,
            'timestamp': timestamp,
            'source_count': len(sources),
            'expected_source_count': expected_sources,
            'receiver_ids': ';'.join(str(item) for item in receiver_ids),
            'source_pred_sum': source_pred_sum,
            'fused_pred_boxes': fused_pred,
            'suppressed_pred_boxes': source_pred_sum - fused_pred,
            'source_gt_sum': source_gt_sum,
            'fused_gt_boxes': fused_gt,
            'deduplicated_gt_boxes': source_gt_sum - fused_gt,
            'total_comm_bytes': total_comm,
            'ap30': ap.get('ap30', ''),
            'ap50': ap.get('ap50', ''),
            'ap70': ap.get('ap70', ''),
        })
    return rows


def write_rows(path, rows):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fieldnames = [
        'label',
        'frame_index',
        'scenario_id',
        'timestamp',
        'source_count',
        'expected_source_count',
        'receiver_ids',
        'source_pred_sum',
        'fused_pred_boxes',
        'suppressed_pred_boxes',
        'source_gt_sum',
        'fused_gt_boxes',
        'deduplicated_gt_boxes',
        'total_comm_bytes',
        'ap30',
        'ap50',
        'ap70',
    ]
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
    sources_by_key, fused_by_key, ap = parse_log(args.log)
    rows = summarize(args.label, sources_by_key, fused_by_key, ap)
    write_rows(args.output_csv, rows)

    print(
        'label=%s frames=%d avg_source_pred=%.2f avg_fused_pred=%.2f '
        'avg_suppressed_pred=%.2f avg_fused_gt=%.2f total_comm_bytes=%d '
        'ap=%s/%s/%s output=%s' % (
            args.label,
            len(rows),
            mean(rows, 'source_pred_sum'),
            mean(rows, 'fused_pred_boxes'),
            mean(rows, 'suppressed_pred_boxes'),
            mean(rows, 'fused_gt_boxes'),
            sum(parse_int(row['total_comm_bytes']) for row in rows),
            ap.get('ap30', ''),
            ap.get('ap50', ''),
            ap.get('ap70', ''),
            args.output_csv))


if __name__ == '__main__':
    main()
