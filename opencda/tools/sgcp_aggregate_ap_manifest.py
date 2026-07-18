# -*- coding: utf-8 -*-
"""Build aggregate-AP manifest rows from offline inference logs.

This tool does not recompute AP.  It extracts the pooled evaluator AP reported
by ``offline_inference`` and joins it with trace metadata so paper tables can
record the receiver-frame scope and fusion scaffold for every result.
"""

import argparse
import csv
import os
import re
from collections import Counter


AP_PATTERN = re.compile(
    r'The Average Precision at IOU 0\.3 is\s+([0-9.]+),\s+'
    r'The Average Precision at IOU 0\.5 is\s+([0-9.]+),\s+'
    r'The Average Precision at IOU 0\.7 is\s+([0-9.]+)')
CP_COUNTER_PATTERN = re.compile(r'cp counter:\s*(\d+)')
FUSION_METHOD_PATTERN = re.compile(r'- Fusion method:\s*(\S+)')
SGCP_SUMMARY_PATTERN = re.compile(
    r'sgcp_summary\s+frames=(?P<trace_rows>\d+)\s+'
    r'avg_comm_bytes=(?P<avg_comm>[0-9.]+)\s+'
    r'total_comm_bytes=(?P<total_comm>\d+)\s+'
    r'avg_source_cavs=(?P<avg_sources>[0-9.]+)\s+'
    r'avg_selected_grids=(?P<avg_grids>[0-9.]+)')
FIELDNAMES = [
    'label',
    'ap_03',
    'ap_05',
    'ap_07',
    'aggregate_ap_scope',
    'evaluated_samples',
    'trace_rows',
    'unique_timestamps',
    'receiver_policy',
    'inter_cluster_late_fusion',
    'fusion_method',
    'resource_allocation',
    'clustering',
    'upload_mode',
    'grid_selection_mode',
    'grid_score_mode',
    'cluster_count_mode',
    'payload_bytes',
    'mbps',
    'avg_comm_bytes_per_trace_row',
    'avg_source_cavs',
    'avg_selected_grids',
    'log_path',
    'trace_path',
    'notes',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create aggregate-AP manifest rows from logs/traces.')
    parser.add_argument('--run', action='append', required=True,
                        help='Run spec label=log_path,trace_path. '
                             'trace_path may be empty if unavailable.')
    parser.add_argument('--output-csv', required=True,
                        help='Manifest CSV output path.')
    parser.add_argument('--frame-interval-s', type=float, default=0.1,
                        help='Seconds per evaluated frame for Mbps. '
                             'Defaults to 0.1 for 10 Hz CP.')
    parser.add_argument('--notes', default='',
                        help='Optional note copied to every row.')
    parser.add_argument('--override', action='append', default=[],
                        help='Override a manifest field with '
                             'label.field=value. May be repeated.')
    return parser.parse_args()


def split_run_spec(spec):
    if '=' not in spec:
        raise ValueError('--run must use label=log_path,trace_path')
    label, rest = spec.split('=', 1)
    parts = rest.split(',', 1)
    log_path = parts[0].strip()
    trace_path = parts[1].strip() if len(parts) > 1 else ''
    if not label.strip() or not log_path:
        raise ValueError('--run must include non-empty label and log path')
    return label.strip(), log_path, trace_path


def split_override_spec(spec):
    if '=' not in spec or '.' not in spec.split('=', 1)[0]:
        raise ValueError('--override must use label.field=value')
    left, value = spec.split('=', 1)
    label, field = left.split('.', 1)
    label = label.strip()
    field = field.strip()
    if not label or not field:
        raise ValueError('--override must include non-empty label and field')
    if field not in FIELDNAMES:
        raise ValueError('Unknown manifest field in override: %s' % field)
    return label, field, value


def parse_overrides(specs):
    overrides = {}
    for spec in specs:
        label, field, value = split_override_spec(spec)
        overrides.setdefault(label, {})[field] = value
    return overrides


def read_text(path):
    with open(path, 'r', errors='replace') as stream:
        return stream.read()


def parse_log(path):
    text = read_text(path)
    ap_matches = AP_PATTERN.findall(text)
    cp_matches = CP_COUNTER_PATTERN.findall(text)
    fusion_matches = FUSION_METHOD_PATTERN.findall(text)
    summary_matches = list(SGCP_SUMMARY_PATTERN.finditer(text))
    result = {
        'ap_03': '',
        'ap_05': '',
        'ap_07': '',
        'cp_counter': '',
        'fusion_method': '',
        'log_summary_trace_rows': '',
        'log_summary_total_comm_bytes': '',
        'log_summary_avg_comm_bytes': '',
        'log_summary_avg_source_cavs': '',
        'log_summary_avg_selected_grids': '',
    }
    if ap_matches:
        result['ap_03'], result['ap_05'], result['ap_07'] = ap_matches[-1]
    if cp_matches:
        result['cp_counter'] = cp_matches[-1]
    if fusion_matches:
        result['fusion_method'] = fusion_matches[-1]
    if summary_matches:
        groups = summary_matches[-1].groupdict()
        result['log_summary_trace_rows'] = groups['trace_rows']
        result['log_summary_total_comm_bytes'] = groups['total_comm']
        result['log_summary_avg_comm_bytes'] = groups['avg_comm']
        result['log_summary_avg_source_cavs'] = groups['avg_sources']
        result['log_summary_avg_selected_grids'] = groups['avg_grids']
    return result


def parse_int(value, default=0):
    try:
        if value == '' or value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value, default=0.0):
    try:
        if value == '' or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def most_common(values):
    values = [value for value in values if value != '']
    if not values:
        return ''
    return Counter(values).most_common(1)[0][0]


def parse_trace(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    timestamps = sorted(set(row.get('timestamp', '') for row in rows))
    comm_values = [parse_int(row.get('communication_bytes')) for row in rows]
    selected_grid_total = 0
    for row in rows:
        # selected_grid_counts_json can contain commas and quotes; rather than
        # parsing JSON here, use log_summary avg_selected_grids when available.
        value = row.get('selected_grid_counts_json', '')
        if value:
            selected_grid_total += value.count(':')
    return {
        'trace_path': path,
        'trace_rows': len(rows),
        'unique_timestamps': len([item for item in timestamps if item != '']),
        'receiver_policy': most_common(
            [row.get('receiver_policy', '') for row in rows]),
        'resource_allocation': most_common(
            [row.get('resource_allocation', '') for row in rows]),
        'clustering': most_common([row.get('clustering', '') for row in rows]),
        'upload_mode': most_common([row.get('upload_mode', '') for row in rows]),
        'grid_selection_mode': most_common(
            [row.get('grid_selection_mode', '') for row in rows]),
        'grid_score_mode': most_common(
            [row.get('grid_score_mode', '') for row in rows]),
        'cluster_count_mode': most_common(
            [row.get('cluster_count', '') for row in rows]),
        'total_trace_comm_bytes': sum(comm_values),
        'avg_trace_comm_bytes': (
            sum(comm_values) / float(len(comm_values)) if rows else ''),
        'selected_grid_nonempty_entries': selected_grid_total,
    }


def compute_mbps(total_bytes, duration_steps, frame_interval_s):
    total_bytes = parse_int(total_bytes)
    duration_steps = parse_int(duration_steps)
    if duration_steps <= 0:
        return ''
    duration_s = duration_steps * frame_interval_s
    if duration_s <= 0:
        return ''
    if total_bytes <= 0:
        return 0.0
    return total_bytes * 8.0 / duration_s / 1e6


def row_for_run(label, log_path, trace_path, frame_interval_s, notes):
    log = parse_log(log_path)
    trace = parse_trace(trace_path)
    evaluated_samples = log.get('cp_counter', '')
    if evaluated_samples == '' and trace.get('unique_timestamps'):
        evaluated_samples = trace.get('unique_timestamps')
    total_bytes = (
        log.get('log_summary_total_comm_bytes') or
        trace.get('total_trace_comm_bytes', ''))
    duration_steps = trace.get('unique_timestamps') or evaluated_samples
    inter_cluster = 'yes' if (
        parse_int(log.get('cp_counter')) and
        trace.get('trace_rows') and
        parse_int(log.get('cp_counter')) < parse_int(trace.get('trace_rows'))
    ) else 'no'
    row = {
        'label': label,
        'ap_03': log.get('ap_03', ''),
        'ap_05': log.get('ap_05', ''),
        'ap_07': log.get('ap_07', ''),
        'aggregate_ap_scope': 'pooled evaluator over evaluated samples',
        'evaluated_samples': evaluated_samples,
        'trace_rows': trace.get('trace_rows', log.get(
            'log_summary_trace_rows', '')),
        'unique_timestamps': trace.get('unique_timestamps', ''),
        'receiver_policy': trace.get('receiver_policy', ''),
        'inter_cluster_late_fusion': inter_cluster,
        'fusion_method': log.get('fusion_method', ''),
        'resource_allocation': trace.get('resource_allocation', ''),
        'clustering': trace.get('clustering', ''),
        'upload_mode': trace.get('upload_mode', ''),
        'grid_selection_mode': trace.get('grid_selection_mode', ''),
        'grid_score_mode': trace.get('grid_score_mode', ''),
        'cluster_count_mode': trace.get('cluster_count_mode', ''),
        'payload_bytes': total_bytes,
        'mbps': compute_mbps(
            total_bytes,
            duration_steps,
            frame_interval_s),
        'avg_comm_bytes_per_trace_row': (
            log.get('log_summary_avg_comm_bytes') or
            trace.get('avg_trace_comm_bytes', '')),
        'avg_source_cavs': log.get('log_summary_avg_source_cavs', ''),
        'avg_selected_grids': log.get('log_summary_avg_selected_grids', ''),
        'log_path': log_path,
        'trace_path': trace_path,
        'notes': notes,
    }
    return row


def apply_overrides(row, overrides, frame_interval_s):
    for field, value in overrides.items():
        row[field] = value
    if ('payload_bytes' in overrides or 'evaluated_samples' in overrides or
            'unique_timestamps' in overrides):
        duration_steps = row.get('unique_timestamps') or row.get(
            'evaluated_samples', '')
        row['mbps'] = compute_mbps(
            row.get('payload_bytes', ''),
            duration_steps,
            frame_interval_s)
    return row


def write_rows(path, rows):
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    overrides = parse_overrides(args.override)
    rows = []
    for spec in args.run:
        label, log_path, trace_path = split_run_spec(spec)
        row = row_for_run(
            label,
            log_path,
            trace_path,
            args.frame_interval_s,
            args.notes)
        row = apply_overrides(
            row,
            overrides.get(label, {}),
            args.frame_interval_s)
        rows.append(row)
    write_rows(args.output_csv, rows)
    for row in rows:
        print('label=%s AP=%s/%s/%s samples=%s trace_rows=%s '
              'late=%s payload=%s Mbps=%s' % (
                  row['label'],
                  row['ap_03'],
                  row['ap_05'],
                  row['ap_07'],
                  row['evaluated_samples'],
                  row['trace_rows'],
                  row['inter_cluster_late_fusion'],
                  row['payload_bytes'],
                  row['mbps']))


if __name__ == '__main__':
    main()
