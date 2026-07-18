# -*- coding: utf-8 -*-
"""Summarize network-level satisfaction from SGCP object diagnostics.

The input is the per-GT CSV written by:

    python -m opencda.tools.offline_inference --object-diagnostics-output ...

For each receiver-frame sample, we compare the method detections against the
full 20-CAV reference detections already stored in the diagnostics file.  The
default satisfaction metric is the fraction of full-reference-detectable GT
objects that are also detected by the method.  This gives a network-level
coverage/recovery metric that is complementary to AP.
"""

import argparse
import csv
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Summarize SGCP/CP satisfaction from object diagnostics.')
    parser.add_argument('--object-csv', action='append', required=True,
                        help='Object diagnostics CSV. Can be repeated. Use '
                             'label=path to set the method label.')
    parser.add_argument('--satisfaction-threshold', type=float, default=0.70,
                        help='Receiver-frame satisfaction threshold on '
                             'full-reference recovery. Defaults to 0.70.')
    parser.add_argument('--sample-output', default=None,
                        help='Optional per receiver-frame output CSV.')
    parser.add_argument('--summary-output', default=None,
                        help='Optional per-method summary output CSV.')
    return parser.parse_args()


def split_label_path(value):
    if '=' in value:
        label, path = value.split('=', 1)
        label = label.strip()
        path = path.strip()
        if label:
            return label, path
    path = value.strip()
    label = os.path.splitext(os.path.basename(path))[0]
    return label, path


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


def sample_key(row):
    return (
        row.get('scenario_id', ''),
        row.get('timestamp', ''),
        row.get('sample_label', ''),
        row.get('receiver_id', ''),
    )


def load_sample_rows(label, path, threshold):
    groups = defaultdict(list)
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            groups[sample_key(row)].append(row)

    samples = []
    for key, rows in sorted(groups.items()):
        scenario_id, timestamp, sample_label, receiver_id = key
        gt_total = len(rows)
        full_detectable = sum(
            parse_int(row.get('full_reference_matched')) for row in rows)
        method_matched = sum(
            parse_int(row.get('method_matched')) for row in rows)
        recovered_full = sum(
            1
            for row in rows
            if parse_int(row.get('full_reference_matched')) and
            parse_int(row.get('method_matched')))
        full_missed = max(0, full_detectable - recovered_full)
        if full_detectable > 0:
            recovery_rate = recovered_full / float(full_detectable)
            valid = 1
        else:
            recovery_rate = ''
            valid = 0
        method_recall_all = (
            method_matched / float(gt_total) if gt_total > 0 else '')
        full_recall_all = (
            full_detectable / float(gt_total) if gt_total > 0 else '')
        communication_bytes = parse_int(rows[0].get('communication_bytes'))
        samples.append({
            'method': label,
            'source_csv': path,
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'sample_label': sample_label,
            'receiver_id': receiver_id,
            'resource_allocation': rows[0].get('resource_allocation', ''),
            'gt_total': gt_total,
            'full_detectable': full_detectable,
            'method_matched': method_matched,
            'recovered_full': recovered_full,
            'full_missed': full_missed,
            'full_reference_recall_all': full_recall_all,
            'method_recall_all': method_recall_all,
            'full_reference_recovery_rate': recovery_rate,
            'satisfied': (
                int(valid and recovery_rate >= threshold)
                if recovery_rate != '' else ''),
            'communication_bytes': communication_bytes,
        })
    return samples


def numeric_values(rows, field):
    values = []
    for row in rows:
        value = row.get(field, '')
        if value == '':
            continue
        values.append(float(value))
    return values


def percentile(values, pct):
    if not values:
        return ''
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = pct / 100.0 * (len(values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values):
    if not values:
        return ''
    return sum(values) / float(len(values))


def summarize(label, samples, threshold):
    valid_samples = [
        row for row in samples
        if row.get('full_reference_recovery_rate') != ''
    ]
    recovery = numeric_values(samples, 'full_reference_recovery_rate')
    method_recall = numeric_values(samples, 'method_recall_all')
    full_recall = numeric_values(samples, 'full_reference_recall_all')
    satisfied_count = sum(parse_int(row.get('satisfied')) for row in samples)
    valid_count = len(valid_samples)
    total_comm = sum(parse_int(row.get('communication_bytes'))
                     for row in samples)
    return {
        'method': label,
        'samples': len(samples),
        'valid_samples': valid_count,
        'satisfaction_threshold': threshold,
        'satisfied_samples': satisfied_count,
        'satisfaction_rate': (
            satisfied_count / float(valid_count) if valid_count else ''),
        'mean_full_reference_recovery': mean(recovery),
        'p10_full_reference_recovery': percentile(recovery, 10),
        'p50_full_reference_recovery': percentile(recovery, 50),
        'p90_full_reference_recovery': percentile(recovery, 90),
        'mean_method_recall_all_gt': mean(method_recall),
        'mean_full_reference_recall_all_gt': mean(full_recall),
        'total_gt': sum(parse_int(row.get('gt_total')) for row in samples),
        'total_full_detectable': sum(
            parse_int(row.get('full_detectable')) for row in samples),
        'total_recovered_full': sum(
            parse_int(row.get('recovered_full')) for row in samples),
        'total_full_missed': sum(
            parse_int(row.get('full_missed')) for row in samples),
        'total_communication_bytes': total_comm,
        'mean_communication_bytes_per_sample': (
            total_comm / float(len(samples)) if samples else ''),
    }


def write_csv(path, rows, fieldnames):
    if not path:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_value(value):
    if value == '':
        return ''
    if isinstance(value, float):
        return '%.6f' % value
    return str(value)


def main():
    args = parse_args()
    all_samples = []
    summaries = []
    for item in args.object_csv:
        label, path = split_label_path(item)
        samples = load_sample_rows(
            label,
            path,
            args.satisfaction_threshold)
        all_samples.extend(samples)
        summaries.append(summarize(
            label,
            samples,
            args.satisfaction_threshold))

    sample_fields = [
        'method',
        'source_csv',
        'scenario_id',
        'timestamp',
        'sample_label',
        'receiver_id',
        'resource_allocation',
        'gt_total',
        'full_detectable',
        'method_matched',
        'recovered_full',
        'full_missed',
        'full_reference_recall_all',
        'method_recall_all',
        'full_reference_recovery_rate',
        'satisfied',
        'communication_bytes',
    ]
    summary_fields = [
        'method',
        'samples',
        'valid_samples',
        'satisfaction_threshold',
        'satisfied_samples',
        'satisfaction_rate',
        'mean_full_reference_recovery',
        'p10_full_reference_recovery',
        'p50_full_reference_recovery',
        'p90_full_reference_recovery',
        'mean_method_recall_all_gt',
        'mean_full_reference_recall_all_gt',
        'total_gt',
        'total_full_detectable',
        'total_recovered_full',
        'total_full_missed',
        'total_communication_bytes',
        'mean_communication_bytes_per_sample',
    ]

    write_csv(args.sample_output, all_samples, sample_fields)
    write_csv(args.summary_output, summaries, summary_fields)

    for summary in summaries:
        print('method=%s samples=%s valid=%s satisfaction@%.2f=%s '
              'mean_recovery=%s p10=%s p50=%s comm_bytes=%s' % (
                  summary['method'],
                  summary['samples'],
                  summary['valid_samples'],
                  args.satisfaction_threshold,
                  format_value(summary['satisfaction_rate']),
                  format_value(summary['mean_full_reference_recovery']),
                  format_value(summary['p10_full_reference_recovery']),
                  format_value(summary['p50_full_reference_recovery']),
                  summary['total_communication_bytes']))


if __name__ == '__main__':
    main()
