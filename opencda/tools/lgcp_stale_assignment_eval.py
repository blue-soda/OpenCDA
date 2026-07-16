# -*- coding: utf-8 -*-
"""
Evaluate stale LGCP area-confidence assignments.

The tool reuses area confidence and area quality CSVs. For each target frame, it
compares current area quality against confidence from an earlier frame
(`lag_steps`). This approximates longer update intervals or stale RSU
assignments without rerunning CARLA.
"""

import argparse
import csv
import os
from collections import OrderedDict, defaultdict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP stale assignment sensitivity.')
    parser.add_argument('--area-records', required=True)
    parser.add_argument('--area-quality', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--confidence-field', default='density_distance')
    parser.add_argument('--quality-field', default='recall_05')
    parser.add_argument('--lags', default='0,1,2,3',
                        help='Comma separated frame lags.')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top area overlap cutoff.')
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
        if value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_corr(xs, ys, rank=False):
    if len(xs) < 2 or len(ys) < 2:
        return ''
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return ''
    if rank:
        x = rankdata(x)
        y = rankdata(y)
        if np.std(x) == 0 or np.std(y) == 0:
            return ''
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values):
    return np.argsort(np.argsort(values)).astype(np.float64)


def noisy_or(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    if len(values) == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - values))


def build_confidence(area_records, confidence_field):
    grouped = defaultdict(lambda: {'density_linear': [], confidence_field: []})
    timestamps = []
    seen_timestamps = set()
    for row in area_records:
        timestamp = row['timestamp']
        if timestamp not in seen_timestamps:
            seen_timestamps.add(timestamp)
            timestamps.append(timestamp)
        agent_id = str(row.get('agent_id', ''))
        if agent_id == '-1':
            continue
        grouped[(timestamp, row['area_id'])]['density_linear'].append(
            as_float(row.get('density_linear')))
        grouped[(timestamp, row['area_id'])][confidence_field].append(
            as_float(row.get(confidence_field)))

    confidence = {}
    for key, values in grouped.items():
        linear = np.asarray(values['density_linear'], dtype=np.float64)
        selected = np.asarray(values[confidence_field], dtype=np.float64)
        confidence[key] = {
            'confidence_mean': float(np.mean(linear)) if len(linear) else 0.0,
            'confidence_max': float(np.max(linear)) if len(linear) else 0.0,
            'confidence_noisy_or': noisy_or(linear),
            'selected_mean': float(np.mean(selected)) if len(selected) else 0.0,
            'selected_max': float(np.max(selected)) if len(selected) else 0.0,
        }
    return timestamps, confidence


def build_quality(area_quality, quality_field):
    quality = {}
    for row in area_quality:
        value = as_float(row.get(quality_field), default=np.nan)
        if not np.isfinite(value):
            continue
        if quality_field.startswith('recall') and as_float(row.get('gt_count')) <= 0:
            continue
        if quality_field.startswith('precision') and as_float(row.get('pred_count')) <= 0:
            continue
        quality[(row['timestamp'], row['area_id'])] = value
    return quality


def top_areas(confidence, timestamp, field, top_k):
    rows = [
        (area_id, values[field])
        for (ts, area_id), values in confidence.items()
        if ts == timestamp
    ]
    rows.sort(key=lambda item: item[1], reverse=True)
    return set(area_id for area_id, _ in rows[:top_k])


def evaluate_lag(timestamps, confidence, quality, lag, top_k):
    timestamp_index = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    records = []
    overlap_rows = []
    for target_ts in timestamps:
        source_idx = timestamp_index[target_ts] - lag
        if source_idx < 0:
            continue
        source_ts = timestamps[source_idx]
        current_top = top_areas(confidence, target_ts, 'confidence_noisy_or', top_k)
        stale_top = top_areas(confidence, source_ts, 'confidence_noisy_or', top_k)
        union = current_top | stale_top
        overlap = (len(current_top & stale_top) / float(len(union))
                   if union else 0.0)
        overlap_rows.append(OrderedDict({
            'lag_steps': lag,
            'source_timestamp': source_ts,
            'target_timestamp': target_ts,
            'top_k': top_k,
            'current_top_count': len(current_top),
            'stale_top_count': len(stale_top),
            'top_jaccard': '%.6f' % overlap,
        }))
        for (quality_ts, area_id), quality_value in quality.items():
            if quality_ts != target_ts:
                continue
            conf = confidence.get((source_ts, area_id))
            if not conf:
                continue
            records.append(OrderedDict({
                'lag_steps': lag,
                'source_timestamp': source_ts,
                'target_timestamp': target_ts,
                'area_id': area_id,
                'quality': '%.6f' % quality_value,
                'confidence_mean': '%.6f' % conf['confidence_mean'],
                'confidence_max': '%.6f' % conf['confidence_max'],
                'confidence_noisy_or': '%.6f' % conf['confidence_noisy_or'],
                'selected_mean': '%.6f' % conf['selected_mean'],
                'selected_max': '%.6f' % conf['selected_max'],
            }))
    return records, overlap_rows


def summarize(lag, records, overlap_rows):
    row = OrderedDict({'lag_steps': lag, 'samples': len(records)})
    qualities = [as_float(r['quality']) for r in records]
    for field in [
            'confidence_mean', 'confidence_max', 'confidence_noisy_or',
            'selected_mean', 'selected_max']:
        values = [as_float(r[field]) for r in records]
        pearson = safe_corr(values, qualities)
        spearman = safe_corr(values, qualities, rank=True)
        row[field + '_pearson'] = '' if pearson == '' else '%.6f' % pearson
        row[field + '_spearman'] = '' if spearman == '' else '%.6f' % spearman
    overlaps = [as_float(r['top_jaccard']) for r in overlap_rows]
    row['top_jaccard_mean'] = '%.6f' % (
        float(np.mean(overlaps)) if overlaps else 0.0)
    row['top_jaccard_min'] = '%.6f' % (
        float(np.min(overlaps)) if overlaps else 0.0)
    return row


def write_notes(path, args, summary_rows):
    with open(path, 'w') as stream:
        stream.write('# LGCP Stale Assignment Sensitivity\n\n')
        stream.write('This run compares current area quality against ')
        stream.write('area-confidence reports from earlier frames.\n\n')
        stream.write('- confidence_field: `%s`\n' % args.confidence_field)
        stream.write('- quality_field: `%s`\n' % args.quality_field)
        stream.write('- lags: `%s`\n' % args.lags)
        stream.write('- top_k: `%d`\n\n' % args.top_k)
        stream.write('| Lag | Samples | Noisy-or Spearman | Top-k Jaccard mean |\n')
        stream.write('| --- | ---: | ---: | ---: |\n')
        for row in summary_rows:
            stream.write('| %s | %s | %s | %s |\n' % (
                row['lag_steps'],
                row['samples'],
                row['confidence_noisy_or_spearman'],
                row['top_jaccard_mean']))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    area_records = read_csv(args.area_records)
    area_quality = read_csv(args.area_quality)
    timestamps, confidence = build_confidence(area_records, args.confidence_field)
    quality = build_quality(area_quality, args.quality_field)
    lags = [int(item.strip()) for item in args.lags.split(',') if item.strip()]

    all_records = []
    all_overlaps = []
    summary_rows = []
    for lag in lags:
        records, overlap_rows = evaluate_lag(
            timestamps, confidence, quality, lag, args.top_k)
        all_records.extend(records)
        all_overlaps.extend(overlap_rows)
        summary_rows.append(summarize(lag, records, overlap_rows))

    record_fields = [
        'lag_steps', 'source_timestamp', 'target_timestamp', 'area_id',
        'quality', 'confidence_mean', 'confidence_max',
        'confidence_noisy_or', 'selected_mean', 'selected_max',
    ]
    overlap_fields = [
        'lag_steps', 'source_timestamp', 'target_timestamp', 'top_k',
        'current_top_count', 'stale_top_count', 'top_jaccard',
    ]
    summary_fields = list(summary_rows[0].keys()) if summary_rows else [
        'lag_steps', 'samples']

    write_csv(os.path.join(args.output_dir, 'stale_assignment_records.csv'),
              record_fields, all_records)
    write_csv(os.path.join(args.output_dir, 'stale_topk_overlap.csv'),
              overlap_fields, all_overlaps)
    write_csv(os.path.join(args.output_dir, 'stale_assignment_summary.csv'),
              summary_fields, summary_rows)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump({
            'area_records': os.path.abspath(args.area_records),
            'area_quality': os.path.abspath(args.area_quality),
            'confidence_field': args.confidence_field,
            'quality_field': args.quality_field,
            'lags': lags,
            'top_k': args.top_k,
            'note': 'Offline stale assignment sensitivity; no CARLA rerun.',
        }, stream, sort_keys=False)
    write_notes(os.path.join(args.output_dir, 'notes.md'), args, summary_rows)

    for row in summary_rows:
        print('lag=%s samples=%s noisy_or_spearman=%s top_jaccard_mean=%s' % (
            row['lag_steps'],
            row['samples'],
            row['confidence_noisy_or_spearman'],
            row['top_jaccard_mean']))


if __name__ == '__main__':
    main()
