# -*- coding: utf-8 -*-
"""
Estimate LGCP CAV / RSU computation-capacity sensitivity.

The proxy uses hierarchy_frame_summary.csv:
  - leader_max_load approximates the worst CAV leader local-fusion workload.
  - covered_area_count approximates RSU global aggregation workload.

Capacities are abstract workload units per millisecond. This keeps the tool
model-agnostic while exposing the capacity assumptions in config.yaml.
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP compute-capacity latency proxy.')
    parser.add_argument('--hierarchy-frame-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cav-capacities', default='2,4,8,16',
                        help='CAV leader fusion workload units per ms.')
    parser.add_argument('--rsu-capacities', default='10,20,40,80',
                        help='RSU aggregation area units per ms.')
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


def parse_list(text):
    return [float(item.strip()) for item in text.split(',') if item.strip()]


def summarize(values):
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.percentile(values, 95)),
        float(np.max(values)),
    )


def evaluate(frame_rows, cav_capacities, rsu_capacities):
    detail_rows = []
    summary_rows = []
    for cav_capacity in cav_capacities:
        for rsu_capacity in rsu_capacities:
            local_latencies = []
            rsu_latencies = []
            total_latencies = []
            bottlenecks = []
            for row in frame_rows:
                local_units = as_float(row.get('leader_max_load'))
                rsu_units = as_float(row.get('covered_area_count'))
                local_ms = local_units / cav_capacity if cav_capacity > 0 else 0.0
                rsu_ms = rsu_units / rsu_capacity if rsu_capacity > 0 else 0.0
                total_ms = local_ms + rsu_ms
                bottleneck = 'cav_local_fusion' if local_ms >= rsu_ms else 'rsu_aggregation'
                local_latencies.append(local_ms)
                rsu_latencies.append(rsu_ms)
                total_latencies.append(total_ms)
                bottlenecks.append(bottleneck)
                detail_rows.append(OrderedDict({
                    'cav_capacity_units_per_ms': '%.6f' % cav_capacity,
                    'rsu_capacity_areas_per_ms': '%.6f' % rsu_capacity,
                    'timestamp': row['timestamp'],
                    'leader_max_load': '%.6f' % local_units,
                    'covered_area_count': '%.6f' % rsu_units,
                    'local_fusion_ms': '%.6f' % local_ms,
                    'rsu_aggregation_ms': '%.6f' % rsu_ms,
                    'compute_latency_ms': '%.6f' % total_ms,
                    'bottleneck': bottleneck,
                }))
            local_mean, local_p95, local_max = summarize(local_latencies)
            rsu_mean, rsu_p95, rsu_max = summarize(rsu_latencies)
            total_mean, total_p95, total_max = summarize(total_latencies)
            cav_bottleneck_ratio = (
                bottlenecks.count('cav_local_fusion') / float(len(bottlenecks))
                if bottlenecks else 0.0)
            summary_rows.append(OrderedDict({
                'cav_capacity_units_per_ms': '%.6f' % cav_capacity,
                'rsu_capacity_areas_per_ms': '%.6f' % rsu_capacity,
                'frames': len(frame_rows),
                'local_fusion_ms_mean': '%.6f' % local_mean,
                'local_fusion_ms_p95': '%.6f' % local_p95,
                'local_fusion_ms_max': '%.6f' % local_max,
                'rsu_aggregation_ms_mean': '%.6f' % rsu_mean,
                'rsu_aggregation_ms_p95': '%.6f' % rsu_p95,
                'rsu_aggregation_ms_max': '%.6f' % rsu_max,
                'compute_latency_ms_mean': '%.6f' % total_mean,
                'compute_latency_ms_p95': '%.6f' % total_p95,
                'compute_latency_ms_max': '%.6f' % total_max,
                'cav_bottleneck_frame_ratio': '%.6f' % cav_bottleneck_ratio,
            }))
    return detail_rows, summary_rows


def write_notes(path, summary_rows):
    with open(path, 'w') as stream:
        stream.write('# LGCP Compute Capacity Sensitivity\n\n')
        stream.write('This run estimates local-fusion and RSU-aggregation ')
        stream.write('compute latency from hierarchy frame summaries.\n\n')
        stream.write('| CAV cap | RSU cap | Mean compute ms | Max compute ms | CAV bottleneck ratio |\n')
        stream.write('| ---: | ---: | ---: | ---: | ---: |\n')
        for row in summary_rows:
            stream.write('| %s | %s | %s | %s | %s |\n' % (
                row['cav_capacity_units_per_ms'],
                row['rsu_capacity_areas_per_ms'],
                row['compute_latency_ms_mean'],
                row['compute_latency_ms_max'],
                row['cav_bottleneck_frame_ratio']))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    frame_rows = read_csv(args.hierarchy_frame_summary)
    cav_capacities = parse_list(args.cav_capacities)
    rsu_capacities = parse_list(args.rsu_capacities)
    detail_rows, summary_rows = evaluate(
        frame_rows, cav_capacities, rsu_capacities)

    write_csv(os.path.join(args.output_dir, 'compute_capacity_frames.csv'),
              list(detail_rows[0].keys()), detail_rows)
    write_csv(os.path.join(args.output_dir, 'compute_capacity_summary.csv'),
              list(summary_rows[0].keys()), summary_rows)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump({
            'hierarchy_frame_summary': os.path.abspath(
                args.hierarchy_frame_summary),
            'cav_capacities_units_per_ms': cav_capacities,
            'rsu_capacities_areas_per_ms': rsu_capacities,
            'note': 'Compute latency proxy; units are abstract workload units.',
        }, stream, sort_keys=False)
    write_notes(os.path.join(args.output_dir, 'notes.md'), summary_rows)

    for row in summary_rows:
        print('cav_cap=%s rsu_cap=%s compute_mean_ms=%s compute_max_ms=%s bottleneck_ratio=%s' % (
            row['cav_capacity_units_per_ms'],
            row['rsu_capacity_areas_per_ms'],
            row['compute_latency_ms_mean'],
            row['compute_latency_ms_max'],
            row['cav_bottleneck_frame_ratio']))


if __name__ == '__main__':
    main()
