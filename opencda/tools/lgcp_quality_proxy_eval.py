# -*- coding: utf-8 -*-
"""
Compute scalable perception-quality proxies for LGCP selective sharing.

The proxy is based on selected CAVs' area confidence and can be used when
large-scale co-simulation only records communication / latency metrics.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP scalable perception-quality proxies.')
    parser.add_argument('--area-records', required=True)
    parser.add_argument('--subset-frame-records', required=True)
    parser.add_argument('--ablation-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--area-quality', default=None)
    parser.add_argument('--confidence-field', default='density_distance')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_agents(value):
    return [item for item in str(value).split(';') if item]


def build_area_tables(area_records, confidence_field):
    table = defaultdict(lambda: defaultdict(dict))
    for row in area_records:
        timestamp = row['timestamp']
        area_id = row['area_id']
        agent_id = str(row['agent_id'])
        if agent_id == '-1':
            continue
        table[timestamp][area_id][agent_id] = float(row[confidence_field])
    return table


def build_area_weights(area_quality):
    weights = defaultdict(dict)
    if not area_quality:
        return weights
    for row in area_quality:
        timestamp = row['timestamp']
        area_id = row['area_id']
        weights[timestamp][area_id] = max(
            weights[timestamp].get(area_id, 0.0),
            float(row.get('gt_count', 0) or 0))
    return weights


def aggregate_area_confidence(area_scores, selected_agents):
    values = [area_scores.get(agent, 0.0) for agent in selected_agents]
    if not values:
        return 0.0, 0.0, 0
    max_conf = max(values)
    noisy_or = 1.0
    active = 0
    for value in values:
        if value > 0:
            active += 1
        noisy_or *= max(0.0, 1.0 - value)
    return max_conf, 1.0 - noisy_or, active


def weighted_mean(items):
    numerator = sum(value * weight for value, weight in items)
    denominator = sum(weight for _, weight in items)
    if denominator <= 0:
        return sum(value for value, _ in items) / float(len(items) or 1)
    return numerator / denominator


def compute_frame_proxies(subset_rows, area_table, area_weights):
    frame_rows = []
    for row in subset_rows:
        timestamp = row['timestamp']
        selected_agents = parse_agents(row['selected_agents'])
        area_items_max = []
        area_items_noisy_or = []
        active_counts = []
        for area_id, area_scores in area_table.get(timestamp, {}).items():
            weight = 1.0 + area_weights.get(timestamp, {}).get(area_id, 0.0)
            max_conf, noisy_or, active_count = aggregate_area_confidence(
                area_scores,
                selected_agents)
            area_items_max.append((max_conf, weight))
            area_items_noisy_or.append((noisy_or, weight))
            active_counts.append(active_count)

        max_proxy = weighted_mean(area_items_max)
        noisy_or_proxy = weighted_mean(area_items_noisy_or)
        coverage_proxy = (
            sum(1 for count in active_counts if count > 0) /
            float(len(active_counts) or 1))
        frame_rows.append(OrderedDict({
            'scenario_id': row['scenario_id'],
            'timestamp': timestamp,
            'method': row['method'],
            'budget': row['budget'],
            'selected_count': row['selected_count'],
            'non_ego_selected_count': row.get('non_ego_selected_count', ''),
            'area_count': len(active_counts),
            'area_coverage_proxy': '%.6f' % coverage_proxy,
            'confidence_max_proxy': '%.6f' % max_proxy,
            'confidence_noisy_or_proxy': '%.6f' % noisy_or_proxy,
        }))
    return frame_rows


def summarize_proxies(frame_rows):
    grouped = defaultdict(list)
    for row in frame_rows:
        grouped[(row['method'], row['budget'])].append(row)

    summary_rows = []
    for (method, budget), rows in sorted(grouped.items()):
        summary_rows.append(OrderedDict({
            'method': method,
            'budget': budget,
            'frames': len(rows),
            'area_coverage_proxy_mean': '%.6f' % np.mean([
                float(row['area_coverage_proxy']) for row in rows]),
            'confidence_max_proxy_mean': '%.6f' % np.mean([
                float(row['confidence_max_proxy']) for row in rows]),
            'confidence_noisy_or_proxy_mean': '%.6f' % np.mean([
                float(row['confidence_noisy_or_proxy']) for row in rows]),
        }))
    return summary_rows


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def corr(x_values, y_values):
    if len(x_values) < 2:
        return '', ''
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return '', ''
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = rankdata(x)
    ry = rankdata(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return '%.6f' % pearson, '%.6f' % spearman


def join_proxy_with_ap(proxy_summary, ablation_summary):
    ap_by_key = {
        (row['method'], row['budget']): row
        for row in ablation_summary
    }
    joined = []
    for row in proxy_summary:
        ap_row = ap_by_key.get((row['method'], row['budget']))
        if not ap_row:
            continue
        out = OrderedDict(row)
        out['ap_03'] = ap_row['ap_03']
        out['ap_05'] = ap_row['ap_05']
        out['ap_07'] = ap_row['ap_07']
        joined.append(out)
    return joined


def compute_correlations(joined_rows):
    proxy_fields = [
        'area_coverage_proxy_mean',
        'confidence_max_proxy_mean',
        'confidence_noisy_or_proxy_mean',
    ]
    ap_fields = ['ap_03', 'ap_05', 'ap_07']
    rows = []
    for proxy_field in proxy_fields:
        for ap_field in ap_fields:
            pairs = [
                (float(row[proxy_field]), float(row[ap_field]))
                for row in joined_rows
                if row.get(proxy_field) not in ('', None)
                and row.get(ap_field) not in ('', None)
            ]
            pearson, spearman = corr(
                [item[0] for item in pairs],
                [item[1] for item in pairs])
            rows.append(OrderedDict({
                'proxy': proxy_field,
                'quality': ap_field,
                'samples': len(pairs),
                'pearson': pearson,
                'spearman': spearman,
            }))
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    area_records = read_csv(args.area_records)
    subset_rows = read_csv(args.subset_frame_records)
    ablation_summary = read_csv(args.ablation_summary)
    area_quality = read_csv(args.area_quality) if args.area_quality else []

    area_table = build_area_tables(area_records, args.confidence_field)
    area_weights = build_area_weights(area_quality)
    frame_rows = compute_frame_proxies(
        subset_rows,
        area_table,
        area_weights)
    proxy_summary = summarize_proxies(frame_rows)
    joined_rows = join_proxy_with_ap(proxy_summary, ablation_summary)
    correlation_rows = compute_correlations(joined_rows)

    write_csv(os.path.join(args.output_dir, 'quality_proxy_frame_records.csv'),
              ['scenario_id', 'timestamp', 'method', 'budget',
               'selected_count', 'non_ego_selected_count', 'area_count',
               'area_coverage_proxy', 'confidence_max_proxy',
               'confidence_noisy_or_proxy'],
              frame_rows)
    write_csv(os.path.join(args.output_dir, 'quality_proxy_summary.csv'),
              ['method', 'budget', 'frames', 'area_coverage_proxy_mean',
               'confidence_max_proxy_mean',
               'confidence_noisy_or_proxy_mean'],
              proxy_summary)
    write_csv(os.path.join(args.output_dir, 'quality_proxy_ap_joined.csv'),
              ['method', 'budget', 'frames', 'area_coverage_proxy_mean',
               'confidence_max_proxy_mean', 'confidence_noisy_or_proxy_mean',
               'ap_03', 'ap_05', 'ap_07'],
              joined_rows)
    write_csv(os.path.join(args.output_dir, 'quality_proxy_correlation.csv'),
              ['proxy', 'quality', 'samples', 'pearson', 'spearman'],
              correlation_rows)

    config = {
        'area_records': os.path.abspath(args.area_records),
        'subset_frame_records': os.path.abspath(args.subset_frame_records),
        'ablation_summary': os.path.abspath(args.ablation_summary),
        'area_quality': None if args.area_quality is None
        else os.path.abspath(args.area_quality),
        'confidence_field': args.confidence_field,
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Scalable Quality Proxy\n\n')
        stream.write('This run estimates scalable perception-quality proxies ')
        stream.write('from selected CAV area confidence and compares them with ')
        stream.write('offline AP across methods / budgets.\n\n')
        stream.write('- frame_records: `%d`\n' % len(frame_rows))
        stream.write('- summary_rows: `%d`\n' % len(proxy_summary))
        stream.write('- correlation_rows: `%d`\n' % len(correlation_rows))

    print('Wrote quality proxy evaluation to %s' % args.output_dir)
    for row in correlation_rows:
        print('%s vs %s pearson=%s spearman=%s samples=%s' % (
            row['proxy'],
            row['quality'],
            row['pearson'],
            row['spearman'],
            row['samples']))


if __name__ == '__main__':
    main()
