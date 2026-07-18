# -*- coding: utf-8 -*-
"""
Compute raw LiDAR byte accounting for flat selective-sharing ablations.

Existing LGCP flat baselines report a fixed proxy such as 10KB per selected
non-ego agent. This tool keeps the same selected-agent decisions and estimates
their raw LiDAR upload bytes from the dumped PCD point counts, making the byte
accounting more comparable with LGCP raw area-slice plans.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate raw LiDAR bytes for flat sharing baselines.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--subset-frame-records', required=True,
                        help='subset_frame_records.csv from subset ablation.')
    parser.add_argument('--ablation-summary', required=True,
                        help='ablation_summary.csv to join AP fields.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--bytes-per-point', type=int, default=16)
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


def cav_point_count(dataset, scenario_id, timestamp, cav_id):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=cav_id,
        cav_ids=[cav_id],
        add_transformation=False)
    key = next(iter(frame.keys()))
    return int(frame[key]['lidar_np'].shape[0])


def build_ap_lookup(summary_rows):
    lookup = {}
    for row in summary_rows:
        lookup[(row['method'], row['budget'])] = row
    return lookup


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = OPV2VFrameDataset(args.dataset_root)
    frame_rows = read_csv(args.subset_frame_records)
    ap_lookup = build_ap_lookup(read_csv(args.ablation_summary))
    point_cache = {}

    detail_rows = []
    stats = defaultdict(lambda: {
        'frames': 0,
        'selected_agents': 0,
        'non_ego_agents': 0,
        'raw_points': 0,
        'raw_bytes': 0,
    })

    for row in frame_rows:
        scenario_id = row.get('scenario_id') or args.scenario_id
        timestamp = row['timestamp']
        method = row['method']
        budget = row['budget']
        selected = parse_agents(row['selected_agents'])
        non_ego = [agent for agent in selected
                   if str(agent) != str(args.ego_cav_id)]

        raw_points = 0
        for agent in non_ego:
            key = (scenario_id, timestamp, str(agent))
            if key not in point_cache:
                point_cache[key] = cav_point_count(
                    dataset,
                    scenario_id,
                    timestamp,
                    str(agent))
            raw_points += point_cache[key]
        raw_bytes = raw_points * args.bytes_per_point

        detail_rows.append(OrderedDict({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'method': method,
            'budget': budget,
            'selected_agents': ';'.join(selected),
            'non_ego_agents': ';'.join(non_ego),
            'non_ego_count': len(non_ego),
            'raw_points': raw_points,
            'raw_bytes': raw_bytes,
        }))

        item = stats[(method, budget)]
        item['frames'] += 1
        item['selected_agents'] += len(selected)
        item['non_ego_agents'] += len(non_ego)
        item['raw_points'] += raw_points
        item['raw_bytes'] += raw_bytes

    summary_rows = []
    for (method, budget), item in sorted(stats.items()):
        frames = max(1, item['frames'])
        ap = ap_lookup.get((method, budget), {})
        summary_rows.append(OrderedDict({
            'method': method,
            'budget': budget,
            'frames': item['frames'],
            'selected_mean': '%.6f' % (
                item['selected_agents'] / float(frames)),
            'non_ego_selected_mean': '%.6f' % (
                item['non_ego_agents'] / float(frames)),
            'raw_points_total': item['raw_points'],
            'raw_points_mean': '%.6f' % (
                item['raw_points'] / float(frames)),
            'raw_bytes_total': item['raw_bytes'],
            'raw_bytes_mean': '%.6f' % (
                item['raw_bytes'] / float(frames)),
            'bytes_per_point': args.bytes_per_point,
            'ap_03': ap.get('ap_03', ''),
            'ap_05': ap.get('ap_05', ''),
            'ap_07': ap.get('ap_07', ''),
            'gt_total': ap.get('gt_total', ''),
            'pred_samples': ap.get('pred_samples', ''),
        }))

    write_csv(os.path.join(args.output_dir, 'flat_raw_byte_frame_records.csv'),
              ['scenario_id', 'timestamp', 'method', 'budget',
               'selected_agents', 'non_ego_agents', 'non_ego_count',
               'raw_points', 'raw_bytes'],
              detail_rows)
    write_csv(os.path.join(args.output_dir, 'flat_raw_byte_summary.csv'),
              ['method', 'budget', 'frames', 'selected_mean',
               'non_ego_selected_mean', 'raw_points_total',
               'raw_points_mean', 'raw_bytes_total', 'raw_bytes_mean',
               'bytes_per_point', 'ap_03', 'ap_05', 'ap_07',
               'gt_total', 'pred_samples'],
              summary_rows)

    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'subset_frame_records': os.path.abspath(args.subset_frame_records),
        'ablation_summary': os.path.abspath(args.ablation_summary),
        'ego_cav_id': str(args.ego_cav_id),
        'bytes_per_point': args.bytes_per_point,
        'note': 'Raw LiDAR bytes for existing flat selected-agent decisions.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# Flat Raw Byte Accounting\n\n')
        stream.write('This run preserves the existing flat baseline selected ')
        stream.write('agents and estimates upload bytes from raw PCD point ')
        stream.write('counts. It is used only for common byte-accounting ')
        stream.write('analysis; it does not rerun perception.\n\n')
        stream.write('- frame rows: `%d`\n' % len(detail_rows))
        stream.write('- summary rows: `%d`\n' % len(summary_rows))

    print('Wrote flat raw byte accounting to %s' % args.output_dir)
    for row in summary_rows:
        print('%s budget=%s raw_bytes_mean=%s AP50=%s' % (
            row['method'],
            row['budget'],
            row['raw_bytes_mean'],
            row['ap_05']))


if __name__ == '__main__':
    main()
