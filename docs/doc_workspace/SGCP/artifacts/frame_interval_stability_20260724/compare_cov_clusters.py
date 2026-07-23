# -*- coding: utf-8 -*-
"""Compare SGCP-CV coalition results under two stability horizons."""

import argparse
import csv
import os
from collections import Counter

import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    clear_sgcp_globals,
)
from opencda.core.clustering.algorithms.clustering.cov_coalition_game import (
    COVCoalitionGame,
)


def load_protocol(dataset, scenario_id):
    protocol_path = os.path.join(
        dataset.scenarios[scenario_id]['path'],
        'data_protocol.yaml')
    with open(protocol_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)


def fixed_delta_from_protocol(protocol, fallback=0.05):
    try:
        return float(protocol['world']['fixed_delta_seconds'])
    except (KeyError, TypeError, ValueError):
        return fallback


def frame_interval_seconds(timestamps, fixed_delta_seconds):
    if len(timestamps) >= 2:
        return (int(timestamps[1]) - int(timestamps[0])) * fixed_delta_seconds
    return fixed_delta_seconds


def cav_sort_key(cav_id):
    try:
        return (0, int(cav_id))
    except ValueError:
        return (1, str(cav_id))


def select_cav_ids(dataset, scenario_id, ego_cav_id):
    cav_ids = sorted(dataset.scenarios[scenario_id]['cav_ids'],
                     key=cav_sort_key)
    ego_id = str(ego_cav_id)
    if ego_id not in cav_ids:
        cav_ids = [ego_id] + cav_ids
    return cav_ids


def canonical_clusters(frame, protocol, ego_cav_id, rho_th, n_max,
                       stability_horizon_s):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=str(ego_cav_id),
        protocol=protocol,
        density_threshold=rho_th)
    algorithm = COVCoalitionGame(world)
    algorithm.p.N_max = n_max
    algorithm.p.T_min_stab = stability_horizon_s
    clusters = algorithm.run()
    rows = []
    for cluster in clusters:
        rows.append((
            int(cluster.head_id),
            tuple(sorted(int(member_id) for member_id in cluster.members))))
    return tuple(sorted(rows, key=lambda item: (item[1], item[0])))


def format_clusters(clusters):
    return '|'.join(
        '%s:%s' % (head_id, ','.join(str(item) for item in members))
        for head_id, members in clusters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default=r'D:\Data\Carla')
    parser.add_argument('--scenario-id', default='2026_07_15_01_26_56')
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--max-frames', type=int, default=41)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--old-stability-s', type=float, default=1.0)
    parser.add_argument('--n-max', type=int, default=4)
    parser.add_argument('--rho-th', type=float, default=3.0)
    parser.add_argument('--output-csv', default=None)
    args = parser.parse_args()

    dataset = OPV2VFrameDataset(args.dataset_root)
    protocol = load_protocol(dataset, args.scenario_id)
    timestamps_all = dataset.scenarios[args.scenario_id]['timestamps']
    selected = (timestamps_all[args.start_index:]
                if args.max_frames == 0 else
                timestamps_all[args.start_index:
                               args.start_index + args.max_frames])
    inferred_dt = frame_interval_seconds(
        timestamps_all[args.start_index:args.start_index + 2],
        fixed_delta_from_protocol(protocol))
    cav_ids = select_cav_ids(dataset, args.scenario_id, args.ego_cav_id)

    rows = []
    mismatch_count = 0
    cluster_count_pairs = Counter()
    for timestamp in selected:
        frame = dataset.load_frame(
            args.scenario_id,
            timestamp,
            ego_cav_id=args.ego_cav_id,
            cav_ids=cav_ids)
        old_clusters = canonical_clusters(
            frame,
            protocol,
            args.ego_cav_id,
            args.rho_th,
            args.n_max,
            args.old_stability_s)
        new_clusters = canonical_clusters(
            frame,
            protocol,
            args.ego_cav_id,
            args.rho_th,
            args.n_max,
            inferred_dt)
        equivalent = old_clusters == new_clusters
        mismatch_count += 0 if equivalent else 1
        cluster_count_pairs[(len(old_clusters), len(new_clusters))] += 1
        rows.append({
            'timestamp': timestamp,
            'old_stability_s': args.old_stability_s,
            'new_stability_s': inferred_dt,
            'old_cluster_count': len(old_clusters),
            'new_cluster_count': len(new_clusters),
            'equivalent': equivalent,
            'old_clusters': format_clusters(old_clusters),
            'new_clusters': format_clusters(new_clusters),
        })

    if args.output_csv:
        output_dir = os.path.dirname(os.path.abspath(args.output_csv))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(args.output_csv, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print('scenario=%s' % args.scenario_id)
    print('frames=%s range=%s..%s' %
          (len(selected), selected[0], selected[-1]))
    print('fixed_delta_seconds=%.6f' % fixed_delta_from_protocol(protocol))
    print('inferred_frame_interval_s=%.6f' % inferred_dt)
    print('old_stability_s=%.6f new_stability_s=%.6f' %
          (args.old_stability_s, inferred_dt))
    print('n_max=%s rho_th=%s' % (args.n_max, args.rho_th))
    print('cluster_count_pairs=%s' % dict(cluster_count_pairs))
    print('mismatch_frames=%s' % mismatch_count)
    if mismatch_count:
        for row in rows:
            if not row['equivalent']:
                print('first_mismatch=%s' % row['timestamp'])
                print('old=%s' % row['old_clusters'])
                print('new=%s' % row['new_clusters'])
                break


if __name__ == '__main__':
    main()
