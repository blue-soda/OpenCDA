# -*- coding: utf-8 -*-
"""Rewrite an LGCP area assignment plan with a limited leader set.

This diagnostic keeps the selected areas and their original high-confidence
members, but forces each frame to use at most K leaders. It is intended for
testing RSU-side leader-packet fusion under a checkpoint-friendly packet count.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.tools.lgcp_pointpillar_rsu_bev_fusion import (
    grouped_by_timestamp,
    read_csv,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reassign an LGCP plan to at most K leaders per frame.')
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-leaders', type=int, default=4)
    parser.add_argument('--max-areas-per-frame', type=int, default=0)
    parser.add_argument('--leader-score',
                        choices=['priority_sum', 'area_count'],
                        default='priority_sum')
    parser.add_argument('--load-weight', type=float, default=8.0,
                        help='Meters of distance penalty per assigned area.')
    parser.add_argument('--member-bonus', type=float, default=30.0,
                        help='Distance bonus if candidate leader is already '
                             'in the original group.')
    return parser.parse_args()


def parse_members(value):
    return [str(item) for item in str(value).split(';') if item != '']


def unique_strings(values):
    return list(OrderedDict((str(value), None) for value in values).keys())


def sort_id(value):
    try:
        return (0, int(value))
    except ValueError:
        return (1, str(value))


def select_frame_leaders(rows, max_leaders, score_mode):
    scores = defaultdict(float)
    counts = defaultdict(int)
    for row in rows:
        leader = str(row['leader_id'])
        scores[leader] += float(row.get('priority', 0) or 0)
        counts[leader] += 1
    if score_mode == 'area_count':
        key_fn = lambda item: (counts[item], scores[item], sort_id(item))
    else:
        key_fn = lambda item: (scores[item], counts[item], sort_id(item))
    ranked = sorted(scores.keys(), key=key_fn, reverse=True)
    return ranked[:max_leaders]


def load_leader_positions(dataset, scenario_id, timestamp, leader_ids):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=leader_ids[0],
        cav_ids=leader_ids,
        add_transformation=False)
    positions = {}
    for cav_id, cav in frame.items():
        positions[str(cav_id)] = np.asarray(
            cav['params']['lidar_pose'][:2],
            dtype=np.float32)
    return positions


def choose_reassigned_leader(row, candidate_leaders, positions, loads,
                             load_weight, member_bonus):
    area_xy = np.asarray([
        float(row['area_center_x']),
        float(row['area_center_y']),
    ], dtype=np.float32)
    original_members = set(parse_members(row['group_members']))
    original_members.add(str(row['leader_id']))
    best = None
    for leader in candidate_leaders:
        position = positions.get(str(leader), area_xy)
        distance = float(np.linalg.norm(position - area_xy))
        bonus = member_bonus if str(leader) in original_members else 0.0
        cost = distance + load_weight * loads[str(leader)] - bonus
        candidate = (cost, loads[str(leader)], sort_id(str(leader)), str(leader))
        if best is None or candidate < best:
            best = candidate
    return best[-1]


def rewrite_plan(rows, dataset, scenario_id, args):
    grouped = grouped_by_timestamp(rows)
    output_rows = []
    summary_rows = []
    for timestamp, frame_rows in grouped.items():
        if args.max_areas_per_frame:
            frame_rows = frame_rows[:args.max_areas_per_frame]
        leaders = select_frame_leaders(
            frame_rows,
            args.max_leaders,
            args.leader_score)
        positions = load_leader_positions(
            dataset,
            scenario_id,
            timestamp,
            leaders)
        loads = defaultdict(int)
        reassigned_counts = defaultdict(int)
        member_counts = []
        for row in frame_rows:
            new_leader = choose_reassigned_leader(
                row,
                leaders,
                positions,
                loads,
                args.load_weight,
                args.member_bonus)
            members = parse_members(row['group_members'])
            members.append(new_leader)
            members = unique_strings(members)
            loads[new_leader] += 1
            reassigned_counts[new_leader] += 1
            member_counts.append(len(members))

            rewritten = OrderedDict(row)
            rewritten['group_members'] = ';'.join(members)
            rewritten['group_size'] = len(members)
            rewritten['leader_id'] = new_leader
            rewritten['member_uploads'] = max(0, len(members) - 1)
            rewritten['leader_uploads'] = 1
            output_rows.append(rewritten)

        summary_rows.append(OrderedDict({
            'timestamp': timestamp,
            'area_count': len(frame_rows),
            'leader_count': len(leaders),
            'leaders': ';'.join(leaders),
            'leader_area_counts': ';'.join(
                '%s:%s' % (leader, reassigned_counts[leader])
                for leader in leaders),
            'avg_group_size': (
                '%.6f' % float(np.mean(member_counts))
                if member_counts else '0.000000'),
            'max_group_size': max(member_counts) if member_counts else 0,
        }))
    return output_rows, summary_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_csv(args.assignment_plan)
    dataset = OPV2VFrameDataset(args.dataset_root)
    output_rows, summary_rows = rewrite_plan(
        rows,
        dataset,
        args.scenario_id,
        args)
    write_csv(
        os.path.join(args.output_dir, 'area_assignment_plan.csv'),
        list(output_rows[0].keys()),
        output_rows)
    write_csv(
        os.path.join(args.output_dir, 'leader_reassignment_summary.csv'),
        list(summary_rows[0].keys()),
        summary_rows)
    print('Wrote limited-leader LGCP plan to %s' % args.output_dir)


if __name__ == '__main__':
    main()
