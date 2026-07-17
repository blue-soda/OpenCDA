# -*- coding: utf-8 -*-
"""
Offline subset ablation for LGCP.

This evaluates perception-only selective sharing variants by selecting subsets
of agents from an OPV2V-style OpenCDA dump and running OpenCOOD inference.
It does not model real leader local fusion or RSU aggregation yet.
"""

import argparse
import csv
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import load_coperception_params
from opencood.utils import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run offline LGCP selective-sharing subset ablation.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--area-records', required=True,
                        help='area_records.csv from lgcp_area_confidence_eval.')
    parser.add_argument('--area-quality', default=None,
                        help='Optional area_quality.csv to rank important areas.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--fusion-method', default=None)
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--max-frames', type=int, default=3)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--budgets', default='5,10',
                        help='Comma-separated source-agent budgets.')
    parser.add_argument('--methods', default='full,random,confidence_topk,comm_aware_topk,area_aware_union',
                        help='Comma-separated methods to run.')
    parser.add_argument('--confidence-field', default='density_distance')
    parser.add_argument('--random-seed', type=int, default=7)
    parser.add_argument('--feature-packet-bytes', type=int, default=10000,
                        help='Byte proxy for one selected non-ego feature packet.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_confidence_tables(area_records, confidence_field):
    by_timestamp_agent = defaultdict(lambda: defaultdict(float))
    by_timestamp_area_agent = defaultdict(lambda: defaultdict(dict))
    by_timestamp_agent_position = defaultdict(dict)
    for row in area_records:
        timestamp = row['timestamp']
        area_id = row['area_id']
        agent_id = str(row['agent_id'])
        confidence = float(row[confidence_field])
        by_timestamp_agent[timestamp][agent_id] += confidence
        by_timestamp_area_agent[timestamp][area_id][agent_id] = confidence
        if agent_id not in by_timestamp_agent_position[timestamp]:
            by_timestamp_agent_position[timestamp][agent_id] = (
                float(row['agent_x']),
                float(row['agent_y']))
    return by_timestamp_agent, by_timestamp_area_agent, by_timestamp_agent_position


def build_area_importance(area_quality):
    importance = defaultdict(lambda: defaultdict(float))
    if not area_quality:
        return importance
    for row in area_quality:
        timestamp = row['timestamp']
        area_id = row['area_id']
        importance[timestamp][area_id] = max(
            importance[timestamp][area_id],
            float(row.get('gt_count', 0) or 0))
    return importance


def normalize_agent_ids(cav_ids):
    return [str(cav_id) for cav_id in cav_ids]


def ensure_ego(selected, ego_id):
    selected = [str(agent) for agent in selected]
    if str(ego_id) not in selected:
        selected = [str(ego_id)] + selected
    # Preserve order and remove duplicates.
    return list(OrderedDict((agent, None) for agent in selected).keys())


def select_random(agents, budget, ego_id, rng):
    pool = [agent for agent in agents if agent != str(ego_id)]
    rng.shuffle(pool)
    return ensure_ego(pool[:max(0, budget - 1)], ego_id)[:budget]


def select_confidence_topk(agent_scores, agents, budget, ego_id):
    ranked = sorted(
        [agent for agent in agents if agent != str(ego_id)],
        key=lambda agent: agent_scores.get(agent, 0.0),
        reverse=True)
    return ensure_ego(ranked[:max(0, budget - 1)], ego_id)[:budget]


def select_comm_aware_topk(agent_scores, agent_positions, agents, budget,
                           ego_id):
    ego_pos = agent_positions.get(str(ego_id))

    def score(agent):
        confidence = agent_scores.get(agent, 0.0)
        if ego_pos is None or agent not in agent_positions:
            return confidence
        dx = agent_positions[agent][0] - ego_pos[0]
        dy = agent_positions[agent][1] - ego_pos[1]
        distance = float(np.sqrt(dx * dx + dy * dy))
        distance_cost = 1.0 + distance / 100.0
        return confidence / distance_cost

    ranked = sorted(
        [agent for agent in agents if agent != str(ego_id)],
        key=score,
        reverse=True)
    return ensure_ego(ranked[:max(0, budget - 1)], ego_id)[:budget]


def select_area_aware(area_agent_scores, area_importance, agents, budget,
                      ego_id):
    selected = ensure_ego([], ego_id)
    selected_set = set(selected)
    candidate_areas = sorted(
        area_agent_scores.keys(),
        key=lambda area: (
            area_importance.get(area, 0.0),
            max(area_agent_scores[area].values())
            if area_agent_scores[area] else 0.0),
        reverse=True)

    while len(selected) < budget:
        best_agent = None
        best_gain = 0.0
        for agent in agents:
            if agent in selected_set:
                continue
            gain = 0.0
            for area in candidate_areas:
                scores = area_agent_scores[area]
                before = max([scores.get(sel, 0.0) for sel in selected_set]
                             or [0.0])
                after = max(before, scores.get(agent, 0.0))
                gain += (1.0 + area_importance.get(area, 0.0)) * (after - before)
            if gain > best_gain:
                best_gain = gain
                best_agent = agent
        if best_agent is None:
            break
        selected.append(best_agent)
        selected_set.add(best_agent)
    return selected[:budget]


def run_inference(manager, dataset, scenario_id, timestamp, ego_id, cav_ids):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=ego_id,
        cav_ids=cav_ids)
    ego = next(cav for cav in frame.values() if cav['ego'])
    ego_lidar_pose = ego['params']['lidar_pose']
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        ego_lidar_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    batch_data = manager.to_device(output_dict)
    ret = manager.inference(
        batch_data,
        with_stats=False,
        return_object_ids=True)
    return ret[0], ret[1], ret[2]


def update_stats(result_stat, pred_box_tensor, pred_score, gt_box_tensor):
    for iou in (0.3, 0.5, 0.7):
        eval_utils.calculate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor,
                                   result_stat, iou)


def calculate_ap_safe(result_stat, iou):
    stat = {
        iou: {
            'tp': list(result_stat[iou]['tp']),
            'fp': list(result_stat[iou]['fp']),
            'gt': result_stat[iou]['gt'],
        }
    }
    if stat[iou]['gt'] == 0:
        return ''
    ap, _, _ = eval_utils.calculate_ap(stat, iou)
    return '%.6f' % ap


def update_budget_stats(budget_stats, selected, ego_id, pred_count, gt_count):
    non_ego_count = sum(1 for agent in selected if agent != str(ego_id))
    budget_stats['frames'] += 1
    budget_stats['selected_total'] += len(selected)
    budget_stats['non_ego_selected_total'] += non_ego_count
    budget_stats['pred_total'] += pred_count
    budget_stats['gt_total_observed'] += gt_count


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    budgets = [int(value.strip()) for value in args.budgets.split(',')
               if value.strip()]
    dataset = OPV2VFrameDataset(args.dataset_root)
    scenario = dataset.scenarios[args.scenario_id]
    timestamps = scenario['timestamps']
    selected_timestamps = (
        timestamps[args.start_index:] if args.max_frames == 0
        else timestamps[args.start_index:args.start_index + args.max_frames])
    agents = normalize_agent_ids(scenario['cav_ids'])
    # Current offline OpenCOOD treats RSU as an agent, but this ablation focuses
    # on CAV selective sharing. Keep RSU out unless it is the ego, which it is not.
    agents = [agent for agent in agents if agent != '-1']

    area_records = read_csv(args.area_records)
    area_quality = read_csv(args.area_quality) if args.area_quality else []
    agent_scores, area_agent_scores, agent_positions = build_confidence_tables(
        area_records,
        args.confidence_field)
    area_importance = build_area_importance(area_quality)

    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    rng = np.random.RandomState(args.random_seed)

    valid_methods = {
        'full',
        'random',
        'confidence_topk',
        'comm_aware_topk',
        'area_aware_union',
    }
    methods = [method.strip() for method in args.methods.split(',')
               if method.strip()]
    unknown = sorted(set(methods) - valid_methods)
    if unknown:
        raise ValueError('Unknown methods: %s' % ', '.join(unknown))
    method_stats = {}
    frame_rows = []
    for budget in budgets:
        for method in methods:
            key = (method, budget)
            method_stats[key] = {
                0.3: {'tp': [], 'fp': [], 'gt': 0},
                0.5: {'tp': [], 'fp': [], 'gt': 0},
                0.7: {'tp': [], 'fp': [], 'gt': 0},
                'budget': {
                    'frames': 0,
                    'selected_total': 0,
                    'non_ego_selected_total': 0,
                    'pred_total': 0,
                    'gt_total_observed': 0,
                },
            }

    for timestamp in selected_timestamps:
        for budget in budgets:
            for method in methods:
                if method == 'full':
                    selected = ensure_ego(agents, args.ego_cav_id)
                elif method == 'random':
                    selected = select_random(
                        list(agents), budget, args.ego_cav_id, rng)
                elif method == 'confidence_topk':
                    selected = select_confidence_topk(
                        agent_scores.get(timestamp, {}),
                        list(agents),
                        budget,
                        args.ego_cav_id)
                elif method == 'comm_aware_topk':
                    selected = select_comm_aware_topk(
                        agent_scores.get(timestamp, {}),
                        agent_positions.get(timestamp, {}),
                        list(agents),
                        budget,
                        args.ego_cav_id)
                elif method == 'area_aware_union':
                    selected = select_area_aware(
                        area_agent_scores.get(timestamp, {}),
                        area_importance.get(timestamp, {}),
                        list(agents),
                        budget,
                        args.ego_cav_id)
                else:
                    raise ValueError(method)

                pred_box_tensor, pred_score, gt_box_tensor = run_inference(
                    manager,
                    dataset,
                    args.scenario_id,
                    timestamp,
                    args.ego_cav_id,
                    selected)
                update_stats(method_stats[(method, budget)], pred_box_tensor,
                             pred_score, gt_box_tensor)
                pred_count = 0 if pred_box_tensor is None else int(
                    pred_box_tensor.shape[0])
                gt_count = 0 if gt_box_tensor is None else int(
                    gt_box_tensor.shape[0])
                update_budget_stats(
                    method_stats[(method, budget)]['budget'],
                    selected,
                    args.ego_cav_id,
                    pred_count,
                    gt_count)

                frame_rows.append(OrderedDict({
                    'scenario_id': args.scenario_id,
                    'timestamp': timestamp,
                    'method': method,
                    'budget': budget,
                    'selected_count': len(selected),
                    'non_ego_selected_count': sum(
                        1 for agent in selected if agent != str(args.ego_cav_id)),
                    'selected_agents': ';'.join(selected),
                    'pred_count': pred_count,
                    'gt_count': gt_count,
                }))
                print('timestamp=%s method=%s budget=%s agents=%s pred=%s gt=%s' % (
                    timestamp, method, budget, len(selected),
                    frame_rows[-1]['pred_count'], frame_rows[-1]['gt_count']))

    summary_rows = []
    for (method, budget), stat in sorted(method_stats.items()):
        budget_stat = stat['budget']
        frame_count = max(1, budget_stat['frames'])
        non_ego_packets = budget_stat['non_ego_selected_total']
        summary_rows.append(OrderedDict({
            'method': method,
            'budget': budget,
            'frames': len(selected_timestamps),
            'selected_mean': '%.6f' % (
                budget_stat['selected_total'] / float(frame_count)),
            'non_ego_selected_mean': '%.6f' % (
                non_ego_packets / float(frame_count)),
            'non_ego_packet_total': non_ego_packets,
            'byte_proxy_total': non_ego_packets * args.feature_packet_bytes,
            'ap_03': calculate_ap_safe(stat, 0.3),
            'ap_05': calculate_ap_safe(stat, 0.5),
            'ap_07': calculate_ap_safe(stat, 0.7),
            'gt_total': stat[0.5]['gt'],
            'pred_samples': len(stat[0.5]['tp']),
        }))

    write_csv(os.path.join(args.output_dir, 'subset_frame_records.csv'),
              ['scenario_id', 'timestamp', 'method', 'budget',
               'selected_count', 'non_ego_selected_count', 'selected_agents',
               'pred_count', 'gt_count'],
              frame_rows)
    write_csv(os.path.join(args.output_dir, 'ablation_summary.csv'),
              ['method', 'budget', 'frames', 'selected_mean',
               'non_ego_selected_mean', 'non_ego_packet_total',
               'byte_proxy_total', 'ap_03', 'ap_05', 'ap_07',
               'gt_total', 'pred_samples'],
              summary_rows)

    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'area_records': os.path.abspath(args.area_records),
        'area_quality': None if args.area_quality is None
        else os.path.abspath(args.area_quality),
        'confidence_field': args.confidence_field,
        'budgets': budgets,
        'methods': methods,
        'random_seed': args.random_seed,
        'feature_packet_bytes': args.feature_packet_bytes,
        'frames': selected_timestamps,
        'fusion_method': coperception_params['fusion_method'],
        'note': 'Perception-only subset ablation; no real feature-slice local fusion or RSU aggregation.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Offline Subset Ablation Smoke\n\n')
        stream.write('This run compares full sharing, random selective sharing, ')
        stream.write('confidence top-K, communication-aware top-K, and ')
        stream.write('area-aware greedy union subsets. ')
        stream.write('It is perception-only and does not model real leader local ')
        stream.write('fusion or RSU aggregation.\n\n')
        stream.write('- frames: `%d`\n' % len(selected_timestamps))
        stream.write('- budgets: `%s`\n' % ', '.join(map(str, budgets)))
        stream.write('- feature_packet_bytes: `%d`\n' %
                     args.feature_packet_bytes)
        stream.write('- frame_records: `%d`\n' % len(frame_rows))
        stream.write('- summary_rows: `%d`\n' % len(summary_rows))

    print('Wrote subset ablation to %s' % args.output_dir)
    for row in summary_rows:
        print('%s budget=%s AP30=%s AP50=%s AP70=%s' % (
            row['method'], row['budget'],
            row['ap_03'], row['ap_05'], row['ap_07']))


if __name__ == '__main__':
    main()
