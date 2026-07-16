# -*- coding: utf-8 -*-
"""
Small-scale optimality-gap evaluation for LGCP greedy group selection.

This script uses area confidence records exported by lgcp_area_confidence_eval.py.
It compares the paper-style Delta_g greedy group construction with exhaustive
subset search on small sampled instances.
"""

import argparse
import csv
import itertools
import os
from collections import defaultdict, OrderedDict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP greedy group-selection optimality gap.')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing area_records.csv.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for gap outputs.')
    parser.add_argument('--confidence-field', default='density_linear',
                        help='Per-agent confidence field to use.')
    parser.add_argument('--max-agents', type=int, default=6,
                        help='Maximum CAV/RSU agents per instance.')
    parser.add_argument('--max-areas', type=int, default=5,
                        help='Maximum areas per instance.')
    parser.add_argument('--max-group-size', type=int, default=4,
                        help='Maximum group size considered by exhaustive search.')
    parser.add_argument('--delta-g', default='0.05,0.075,0.1,0.125',
                        help='Comma-separated Delta_g values.')
    parser.add_argument('--lambda-size', type=float, default=0.02,
                        help='Size penalty for O2 objective.')
    parser.add_argument('--enable-o3', action='store_true',
                        help='Evaluate holistic latency-aware O3 objective.')
    parser.add_argument('--o3-t-delta', type=float, default=1.0,
                        help='Fixed coordination latency term for O3 proxy.')
    parser.add_argument('--o3-packet-weight', type=float, default=0.05,
                        help='Packet/link count weight for O3 proxy.')
    parser.add_argument('--o3-load-weight', type=float, default=0.1,
                        help='Leader max-load weight for O3 proxy.')
    parser.add_argument('--max-instances', type=int, default=0,
                        help='Limit timestamps. 0 means all timestamps.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def noisy_or(values):
    if not values:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - values))


def group_confidence(area_conf, group):
    return noisy_or([area_conf.get(agent_id, 0.0) for agent_id in group])


def greedy_group(area_conf, agents, delta_g):
    selected = []
    current = 0.0
    for agent_id in sorted(agents, key=lambda vid: area_conf.get(vid, 0.0),
                           reverse=True):
        new_conf = group_confidence(area_conf, selected + [agent_id])
        if new_conf - current >= delta_g:
            selected.append(agent_id)
            current = new_conf
    if not selected and agents:
        selected = [max(agents, key=lambda vid: area_conf.get(vid, 0.0))]
        current = group_confidence(area_conf, selected)
    return selected, current


def best_subset(area_conf, agents, objective, lambda_size, max_group_size):
    best_group = []
    best_value = float('-inf')
    best_conf = 0.0
    upper = min(max_group_size, len(agents))
    for size in range(1, upper + 1):
        for subset in itertools.combinations(agents, size):
            conf = group_confidence(area_conf, subset)
            value = objective_value(conf, size, len(agents), objective,
                                    lambda_size)
            if value > best_value:
                best_value = value
                best_group = list(subset)
                best_conf = conf
    return best_group, best_conf, best_value


def objective_value(confidence, group_size, agent_count, objective, lambda_size):
    if objective == 'O1_confidence_only':
        return confidence
    if objective == 'O2_confidence_minus_size':
        return confidence - lambda_size * (float(group_size) / agent_count)
    raise ValueError('Unknown objective: %s' % objective)


def load_area_quality(input_dir):
    path = os.path.join(input_dir, 'area_quality.csv')
    if not os.path.exists(path):
        return {}
    rows = read_csv(path)
    quality = {}
    for row in rows:
        quality[(row['timestamp'], row['area_id'])] = row
    return quality


def build_instances(area_records, quality_rows, confidence_field,
                    max_agents, max_areas, max_instances):
    by_timestamp_area = defaultdict(dict)
    agent_totals = defaultdict(lambda: defaultdict(float))
    area_scores = defaultdict(lambda: defaultdict(float))

    for row in area_records:
        timestamp = row['timestamp']
        area_id = row['area_id']
        agent_id = row['agent_id']
        confidence = float(row[confidence_field])
        by_timestamp_area[(timestamp, area_id)][agent_id] = confidence
        agent_totals[timestamp][agent_id] += confidence
        quality = quality_rows.get((timestamp, area_id))
        if quality is not None and quality.get('gt_count', '') != '':
            area_scores[timestamp][area_id] = max(
                area_scores[timestamp][area_id],
                float(quality['gt_count']))
        else:
            area_scores[timestamp][area_id] += confidence

    timestamps = sorted({row['timestamp'] for row in area_records})
    if max_instances > 0:
        timestamps = timestamps[:max_instances]

    instances = []
    for timestamp in timestamps:
        agents = [
            agent for agent, _ in sorted(
                agent_totals[timestamp].items(),
                key=lambda item: item[1],
                reverse=True)[:max_agents]
        ]
        candidate_areas = [
            area for area, _ in sorted(
                area_scores[timestamp].items(),
                key=lambda item: item[1],
                reverse=True)[:max_areas]
        ]
        if not agents or not candidate_areas:
            continue

        area_conf = OrderedDict()
        for area_id in candidate_areas:
            values = by_timestamp_area.get((timestamp, area_id), {})
            area_conf[area_id] = {
                agent_id: values.get(agent_id, 0.0)
                for agent_id in agents
            }
        instances.append({
            'timestamp': timestamp,
            'agents': agents,
            'areas': candidate_areas,
            'area_conf': area_conf,
        })
    return instances


def evaluate_instance(instance, delta_g, objective, lambda_size, max_group_size):
    greedy_conf_sum = 0.0
    greedy_value_sum = 0.0
    greedy_size_sum = 0
    optimal_conf_sum = 0.0
    optimal_value_sum = 0.0
    optimal_size_sum = 0

    for area_id in instance['areas']:
        area_conf = instance['area_conf'][area_id]
        greedy, greedy_conf = greedy_group(area_conf, instance['agents'],
                                           delta_g)
        greedy_value = objective_value(
            greedy_conf, len(greedy), len(instance['agents']),
            objective, lambda_size)
        optimal, optimal_conf, optimal_value = best_subset(
            area_conf, instance['agents'], objective, lambda_size,
            max_group_size)
        greedy_conf_sum += greedy_conf
        greedy_value_sum += greedy_value
        greedy_size_sum += len(greedy)
        optimal_conf_sum += optimal_conf
        optimal_value_sum += optimal_value
        optimal_size_sum += len(optimal)

    area_count = len(instance['areas'])
    greedy_value_avg = greedy_value_sum / area_count
    optimal_value_avg = optimal_value_sum / area_count
    absolute_gap = optimal_value_avg - greedy_value_avg
    relative_gap = (
        absolute_gap / max(abs(optimal_value_avg), 1e-9)
        if optimal_value_avg != 0 else 0.0)

    return OrderedDict({
        'timestamp': instance['timestamp'],
        'agent_count': len(instance['agents']),
        'area_count': area_count,
        'delta_g': '%.6f' % delta_g,
        'objective': objective,
        'greedy_value': '%.6f' % greedy_value_avg,
        'optimal_value': '%.6f' % optimal_value_avg,
        'absolute_gap': '%.6f' % absolute_gap,
        'relative_gap': '%.6f' % relative_gap,
        'greedy_mean_conf': '%.6f' % (greedy_conf_sum / area_count),
        'optimal_mean_conf': '%.6f' % (optimal_conf_sum / area_count),
        'greedy_total_size': greedy_size_sum,
        'optimal_total_size': optimal_size_sum,
        'greedy_mean_size': '%.6f' % (float(greedy_size_sum) / area_count),
        'optimal_mean_size': '%.6f' % (float(optimal_size_sum) / area_count),
    })


def construct_greedy_groups(instance, delta_g):
    groups = []
    for area_id in instance['areas']:
        area_conf = instance['area_conf'][area_id]
        group, conf = greedy_group(area_conf, instance['agents'], delta_g)
        if group:
            groups.append({
                'area_id': area_id,
                'members': group,
                'confidence': conf,
                'load': len(group),
            })
    return groups


def greedy_leader_assignment(groups, agents):
    loads = {agent: 0 for agent in agents}
    assignments = {}
    for group in sorted(groups, key=lambda item: item['load'], reverse=True):
        leader = min(group['members'], key=lambda agent: (loads[agent], agent))
        assignments[group['area_id']] = leader
        loads[leader] += group['load']
    return assignments, loads


def optimal_leader_assignment(groups, agents):
    if not groups:
        return {}, {agent: 0 for agent in agents}

    best_assignments = None
    best_loads = None
    best_max_load = float('inf')
    choices = [group['members'] for group in groups]
    for leaders in itertools.product(*choices):
        loads = {agent: 0 for agent in agents}
        assignments = {}
        for group, leader in zip(groups, leaders):
            assignments[group['area_id']] = leader
            loads[leader] += group['load']
        max_load = max(loads.values()) if loads else 0
        if max_load < best_max_load:
            best_max_load = max_load
            best_assignments = assignments
            best_loads = loads
    return best_assignments, best_loads


def evaluate_leader_assignment(instance, delta_g):
    groups = construct_greedy_groups(instance, delta_g)
    greedy_assignments, greedy_loads = greedy_leader_assignment(
        groups, instance['agents'])
    optimal_assignments, optimal_loads = optimal_leader_assignment(
        groups, instance['agents'])

    greedy_max = max(greedy_loads.values()) if greedy_loads else 0
    optimal_max = max(optimal_loads.values()) if optimal_loads else 0
    absolute_gap = greedy_max - optimal_max
    relative_gap = (
        float(absolute_gap) / max(float(optimal_max), 1e-9)
        if optimal_max != 0 else 0.0)
    total_load = sum(group['load'] for group in groups)

    return OrderedDict({
        'timestamp': instance['timestamp'],
        'agent_count': len(instance['agents']),
        'area_count': len(instance['areas']),
        'group_count': len(groups),
        'delta_g': '%.6f' % delta_g,
        'total_group_load': total_load,
        'greedy_max_load': greedy_max,
        'optimal_max_load': optimal_max,
        'absolute_gap': '%.6f' % absolute_gap,
        'relative_gap': '%.6f' % relative_gap,
        'greedy_loads': ';'.join('%s:%s' % item for item in
                                sorted(greedy_loads.items())),
        'optimal_loads': ';'.join('%s:%s' % item for item in
                                 sorted(optimal_loads.items())),
        'greedy_assignments': ';'.join('%s:%s' % item for item in
                                      sorted(greedy_assignments.items())),
        'optimal_assignments': ';'.join('%s:%s' % item for item in
                                       sorted(optimal_assignments.items())),
    })


def subset_candidates(area_conf, agents, max_group_size):
    candidates = []
    upper = min(max_group_size, len(agents))
    for size in range(1, upper + 1):
        for subset in itertools.combinations(agents, size):
            group = list(subset)
            candidates.append({
                'members': group,
                'confidence': group_confidence(area_conf, group),
                'load': len(group),
            })
    return candidates


def min_max_leader_load(groups, agents):
    _, loads = optimal_leader_assignment(groups, agents)
    return max(loads.values()) if loads else 0


def o3_value(groups, agents, t_delta, packet_weight, load_weight):
    if not groups:
        return 0.0, 0.0, 0, 0
    mean_conf = float(np.mean([group['confidence'] for group in groups]))
    packet_count = sum(len(group['members']) for group in groups)
    max_load = min_max_leader_load(groups, agents)
    latency = t_delta + packet_weight * packet_count + load_weight * max_load
    return mean_conf / max(latency, 1e-9), mean_conf, packet_count, max_load


def evaluate_o3_instance(instance, delta_g, max_group_size, t_delta,
                         packet_weight, load_weight):
    greedy_groups = construct_greedy_groups(instance, delta_g)
    greedy_value, greedy_conf, greedy_packets, greedy_max_load = o3_value(
        greedy_groups, instance['agents'], t_delta, packet_weight, load_weight)

    area_candidates = []
    for area_id in instance['areas']:
        candidates = subset_candidates(
            instance['area_conf'][area_id], instance['agents'], max_group_size)
        for candidate in candidates:
            candidate['area_id'] = area_id
        area_candidates.append(candidates)

    best_value = float('-inf')
    best_conf = 0.0
    best_packets = 0
    best_max_load = 0
    combinations = 0
    for combination in itertools.product(*area_candidates):
        combinations += 1
        groups = [dict(group) for group in combination]
        value, conf, packets, max_load = o3_value(
            groups, instance['agents'], t_delta, packet_weight, load_weight)
        if value > best_value:
            best_value = value
            best_conf = conf
            best_packets = packets
            best_max_load = max_load

    absolute_gap = best_value - greedy_value
    relative_gap = (
        absolute_gap / max(abs(best_value), 1e-9)
        if best_value != 0 else 0.0)

    return OrderedDict({
        'timestamp': instance['timestamp'],
        'agent_count': len(instance['agents']),
        'area_count': len(instance['areas']),
        'delta_g': '%.6f' % delta_g,
        'objective': 'O3_confidence_latency_ratio',
        'greedy_value': '%.6f' % greedy_value,
        'optimal_value': '%.6f' % best_value,
        'absolute_gap': '%.6f' % absolute_gap,
        'relative_gap': '%.6f' % relative_gap,
        'greedy_mean_conf': '%.6f' % greedy_conf,
        'optimal_mean_conf': '%.6f' % best_conf,
        'greedy_packet_count': greedy_packets,
        'optimal_packet_count': best_packets,
        'greedy_max_load': greedy_max_load,
        'optimal_max_load': best_max_load,
        'candidate_combinations': combinations,
    })


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['objective'], row['delta_g'])].append(row)

    rows = []
    for (objective, delta_g), group in sorted(grouped.items()):
        gaps = np.asarray([float(row['relative_gap']) for row in group])
        abs_gaps = np.asarray([float(row['absolute_gap']) for row in group])
        rows.append(OrderedDict({
            'objective': objective,
            'delta_g': delta_g,
            'instances': len(group),
            'mean_relative_gap': '%.6f' % float(np.mean(gaps)),
            'median_relative_gap': '%.6f' % float(np.median(gaps)),
            'p90_relative_gap': '%.6f' % float(np.percentile(gaps, 90)),
            'max_relative_gap': '%.6f' % float(np.max(gaps)),
            'mean_absolute_gap': '%.6f' % float(np.mean(abs_gaps)),
        }))
    return rows


def summarize_leader(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row['delta_g']].append(row)

    rows = []
    for delta_g, group in sorted(grouped.items()):
        gaps = np.asarray([float(row['relative_gap']) for row in group])
        abs_gaps = np.asarray([float(row['absolute_gap']) for row in group])
        rows.append(OrderedDict({
            'delta_g': delta_g,
            'instances': len(group),
            'mean_relative_gap': '%.6f' % float(np.mean(gaps)),
            'median_relative_gap': '%.6f' % float(np.median(gaps)),
            'p90_relative_gap': '%.6f' % float(np.percentile(gaps, 90)),
            'max_relative_gap': '%.6f' % float(np.max(gaps)),
            'mean_absolute_gap': '%.6f' % float(np.mean(abs_gaps)),
        }))
    return rows


def summarize_o3(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row['delta_g']].append(row)

    rows = []
    for delta_g, group in sorted(grouped.items()):
        gaps = np.asarray([float(row['relative_gap']) for row in group])
        abs_gaps = np.asarray([float(row['absolute_gap']) for row in group])
        greedy_packets = np.asarray(
            [float(row['greedy_packet_count']) for row in group])
        optimal_packets = np.asarray(
            [float(row['optimal_packet_count']) for row in group])
        rows.append(OrderedDict({
            'objective': 'O3_confidence_latency_ratio',
            'delta_g': delta_g,
            'instances': len(group),
            'mean_relative_gap': '%.6f' % float(np.mean(gaps)),
            'median_relative_gap': '%.6f' % float(np.median(gaps)),
            'p90_relative_gap': '%.6f' % float(np.percentile(gaps, 90)),
            'max_relative_gap': '%.6f' % float(np.max(gaps)),
            'mean_absolute_gap': '%.6f' % float(np.mean(abs_gaps)),
            'mean_greedy_packet_count': '%.6f' % float(np.mean(greedy_packets)),
            'mean_optimal_packet_count': '%.6f' % float(np.mean(optimal_packets)),
        }))
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    area_records = read_csv(os.path.join(args.input_dir, 'area_records.csv'))
    quality_rows = load_area_quality(args.input_dir)
    deltas = [float(value.strip()) for value in args.delta_g.split(',')
              if value.strip()]
    instances = build_instances(
        area_records, quality_rows, args.confidence_field,
        args.max_agents, args.max_areas, args.max_instances)

    rows = []
    leader_rows = []
    o3_rows = []
    for instance in instances:
        for delta_g in deltas:
            leader_rows.append(evaluate_leader_assignment(instance, delta_g))
            if args.enable_o3:
                o3_rows.append(evaluate_o3_instance(
                    instance, delta_g, args.max_group_size, args.o3_t_delta,
                    args.o3_packet_weight, args.o3_load_weight))
        for objective in ('O1_confidence_only', 'O2_confidence_minus_size'):
            for delta_g in deltas:
                rows.append(evaluate_instance(
                    instance, delta_g, objective, args.lambda_size,
                    args.max_group_size))

    instance_fields = [
        'timestamp', 'agent_count', 'area_count', 'delta_g', 'objective',
        'greedy_value', 'optimal_value', 'absolute_gap', 'relative_gap',
        'greedy_mean_conf', 'optimal_mean_conf', 'greedy_total_size',
        'optimal_total_size', 'greedy_mean_size', 'optimal_mean_size',
    ]
    write_csv(os.path.join(args.output_dir, 'instance_records.csv'),
              instance_fields, rows)

    summary_rows = summarize(rows)
    summary_fields = [
        'objective', 'delta_g', 'instances', 'mean_relative_gap',
        'median_relative_gap', 'p90_relative_gap', 'max_relative_gap',
        'mean_absolute_gap',
    ]
    write_csv(os.path.join(args.output_dir, 'gap_summary.csv'),
              summary_fields, summary_rows)

    leader_fields = [
        'timestamp', 'agent_count', 'area_count', 'group_count', 'delta_g',
        'total_group_load', 'greedy_max_load', 'optimal_max_load',
        'absolute_gap', 'relative_gap', 'greedy_loads', 'optimal_loads',
        'greedy_assignments', 'optimal_assignments',
    ]
    write_csv(os.path.join(args.output_dir, 'leader_records.csv'),
              leader_fields, leader_rows)

    leader_summary_rows = summarize_leader(leader_rows)
    leader_summary_fields = [
        'delta_g', 'instances', 'mean_relative_gap',
        'median_relative_gap', 'p90_relative_gap', 'max_relative_gap',
        'mean_absolute_gap',
    ]
    write_csv(os.path.join(args.output_dir, 'leader_gap_summary.csv'),
              leader_summary_fields, leader_summary_rows)

    o3_summary_rows = []
    if args.enable_o3:
        o3_fields = [
            'timestamp', 'agent_count', 'area_count', 'delta_g', 'objective',
            'greedy_value', 'optimal_value', 'absolute_gap', 'relative_gap',
            'greedy_mean_conf', 'optimal_mean_conf', 'greedy_packet_count',
            'optimal_packet_count', 'greedy_max_load', 'optimal_max_load',
            'candidate_combinations',
        ]
        write_csv(os.path.join(args.output_dir, 'o3_instance_records.csv'),
                  o3_fields, o3_rows)

        o3_summary_rows = summarize_o3(o3_rows)
        o3_summary_fields = [
            'objective', 'delta_g', 'instances', 'mean_relative_gap',
            'median_relative_gap', 'p90_relative_gap', 'max_relative_gap',
            'mean_absolute_gap', 'mean_greedy_packet_count',
            'mean_optimal_packet_count',
        ]
        write_csv(os.path.join(args.output_dir, 'o3_gap_summary.csv'),
                  o3_summary_fields, o3_summary_rows)

    config = {
        'input_dir': os.path.abspath(args.input_dir),
        'confidence_field': args.confidence_field,
        'max_agents': args.max_agents,
        'max_areas': args.max_areas,
        'max_group_size': args.max_group_size,
        'delta_g': deltas,
        'lambda_size': args.lambda_size,
        'enable_o3': args.enable_o3,
        'o3_t_delta': args.o3_t_delta,
        'o3_packet_weight': args.o3_packet_weight,
        'o3_load_weight': args.o3_load_weight,
        'instances': len(instances),
        'note': 'Group-member exhaustive gap, leader assignment exhaustive load gap, and optional holistic O3 latency-aware exhaustive gap.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Greedy Gap Smoke\n\n')
        stream.write('This run compares Delta_g greedy group construction ')
        stream.write('against exhaustive subset search on small sampled ')
        stream.write('area-frame instances. It also compares greedy leader ')
        stream.write('assignment against exhaustive min-max load assignment ')
        stream.write('for the greedy-selected groups.\n\n')
        stream.write('- instances: `%d`\n' % len(instances))
        stream.write('- instance_records: `%d`\n' % len(rows))
        stream.write('- summary_rows: `%d`\n' % len(summary_rows))
        stream.write('- leader_records: `%d`\n' % len(leader_rows))
        stream.write('- leader_summary_rows: `%d`\n' % len(leader_summary_rows))
        stream.write('- o3_records: `%d`\n' % len(o3_rows))
        stream.write('- o3_summary_rows: `%d`\n' % len(o3_summary_rows))

    print('Wrote %d gap records to %s' % (len(rows), args.output_dir))
    for row in summary_rows:
        print('%s delta=%s mean_gap=%s p90=%s max=%s' % (
            row['objective'], row['delta_g'], row['mean_relative_gap'],
            row['p90_relative_gap'], row['max_relative_gap']))
    for row in leader_summary_rows:
        print('leader delta=%s mean_gap=%s p90=%s max=%s' % (
            row['delta_g'], row['mean_relative_gap'],
            row['p90_relative_gap'], row['max_relative_gap']))
    for row in o3_summary_rows:
        print('O3 delta=%s mean_gap=%s p90=%s max=%s packets=%s/%s' % (
            row['delta_g'], row['mean_relative_gap'],
            row['p90_relative_gap'], row['max_relative_gap'],
            row['mean_greedy_packet_count'],
            row['mean_optimal_packet_count']))


if __name__ == '__main__':
    main()
