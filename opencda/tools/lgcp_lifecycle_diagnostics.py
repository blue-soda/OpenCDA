# -*- coding: utf-8 -*-
"""
Diagnose LGCP request lifecycle summaries by slot, stage, and endpoint.

This utility joins ``request_lifecycle.csv`` from lgcp_ns3_log_eval with the
replayed upload plan. It is meant for pinpointing whether live replay losses
come from scheduling, RLC/PSSCH, or application-level callbacks.
"""

import argparse
import csv
import os
from collections import OrderedDict, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Diagnose LGCP request lifecycle by slot/stage/endpoint.')
    parser.add_argument('--request-lifecycle', required=True,
                        help='request_lifecycle.csv from lgcp_ns3_log_eval.')
    parser.add_argument('--upload-plan', required=True,
                        help='Replayed upload_plan.csv with pkt_id/slot fields.')
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_int(row, key):
    value = row.get(key, '')
    if value == '':
        return 0
    return int(float(value))


def as_float(row, key):
    value = row.get(key, '')
    if value == '':
        return 0.0
    return float(value)


def build_plan_index(plan_rows):
    index = {}
    for row in plan_rows:
        key = (row.get('timestamp', ''), int(float(row.get('pkt_id', 0) or 0)))
        index[key] = row
    return index


def enrich_lifecycle(lifecycle_rows, plan_rows):
    plan_index = build_plan_index(plan_rows)
    enriched = []
    unmatched = []
    for row in lifecycle_rows:
        request_id = int(float(row.get('request_id', 0) or 0))
        key = (row.get('timestamp', ''), request_id)
        plan = plan_index.get(key)
        item = OrderedDict(row)
        if plan is None:
            unmatched.append(row)
            item.update({
                'slot_index': '',
                'stage': row.get('upload_type', ''),
                'scheduled_delay_ms': '',
                'sc_start': '',
                'sc_num': '',
            })
        else:
            item.update({
                'slot_index': plan.get('slot_index', ''),
                'stage': plan.get('stage', plan.get('upload_type', '')),
                'scheduled_delay_ms': plan.get('scheduled_delay_ms', ''),
                'sc_start': plan.get('sc_start', ''),
                'sc_num': plan.get('sc_num', ''),
            })
        enriched.append(item)
    return enriched, unmatched


def summarize_group(rows, key_fields):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, '') for field in key_fields)
        groups[key].append(row)

    summary = []
    for key in sorted(groups.keys()):
        group = groups[key]
        planned = len(group)
        rlc_tx = sum(1 for row in group if as_int(row, 'rlc_tx_count') > 0)
        rlc_rx = sum(1 for row in group if as_int(row, 'rlc_rx_count') > 0)
        pssch_ok = sum(1 for row in group if as_int(row, 'pssch_ok_count') > 0)
        pssch_fail = sum(
            1 for row in group if as_int(row, 'pssch_fail_count') > 0)
        cam = sum(1 for row in group if as_int(row, 'cam_received') > 0)
        bytes_total = sum(as_int(row, 'planned_bytes') for row in group)
        row = OrderedDict()
        for field, value in zip(key_fields, key):
            row[field] = value
        row.update(OrderedDict({
            'planned_requests': planned,
            'planned_bytes': bytes_total,
            'requests_with_rlc_tx': rlc_tx,
            'requests_with_rlc_rx': rlc_rx,
            'requests_with_pssch_ok': pssch_ok,
            'requests_with_pssch_fail': pssch_fail,
            'requests_with_cam_received': cam,
            'rlc_rx_ratio': '%.6f' % (rlc_rx / float(max(planned, 1))),
            'pssch_ok_ratio': '%.6f' % (pssch_ok / float(max(planned, 1))),
            'cam_received_ratio': '%.6f' % (cam / float(max(planned, 1))),
        }))
        summary.append(row)
    return summary


def terminal_summary(rows):
    return summarize_group(rows, ['upload_type', 'terminal_state'])


def write_notes(path, by_stage_slot, by_stage, unmatched_count):
    with open(path, 'w') as stream:
        stream.write('# LGCP Lifecycle Diagnostics\n\n')
        stream.write('This diagnostic joins request lifecycle records with ')
        stream.write('the replayed upload plan to expose slot/stage behavior.\n\n')
        stream.write('- unmatched lifecycle rows: `%d`\n' % unmatched_count)
        stream.write('\n## Stage Summary\n\n')
        stream.write('| Stage | Planned | RLC RX ratio | PSSCH OK ratio | CAM ratio |\n')
        stream.write('| --- | ---: | ---: | ---: | ---: |\n')
        for row in by_stage:
            stream.write('| %s | %s | %s | %s | %s |\n' % (
                row.get('stage', ''),
                row['planned_requests'],
                row['rlc_rx_ratio'],
                row['pssch_ok_ratio'],
                row['cam_received_ratio']))
        stream.write('\n## Slot Summary\n\n')
        stream.write('| Stage | Slot | Planned | RLC RX ratio | CAM ratio |\n')
        stream.write('| --- | ---: | ---: | ---: | ---: |\n')
        for row in by_stage_slot:
            stream.write('| %s | %s | %s | %s | %s |\n' % (
                row.get('stage', ''),
                row.get('slot_index', ''),
                row['planned_requests'],
                row['rlc_rx_ratio'],
                row['cam_received_ratio']))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    lifecycle_rows = read_csv(args.request_lifecycle)
    plan_rows = read_csv(args.upload_plan)
    enriched, unmatched = enrich_lifecycle(lifecycle_rows, plan_rows)

    by_stage = summarize_group(enriched, ['stage'])
    by_stage_slot = summarize_group(enriched, ['stage', 'slot_index'])
    by_type_terminal = terminal_summary(enriched)
    by_target = summarize_group(enriched, ['upload_type', 'target_node'])

    if enriched:
        write_csv(os.path.join(args.output_dir, 'lifecycle_enriched.csv'),
                  list(enriched[0].keys()), enriched)
    if by_stage:
        write_csv(os.path.join(args.output_dir, 'by_stage.csv'),
                  list(by_stage[0].keys()), by_stage)
    if by_stage_slot:
        write_csv(os.path.join(args.output_dir, 'by_stage_slot.csv'),
                  list(by_stage_slot[0].keys()), by_stage_slot)
    if by_type_terminal:
        write_csv(os.path.join(args.output_dir, 'by_type_terminal.csv'),
                  list(by_type_terminal[0].keys()), by_type_terminal)
    if by_target:
        write_csv(os.path.join(args.output_dir, 'by_upload_type_target.csv'),
                  list(by_target[0].keys()), by_target)
    write_notes(
        os.path.join(args.output_dir, 'notes.md'),
        by_stage_slot,
        by_stage,
        len(unmatched))

    print('lifecycle_rows=%s unmatched=%s stages=%s' % (
        len(enriched),
        len(unmatched),
        len(by_stage)))
    for row in by_stage:
        print('stage=%s planned=%s rlc_rx_ratio=%s pssch_ok_ratio=%s cam_ratio=%s' % (
            row.get('stage', ''),
            row['planned_requests'],
            row['rlc_rx_ratio'],
            row['pssch_ok_ratio'],
            row['cam_received_ratio']))


if __name__ == '__main__':
    main()
