# -*- coding: utf-8 -*-
"""
Build a capacity-gated scheduled LGCP upload-plan smoke input.

The offline NS3 replay sends one transfer batch per dumped frame, so this tool
does not model multi-slot frame-internal latency. It creates a conservative
single-slot smoke plan by keeping at most Z requests per timestamp and assigning
each kept request a distinct subchannel.
"""

import argparse
import csv
import os
from collections import OrderedDict, defaultdict

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a scheduled LGCP upload-plan smoke input.')
    parser.add_argument('--upload-plan', required=True,
                        help='LGCP upload plan, e.g. raw_slice_upload_plan.csv.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--subchannels', type=int, default=10,
                        help='Available subchannels per replay frame.')
    parser.add_argument('--leader-reserve', type=int, default=3,
                        help='Reserved subchannels for leader_to_rsu rows.')
    parser.add_argument('--priority-mode', default='hierarchy_bytes',
                        choices=['hierarchy_bytes', 'bytes'],
                        help='Request priority inside each frame.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_bytes(row):
    return int(float(row.get('bytes', 0) or 0))


def row_priority(row, priority_mode):
    upload_type = row.get('upload_type', '')
    if priority_mode == 'hierarchy_bytes':
        type_rank = 0 if upload_type == 'member_to_leader' else 1
        return (type_rank, -row_bytes(row), row.get('area_id', ''))
    return (-row_bytes(row), row.get('upload_type', ''), row.get('area_id', ''))


def group_by_timestamp(rows):
    frames = defaultdict(list)
    for row in rows:
        frames[row['timestamp']].append(row)
    return frames


def schedule_frame(rows, subchannels, leader_reserve, priority_mode):
    leader_capacity = min(max(leader_reserve, 0), subchannels)
    member_capacity = max(subchannels - leader_capacity, 0)
    member_rows = [
        row for row in rows
        if row.get('upload_type') == 'member_to_leader']
    leader_rows = [
        row for row in rows
        if row.get('upload_type') == 'leader_to_rsu']
    other_rows = [
        row for row in rows
        if row.get('upload_type') not in ('member_to_leader', 'leader_to_rsu')]

    selected_members = sorted(
        member_rows, key=lambda row: row_priority(row, priority_mode)
    )[:member_capacity]
    selected_leaders = sorted(
        leader_rows, key=lambda row: row_priority(row, priority_mode)
    )[:leader_capacity]

    remaining_slots = max(
        subchannels - len(selected_members) - len(selected_leaders), 0)
    selected_other = sorted(
        other_rows, key=lambda row: row_priority(row, priority_mode)
    )[:remaining_slots]

    selected = []
    gated = []
    selected_ids = set()
    for row in selected_members + selected_leaders + selected_other:
        selected_ids.add(id(row))
        selected.append(row)
    for row in rows:
        if id(row) not in selected_ids:
            gated.append(row)

    scheduled = []
    channel = 0
    for row in selected_members:
        scheduled.append((channel, row))
        channel += 1
    channel = member_capacity
    for row in selected_leaders:
        scheduled.append((channel, row))
        channel += 1
    for row in selected_other:
        scheduled.append((channel, row))
        channel += 1
    return scheduled, gated


def extend_fieldnames(rows):
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    for key in ['sc_start', 'sc_num', 'schedule_status', 'schedule_reason']:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


def build_scheduled_plan(rows, subchannels, leader_reserve, priority_mode):
    frames = group_by_timestamp(rows)
    scheduled_rows = []
    gated_rows = []
    frame_summary = []

    for timestamp in sorted(frames.keys()):
        frame_rows = frames[timestamp]
        scheduled, gated = schedule_frame(
            frame_rows, subchannels, leader_reserve, priority_mode)
        frame_scheduled_rows = []
        for channel, row in scheduled:
            item = OrderedDict(row)
            item['sc_start'] = channel
            item['sc_num'] = 1
            item['schedule_status'] = 'scheduled'
            item['schedule_reason'] = 'single_slot_capacity'
            scheduled_rows.append(item)
            frame_scheduled_rows.append(item)
        for row in gated:
            item = OrderedDict(row)
            item['sc_start'] = ''
            item['sc_num'] = ''
            item['schedule_status'] = 'capacity_gated'
            item['schedule_reason'] = 'single_slot_capacity'
            gated_rows.append(item)

        member_rows = [
            row for row in frame_scheduled_rows
            if row.get('upload_type') == 'member_to_leader']
        leader_rows = [
            row for row in frame_scheduled_rows
            if row.get('upload_type') == 'leader_to_rsu']
        frame_summary.append(OrderedDict({
            'timestamp': timestamp,
            'input_requests': len(frame_rows),
            'scheduled_requests': len(frame_scheduled_rows),
            'capacity_gated_requests': len(gated),
            'scheduled_member_requests': len(member_rows),
            'scheduled_leader_requests': len(leader_rows),
            'input_bytes': sum(row_bytes(row) for row in frame_rows),
            'scheduled_bytes': sum(row_bytes(row) for row in frame_scheduled_rows),
            'capacity_gated_bytes': sum(row_bytes(row) for row in gated),
        }))
    return scheduled_rows, gated_rows, frame_summary


def summarize(frame_summary):
    input_requests = sum(int(row['input_requests']) for row in frame_summary)
    scheduled_requests = sum(
        int(row['scheduled_requests']) for row in frame_summary)
    input_bytes = sum(int(row['input_bytes']) for row in frame_summary)
    scheduled_bytes = sum(int(row['scheduled_bytes']) for row in frame_summary)
    return [OrderedDict({
        'frames': len(frame_summary),
        'input_requests': input_requests,
        'scheduled_requests': scheduled_requests,
        'capacity_gated_requests': input_requests - scheduled_requests,
        'scheduled_request_ratio': '%.6f' % (
            scheduled_requests / float(max(input_requests, 1))),
        'input_bytes': input_bytes,
        'scheduled_bytes': scheduled_bytes,
        'capacity_gated_bytes': input_bytes - scheduled_bytes,
        'scheduled_byte_ratio': '%.6f' % (
            scheduled_bytes / float(max(input_bytes, 1))),
    })]


def write_notes(path, summary_rows, subchannels, leader_reserve):
    summary = summary_rows[0]
    with open(path, 'w') as stream:
        stream.write('# LGCP Scheduled Upload Plan Smoke\n\n')
        stream.write('This run builds a single-slot, capacity-gated NS3 smoke ')
        stream.write('plan from the raw-slice-aware LGCP upload plan. It keeps ')
        stream.write('at most `%d` requests per frame and assigns unique ' %
                     subchannels)
        stream.write('`sc_start/sc_num` values.\n\n')
        stream.write('- reserved leader-to-RSU subchannels: `%d`\n' % leader_reserve)
        stream.write('- scheduled requests: `%s / %s`\n' % (
            summary['scheduled_requests'], summary['input_requests']))
        stream.write('- scheduled bytes: `%s / %s`\n' % (
            summary['scheduled_bytes'], summary['input_bytes']))
        stream.write('\nBoundary: this is a scheduled replay smoke input, not ')
        stream.write('a full multi-slot LGCP scheduler or final performance row.\n')


def main():
    args = parse_args()
    if args.subchannels <= 0:
        raise ValueError('--subchannels must be positive')
    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_csv(args.upload_plan)
    scheduled_rows, gated_rows, frame_summary = build_scheduled_plan(
        rows, args.subchannels, args.leader_reserve, args.priority_mode)
    summary_rows = summarize(frame_summary)

    scheduled_fieldnames = extend_fieldnames(scheduled_rows or rows)
    gated_fieldnames = extend_fieldnames(gated_rows or rows)
    write_csv(os.path.join(args.output_dir, 'scheduled_upload_plan.csv'),
              scheduled_fieldnames, scheduled_rows)
    write_csv(os.path.join(args.output_dir, 'capacity_gated_upload_rows.csv'),
              gated_fieldnames, gated_rows)
    write_csv(os.path.join(args.output_dir, 'scheduled_frame_summary.csv'),
              list(frame_summary[0].keys()), frame_summary)
    write_csv(os.path.join(args.output_dir, 'scheduled_summary.csv'),
              list(summary_rows[0].keys()), summary_rows)

    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump({
            'upload_plan': os.path.abspath(args.upload_plan),
            'subchannels': args.subchannels,
            'leader_reserve': args.leader_reserve,
            'priority_mode': args.priority_mode,
            'note': 'Single-slot capacity-gated scheduled NS3 smoke input.',
        }, stream, sort_keys=False)
    write_notes(
        os.path.join(args.output_dir, 'notes.md'),
        summary_rows,
        args.subchannels,
        args.leader_reserve)

    row = summary_rows[0]
    print('scheduled_requests=%s/%s ratio=%s scheduled_bytes=%s/%s ratio=%s' % (
        row['scheduled_requests'],
        row['input_requests'],
        row['scheduled_request_ratio'],
        row['scheduled_bytes'],
        row['input_bytes'],
        row['scheduled_byte_ratio']))


if __name__ == '__main__':
    main()
