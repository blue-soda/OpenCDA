# -*- coding: utf-8 -*-
"""
Build scheduled LGCP upload-plan inputs and scheduling proxies.

``single_slot`` creates a conservative NS3 smoke plan by keeping at most Z
requests per timestamp and assigning each kept request a distinct subchannel.
``multi_slot`` schedules every request into member-to-leader and leader-to-RSU
stages, producing a latency proxy for the full LGCP upload plan.
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
    parser.add_argument('--schedule-mode', default='single_slot',
                        choices=['single_slot', 'multi_slot'],
                        help='single_slot gates requests for NS3 smoke; '
                             'multi_slot schedules all requests across '
                             'sequential stage slots.')
    parser.add_argument('--slot-duration-ms', type=float, default=10.0,
                        help='Latency proxy duration per scheduled slot.')
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
    for key in [
            'sc_start',
            'sc_num',
            'slot_index',
            'stage',
            'schedule_status',
            'schedule_reason',
            'scheduled_delay_ms']:
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
            item['slot_index'] = 0
            item['stage'] = item.get('upload_type', '')
            item['schedule_status'] = 'scheduled'
            item['schedule_reason'] = 'single_slot_capacity'
            item['scheduled_delay_ms'] = 0
            scheduled_rows.append(item)
            frame_scheduled_rows.append(item)
        for row in gated:
            item = OrderedDict(row)
            item['sc_start'] = ''
            item['sc_num'] = ''
            item['slot_index'] = ''
            item['stage'] = item.get('upload_type', '')
            item['schedule_status'] = 'capacity_gated'
            item['schedule_reason'] = 'single_slot_capacity'
            item['scheduled_delay_ms'] = ''
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


def schedule_rows_to_slots(rows, subchannels, start_slot, stage,
                           priority_mode, slot_duration_ms):
    scheduled = []
    sorted_rows = sorted(rows, key=lambda row: row_priority(row, priority_mode))
    for index, row in enumerate(sorted_rows):
        slot_offset = index // subchannels
        channel = index % subchannels
        slot_index = start_slot + slot_offset
        item = OrderedDict(row)
        item['sc_start'] = channel
        item['sc_num'] = 1
        item['slot_index'] = slot_index
        item['stage'] = stage
        item['schedule_status'] = 'scheduled'
        item['schedule_reason'] = 'multi_slot_stage'
        item['scheduled_delay_ms'] = '%.6f' % (
            (slot_index + 1) * slot_duration_ms)
        scheduled.append(item)
    slot_count = ((len(sorted_rows) + subchannels - 1) // subchannels
                  if sorted_rows else 0)
    return scheduled, slot_count


def build_multi_slot_plan(rows, subchannels, priority_mode, slot_duration_ms):
    frames = group_by_timestamp(rows)
    scheduled_rows = []
    gated_rows = []
    frame_summary = []

    for timestamp in sorted(frames.keys()):
        frame_rows = frames[timestamp]
        member_rows = [
            row for row in frame_rows
            if row.get('upload_type') == 'member_to_leader']
        leader_rows = [
            row for row in frame_rows
            if row.get('upload_type') == 'leader_to_rsu']
        other_rows = [
            row for row in frame_rows
            if row.get('upload_type') not in ('member_to_leader',
                                              'leader_to_rsu')]

        scheduled_members, member_slots = schedule_rows_to_slots(
            member_rows,
            subchannels,
            0,
            'member_to_leader',
            priority_mode,
            slot_duration_ms)
        scheduled_leaders, leader_slots = schedule_rows_to_slots(
            leader_rows,
            subchannels,
            member_slots,
            'leader_to_rsu',
            priority_mode,
            slot_duration_ms)
        scheduled_other, other_slots = schedule_rows_to_slots(
            other_rows,
            subchannels,
            member_slots + leader_slots,
            'other',
            priority_mode,
            slot_duration_ms)
        frame_scheduled_rows = (
            scheduled_members + scheduled_leaders + scheduled_other)
        scheduled_rows.extend(frame_scheduled_rows)

        frame_summary.append(OrderedDict({
            'timestamp': timestamp,
            'input_requests': len(frame_rows),
            'scheduled_requests': len(frame_scheduled_rows),
            'capacity_gated_requests': 0,
            'scheduled_member_requests': len(scheduled_members),
            'scheduled_leader_requests': len(scheduled_leaders),
            'member_slots': member_slots,
            'leader_slots': leader_slots,
            'other_slots': other_slots,
            'total_slots': member_slots + leader_slots + other_slots,
            'input_bytes': sum(row_bytes(row) for row in frame_rows),
            'scheduled_bytes': sum(row_bytes(row)
                                   for row in frame_scheduled_rows),
            'capacity_gated_bytes': 0,
            'frame_latency_ms': '%.6f' % (
                (member_slots + leader_slots + other_slots) *
                slot_duration_ms),
        }))
    return scheduled_rows, gated_rows, frame_summary


def summarize(frame_summary, slot_duration_ms=None):
    input_requests = sum(int(row['input_requests']) for row in frame_summary)
    scheduled_requests = sum(
        int(row['scheduled_requests']) for row in frame_summary)
    input_bytes = sum(int(row['input_bytes']) for row in frame_summary)
    scheduled_bytes = sum(int(row['scheduled_bytes']) for row in frame_summary)
    summary = OrderedDict({
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
    })
    if frame_summary and 'total_slots' in frame_summary[0]:
        total_slots = [int(row['total_slots']) for row in frame_summary]
        latencies = [float(row['frame_latency_ms']) for row in frame_summary]
        summary.update(OrderedDict({
            'slot_duration_ms': '%.6f' % float(slot_duration_ms),
            'total_slots_mean': '%.6f' % (
                sum(total_slots) / float(len(total_slots))),
            'total_slots_max': max(total_slots),
            'frame_latency_ms_mean': '%.6f' % (
                sum(latencies) / float(len(latencies))),
            'frame_latency_ms_max': '%.6f' % max(latencies),
        }))
    return [summary]


def write_notes(path, summary_rows, subchannels, leader_reserve, schedule_mode):
    summary = summary_rows[0]
    with open(path, 'w') as stream:
        stream.write('# LGCP Scheduled Upload Plan Smoke\n\n')
        if schedule_mode == 'single_slot':
            stream.write('This run builds a single-slot, capacity-gated NS3 ')
            stream.write('smoke plan from the raw-slice-aware LGCP upload ')
            stream.write('plan. It keeps at most `%d` requests per frame and '
                         % subchannels)
            stream.write('assigns unique `sc_start/sc_num` values.\n\n')
            stream.write('- reserved leader-to-RSU subchannels: `%d`\n' %
                         leader_reserve)
        else:
            stream.write('This run schedules every request into sequential ')
            stream.write('member-to-leader and leader-to-RSU slots. It is a ')
            stream.write('latency proxy and full-plan scheduler input, not an ')
            stream.write('NS3 live replay by itself.\n\n')
            stream.write('- subchannels per slot: `%d`\n' % subchannels)
        stream.write('- scheduled requests: `%s / %s`\n' % (
            summary['scheduled_requests'], summary['input_requests']))
        stream.write('- scheduled bytes: `%s / %s`\n' % (
            summary['scheduled_bytes'], summary['input_bytes']))
        if 'frame_latency_ms_mean' in summary:
            stream.write('- mean frame scheduling latency: `%s ms`\n' %
                         summary['frame_latency_ms_mean'])
            stream.write('- max frame scheduling latency: `%s ms`\n' %
                         summary['frame_latency_ms_max'])
        stream.write('\nBoundary: this is a scheduling proxy, not final ')
        stream.write('perception performance.\n')


def main():
    args = parse_args()
    if args.subchannels <= 0:
        raise ValueError('--subchannels must be positive')
    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_csv(args.upload_plan)
    if args.schedule_mode == 'single_slot':
        scheduled_rows, gated_rows, frame_summary = build_scheduled_plan(
            rows, args.subchannels, args.leader_reserve, args.priority_mode)
    else:
        scheduled_rows, gated_rows, frame_summary = build_multi_slot_plan(
            rows, args.subchannels, args.priority_mode, args.slot_duration_ms)
    summary_rows = summarize(frame_summary, args.slot_duration_ms)

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
            'schedule_mode': args.schedule_mode,
            'slot_duration_ms': args.slot_duration_ms,
            'note': 'Scheduled LGCP upload-plan input/proxy.',
        }, stream, sort_keys=False)
    write_notes(
        os.path.join(args.output_dir, 'notes.md'),
        summary_rows,
        args.subchannels,
        args.leader_reserve,
        args.schedule_mode)

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
