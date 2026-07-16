# -*- coding: utf-8 -*-
"""
Estimate LGCP subchannel-count sensitivity from an upload plan.

This is a lightweight scheduling-capacity proxy. It does not run ns-3; instead,
it estimates how many sequential sidelink slots are needed when at most Z
requests can be scheduled in one slot. Member-to-leader and leader-to-RSU
uploads are modeled as separate stages because leader results depend on local
fusion completion.
"""

import argparse
import csv
import math
import os
from collections import OrderedDict, defaultdict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP subchannel-count scheduling proxy.')
    parser.add_argument('--upload-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--z-values', default='5,10,15,20',
                        help='Comma separated subchannel counts.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, q))


def summarize(values):
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), percentile(values, 95), float(np.max(values))


def build_frame_counts(upload_rows):
    frames = defaultdict(lambda: defaultdict(lambda: {'packets': 0, 'bytes': 0}))
    for row in upload_rows:
        timestamp = row['timestamp']
        upload_type = row.get('upload_type', 'unknown')
        frames[timestamp][upload_type]['packets'] += 1
        frames[timestamp][upload_type]['bytes'] += int(float(row.get('bytes', 0) or 0))
    return frames


def evaluate(frames, z_values):
    frame_rows = []
    summary_rows = []
    upload_types = ['member_to_leader', 'leader_to_rsu']
    timestamps = sorted(frames.keys())
    for z in z_values:
        total_slots = []
        max_stage_loads = []
        total_packets = []
        total_bytes = []
        for timestamp in timestamps:
            stage_slots = {}
            stage_packets = {}
            stage_bytes = {}
            stage_loads = {}
            for upload_type in upload_types:
                packets = frames[timestamp][upload_type]['packets']
                bytes_count = frames[timestamp][upload_type]['bytes']
                slots = int(math.ceil(float(packets) / z)) if packets else 0
                stage_packets[upload_type] = packets
                stage_bytes[upload_type] = bytes_count
                stage_slots[upload_type] = slots
                stage_loads[upload_type] = float(packets) / z if z else 0.0
            frame_total_slots = sum(stage_slots.values())
            frame_total_packets = sum(stage_packets.values())
            frame_total_bytes = sum(stage_bytes.values())
            max_stage_load = max(stage_loads.values()) if stage_loads else 0.0
            total_slots.append(frame_total_slots)
            max_stage_loads.append(max_stage_load)
            total_packets.append(frame_total_packets)
            total_bytes.append(frame_total_bytes)
            frame_rows.append(OrderedDict({
                'z': z,
                'timestamp': timestamp,
                'member_to_leader_packets': stage_packets['member_to_leader'],
                'member_to_leader_slots': stage_slots['member_to_leader'],
                'leader_to_rsu_packets': stage_packets['leader_to_rsu'],
                'leader_to_rsu_slots': stage_slots['leader_to_rsu'],
                'total_packets': frame_total_packets,
                'total_bytes': frame_total_bytes,
                'total_slots': frame_total_slots,
                'max_stage_packets_per_subchannel': '%.6f' % max_stage_load,
            }))
        slots_mean, slots_p95, slots_max = summarize(total_slots)
        load_mean, load_p95, load_max = summarize(max_stage_loads)
        packets_mean, _, packets_max = summarize(total_packets)
        bytes_mean, _, bytes_max = summarize(total_bytes)
        summary_rows.append(OrderedDict({
            'z': z,
            'frames': len(timestamps),
            'packets_per_frame_mean': '%.6f' % packets_mean,
            'packets_per_frame_max': '%.6f' % packets_max,
            'bytes_per_frame_mean': '%.6f' % bytes_mean,
            'bytes_per_frame_max': '%.6f' % bytes_max,
            'total_slots_mean': '%.6f' % slots_mean,
            'total_slots_p95': '%.6f' % slots_p95,
            'total_slots_max': '%.6f' % slots_max,
            'max_stage_packets_per_subchannel_mean': '%.6f' % load_mean,
            'max_stage_packets_per_subchannel_p95': '%.6f' % load_p95,
            'max_stage_packets_per_subchannel_max': '%.6f' % load_max,
        }))
    return frame_rows, summary_rows


def write_notes(path, summary_rows):
    with open(path, 'w') as stream:
        stream.write('# LGCP Subchannel Count Sensitivity\n\n')
        stream.write('This run estimates scheduling pressure from the LGCP ')
        stream.write('upload plan. It is a slot proxy, not an ns-3 PHY run.\n\n')
        stream.write('| Z | Mean slots / frame | Max slots / frame | Mean max stage packets / subchannel |\n')
        stream.write('| ---: | ---: | ---: | ---: |\n')
        for row in summary_rows:
            stream.write('| %s | %s | %s | %s |\n' % (
                row['z'],
                row['total_slots_mean'],
                row['total_slots_max'],
                row['max_stage_packets_per_subchannel_mean']))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    z_values = [int(item.strip()) for item in args.z_values.split(',')
                if item.strip()]
    upload_rows = read_csv(args.upload_plan)
    frames = build_frame_counts(upload_rows)
    frame_rows, summary_rows = evaluate(frames, z_values)

    write_csv(os.path.join(args.output_dir, 'subchannel_frame_proxy.csv'),
              list(frame_rows[0].keys()), frame_rows)
    write_csv(os.path.join(args.output_dir, 'subchannel_summary.csv'),
              list(summary_rows[0].keys()), summary_rows)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump({
            'upload_plan': os.path.abspath(args.upload_plan),
            'z_values': z_values,
            'note': 'Scheduling-capacity proxy; not an ns-3 PHY simulation.',
        }, stream, sort_keys=False)
    write_notes(os.path.join(args.output_dir, 'notes.md'), summary_rows)

    for row in summary_rows:
        print('Z=%s slots_mean=%s slots_max=%s max_stage_load_mean=%s' % (
            row['z'],
            row['total_slots_mean'],
            row['total_slots_max'],
            row['max_stage_packets_per_subchannel_mean']))


if __name__ == '__main__':
    main()
