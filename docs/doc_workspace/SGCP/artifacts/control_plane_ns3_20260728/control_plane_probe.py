# -*- coding: utf-8 -*-
"""Synthetic SGCP control-plane replay through the CARLA-NS3 bridge.

This probe sends one frame worth of small control packets representing
coalition proposals/replies, potential-verified checks, membership updates,
and scheduler summaries/grants/ACKs. It does not run perception inference.
"""

import argparse
import csv
import os
import time

from opencda.core.networking.ns3_co_simulation.bridge.carla_ns3_bridge import (
    CarlaNs3Bridge,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ns3-host', default=None)
    parser.add_argument('--vehicles', type=int, default=20)
    parser.add_argument('--packet-size', type=int, default=400)
    parser.add_argument('--subchannels', type=int, default=10)
    parser.add_argument('--profile', choices=['unaggregated', 'aggregated'],
                        default='unaggregated')
    parser.add_argument('--drain-seconds', type=float, default=2.0)
    parser.add_argument('--sync-timeout', type=float, default=20.0)
    parser.add_argument('--batch-size', type=int, default=0,
                        help='0 sends all packets in one transfer_requests '
                             'batch. Positive values send batches and sync '
                             'after each batch.')
    parser.add_argument('--batch-step-ms', type=float, default=0.5)
    parser.add_argument('--first-gap-ms', type=float, default=None,
                        help='Optional larger gap after the first batch, in '
                             'milliseconds. Later batches use '
                             '--batch-step-ms.')
    parser.add_argument('--pre-send-sync-ms', type=float, default=0.0,
                        help='Advance NS3 to this simulator time before '
                             'sending the first transfer batch. Default 0 '
                             'preserves the original probe behavior.')
    parser.add_argument('--endpoint-disjoint', action='store_true',
                        help='Build each subchannel batch as an endpoint-'
                             'disjoint matching when possible.')
    parser.add_argument('--cast-type', choices=['unicast', 'broadcast'],
                        default='unicast',
                        help='Use unicast request targets or one NR sidelink '
                             'broadcast/group address per control summary.')
    parser.add_argument('--limit-requests', type=int, default=0,
                        help='Keep only the first N synthetic control '
                             'requests after building the profile.')
    parser.add_argument('--upload-plan-output', required=True)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_vehicles(count):
    vehicles = []
    for index in range(count):
        row = index // 5
        col = index % 5
        vehicles.append({
            'id': index,
            'carla_id': index + 1,
            'position': {
                'x': float(col * 8),
                'y': float(row * 8),
                'z': 0.0,
            },
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'heading': 0.0,
            'speed': 0.0,
        })
    return vehicles


def control_packet_counts(profile):
    if profile == 'aggregated':
        return [
            ('coalition_round_summary', 60),
            ('scheduler_summary', 6),
            ('scheduler_grant', 4),
        ]
    return [
        ('coalition_proposal', 60),
        ('coalition_reply', 60),
        ('pv_source_target_check', 120),
        ('membership_update', 30),
        ('scheduler_summary', 14),
        ('scheduler_grant', 10),
        ('scheduler_ack_or_reserve', 20),
    ]


def endpoint_disjoint_pair(pkt_id, vehicle_count, subchannels):
    half = vehicle_count // 2
    if half < subchannels:
        return None
    batch_index = (pkt_id - 1) // subchannels
    offset = (pkt_id - 1) % subchannels
    if batch_index % 2 == 0:
        return offset + 1, half + offset + 1
    return half + offset + 1, offset + 1


def build_requests(vehicle_count, packet_size, subchannels, profile,
                   endpoint_disjoint=False, cast_type='unicast'):
    requests = []
    pkt_id = 1
    for upload_type, count in control_packet_counts(profile):
        for local_index in range(count):
            pair = endpoint_disjoint_pair(
                pkt_id, vehicle_count, subchannels) \
                if endpoint_disjoint else None
            if pair is not None:
                source, target = pair
            else:
                source = (pkt_id - 1) % vehicle_count + 1
                target = (source + 6 + local_index) % vehicle_count + 1
                if target == source:
                    target = target % vehicle_count + 1
            request = {
                'source': source,
                'target': target,
                'size': int(packet_size),
                'pkt_id': pkt_id,
                'sc_start': (pkt_id - 1) % subchannels,
                'sc_num': 1,
                'upload_type': upload_type,
            }
            if cast_type == 'broadcast':
                request['target'] = 255
                request['cast_type'] = 'broadcast'
            requests.append(request)
            pkt_id += 1
    return requests


def write_upload_plan(path, requests):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'timestamp',
            'area_id',
            'source_id',
            'target_id',
            'bytes',
            'upload_type',
            'pkt_id',
        ])
        writer.writeheader()
        for request in requests:
            writer.writerow({
                'timestamp': 'control000',
                'area_id': '',
                'source_id': request['source'],
                'target_id': request['target'],
                'bytes': request['size'],
                'upload_type': request['upload_type'],
                'pkt_id': request['pkt_id'],
            })


def send_requests(bridge, requests, batch_size, batch_step_ms,
                  first_gap_ms=None, start_time_s=0.0):
    if batch_size <= 0:
        bridge.send_transfer_requests(requests)
        return
    sim_time = float(start_time_s)
    batch_index = 0
    for start in range(0, len(requests), batch_size):
        batch = requests[start:start + batch_size]
        bridge.send_transfer_requests(batch)
        step_ms = batch_step_ms
        if batch_index == 0 and first_gap_ms is not None:
            step_ms = first_gap_ms
        sim_time += step_ms / 1000.0
        if not bridge.sync_with_ns3(sim_time):
            raise RuntimeError('NS3 sync failed after batch %s' % start)
        batch_index += 1


def main():
    args = parse_args()
    vehicles = build_vehicles(args.vehicles)
    requests = build_requests(args.vehicles, args.packet_size,
                              args.subchannels, args.profile,
                              endpoint_disjoint=args.endpoint_disjoint,
                              cast_type=args.cast_type)
    if args.limit_requests > 0:
        requests = requests[:args.limit_requests]
    write_upload_plan(args.upload_plan_output, requests)
    print('control_plane_probe vehicles=%d requests=%d bytes=%d '
          'packet_size=%d subchannels=%d batch_size=%d profile=%s' % (
              len(vehicles),
              len(requests),
              sum(item['size'] for item in requests),
              args.packet_size,
              args.subchannels,
              args.batch_size,
              args.profile))
    if args.dry_run:
        return

    bridge = CarlaNs3Bridge(ns3_host=args.ns3_host) if args.ns3_host else \
        CarlaNs3Bridge()
    bridge.sync_timeout = args.sync_timeout
    bridge.enable_time_sync(True)
    bridge.start()
    try:
        bridge.send_vehicles_num(len(vehicles))
        bridge.send_vehicles_position(vehicles)
        if not bridge.sync_with_ns3(0.0):
            raise RuntimeError('initial NS3 sync failed')
        if args.pre_send_sync_ms > 0:
            if not bridge.sync_with_ns3(args.pre_send_sync_ms / 1000.0):
                raise RuntimeError('pre-send NS3 sync failed')
        send_requests(bridge, requests, args.batch_size, args.batch_step_ms,
                      first_gap_ms=args.first_gap_ms,
                      start_time_s=args.pre_send_sync_ms / 1000.0)
        if args.drain_seconds > 0:
            if not bridge.sync_with_ns3(args.drain_seconds):
                raise RuntimeError('drain NS3 sync failed')
            time.sleep(min(args.drain_seconds, 0.2))
    finally:
        bridge.stop()


if __name__ == '__main__':
    main()
