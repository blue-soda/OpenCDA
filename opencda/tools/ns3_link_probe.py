# -*- coding: utf-8 -*-
"""Deterministic OpenCDA-to-NS3 link probe without CARLA.

The probe sends a tiny synthetic vehicle state and hand-written transfer
requests to NS3. It is intended for validating that OpenCDA-requested
subchannels are honored by the NR sidelink manual scheduler.
"""

import argparse
import csv
import os
import time

from opencda.core.networking.ns3_co_simulation.bridge.carla_ns3_bridge import (
    CarlaNs3Bridge,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Probe NS3 manual subchannels.')
    parser.add_argument('--ns3-host', default=None)
    parser.add_argument('--sync-timeout', type=float, default=10.0)
    parser.add_argument('--drain-seconds', type=float, default=0.6)
    parser.add_argument('--packet-size', type=int, default=400)
    parser.add_argument('--case', choices=[
        'success', 'edge_success', 'conflict', 'out_of_band'],
                        default='success')
    parser.add_argument('--upload-plan-output', default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_vehicles():
    vehicles = []
    for index, carla_id in enumerate([1, 2, 3, 4]):
        vehicles.append({
            'id': index,
            'carla_id': carla_id,
            'position': {'x': float(index * 5), 'y': 0.0, 'z': 0.0},
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'heading': 0.0,
            'speed': 0.0,
        })
    return vehicles


def build_requests(case_name, packet_size):
    if case_name == 'success':
        return [
            {'source': 1, 'target': 2, 'size': packet_size,
             'pkt_id': 1, 'sc_start': 0, 'sc_num': 1},
            {'source': 3, 'target': 4, 'size': packet_size,
             'pkt_id': 2, 'sc_start': 1, 'sc_num': 1},
        ]
    if case_name == 'edge_success':
        return [
            {'source': 1, 'target': 2, 'size': packet_size,
             'pkt_id': 1, 'sc_start': 9, 'sc_num': 1},
        ]
    if case_name == 'conflict':
        return [
            {'source': 1, 'target': 2, 'size': packet_size,
             'pkt_id': 1, 'sc_start': 0, 'sc_num': 1},
            {'source': 3, 'target': 4, 'size': packet_size,
             'pkt_id': 2, 'sc_start': 0, 'sc_num': 1},
        ]
    return [
        {'source': 1, 'target': 2, 'size': packet_size,
         'pkt_id': 1, 'sc_start': 10, 'sc_num': 1},
    ]


def write_upload_plan(path, requests):
    if not path:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'timestamp', 'area_id', 'source_id', 'target_id', 'bytes',
            'upload_type', 'pkt_id',
        ])
        writer.writeheader()
        for request in requests:
            writer.writerow({
                'timestamp': 'probe000',
                'area_id': '',
                'source_id': request['source'],
                'target_id': request['target'],
                'bytes': request['size'],
                'upload_type': 'probe_' + request.get('case', ''),
                'pkt_id': request['pkt_id'],
            })


def main():
    args = parse_args()
    vehicles = build_vehicles()
    requests = build_requests(args.case, args.packet_size)
    for request in requests:
        request['case'] = args.case
    write_upload_plan(args.upload_plan_output, requests)

    print('case=%s vehicles=%s requests=%s bytes=%s' % (
        args.case, len(vehicles), len(requests),
        sum(request['size'] for request in requests)))
    for request in requests:
        print('request pkt_id={pkt_id} {source}->{target} '
              'sc={sc_start}:{sc_num} bytes={size}'.format(**request))

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
        bridge.send_transfer_requests(requests)
        if args.drain_seconds > 0:
            if not bridge.sync_with_ns3(args.drain_seconds):
                raise RuntimeError('drain NS3 sync failed')
            time.sleep(min(args.drain_seconds, 0.2))
    finally:
        bridge.stop()


if __name__ == '__main__':
    main()
