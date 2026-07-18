# -*- coding: utf-8 -*-
"""Estimate prediction-box exchange cost for late-fusion baselines.

The offline perception trace records one row per receiver/source frame and
includes local ``pred_boxes`` counts.  This utility turns those counts into a
small communication budget for pure late fusion, so the paper can distinguish
raw LiDAR payload from prediction-box sharing overhead.
"""

import argparse
import csv
import math
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate late-fusion detection-box communication budget.')
    parser.add_argument('--trace-csv', required=True,
                        help='offline_inference trace CSV for pure late fusion.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for summary and per-frame CSV outputs.')
    parser.add_argument('--num-cavs', type=int, default=20,
                        help='Number of CAVs in the scene.')
    parser.add_argument('--deadline-ms', type=float, default=100.0,
                        help='Per-frame communication deadline.')
    parser.add_argument('--box-bytes', type=int, default=80,
                        help='Serialized bytes per predicted box.')
    parser.add_argument('--message-overhead-bytes', type=int, default=64,
                        help='Transport/app metadata bytes per sender message.')
    parser.add_argument('--packet-overhead-bytes', type=int, default=48,
                        help='PHY/MAC/header overhead charged per packet.')
    parser.add_argument('--mtu-bytes', type=int, default=1200,
                        help='Maximum payload bytes per packet before headers.')
    parser.add_argument('--total-bandwidth-mhz', type=float, default=20.0,
                        help='Total network bandwidth in MHz.')
    parser.add_argument('--subchannels', type=int, default=10,
                        help='Number of subchannels.')
    parser.add_argument('--spectral-efficiency', type=float, default=6.0,
                        help='Effective bits/s/Hz per subchannel.')
    parser.add_argument('--mac-gap-ms', type=float, default=0.2,
                        help='Scheduling guard/processing gap per packet.')
    parser.add_argument('--contention-round-ms', type=float, default=1.0,
                        help='Random-access contention round duration.')
    return parser.parse_args()


def safe_int(value, default=0):
    try:
        if value is None or value == '':
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_frame_box_counts(path):
    by_timestamp = defaultdict(dict)
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            timestamp = row.get('timestamp', '')
            try:
                receiver_id = int(row.get('receiver_id', ''))
            except ValueError:
                continue
            pred_boxes = safe_int(row.get('pred_boxes'), 0)
            by_timestamp[timestamp][receiver_id] = max(
                pred_boxes, by_timestamp[timestamp].get(receiver_id, 0))
    return by_timestamp


def message_bytes(box_count, box_bytes, message_overhead):
    if box_count <= 0:
        return 0
    return message_overhead + box_count * box_bytes


def packet_count(payload_bytes, mtu_bytes):
    if payload_bytes <= 0:
        return 0
    return int(math.ceil(payload_bytes / float(mtu_bytes)))


def packet_airtime_ms(payload_bytes, packet_overhead, mtu_bytes, rate_mbps):
    if payload_bytes <= 0:
        return []
    remaining = payload_bytes
    airtimes = []
    while remaining > 0:
        chunk = min(remaining, mtu_bytes)
        packet_bytes = chunk + packet_overhead
        airtimes.append(packet_bytes * 8.0 / max(rate_mbps, 1e-9) / 1000.0)
        remaining -= chunk
    return airtimes


def greedy_scheduled_ms(message_payloads, mtu_bytes, packet_overhead,
                        rate_mbps, subchannels, mac_gap_ms):
    channel_loads = [0.0 for _ in range(max(1, subchannels))]
    packet_airtimes = []
    for payload in sorted(message_payloads, reverse=True):
        packet_airtimes.extend(packet_airtime_ms(
            payload, packet_overhead, mtu_bytes, rate_mbps))
    for airtime in sorted(packet_airtimes, reverse=True):
        index = min(range(len(channel_loads)), key=channel_loads.__getitem__)
        channel_loads[index] += airtime + mac_gap_ms
    return max(channel_loads) if channel_loads else 0.0


def random_access_success(messages, subchannels, deadline_ms, round_ms):
    """Estimate slotted random subchannel access with a collision proxy.

    A message succeeds in a contention round only when it is the sole message on
    its chosen subchannel. Collided messages retry in the next round.  The
    expected number of successes for n messages and c subchannels is
    n * ((c - 1) / c) ** (n - 1).
    """
    if messages <= 0:
        return 1.0, 1.0, 0
    rounds = max(1, int(math.floor(deadline_ms / max(round_ms, 1e-9))))
    channels = max(1, subchannels)
    remaining = float(messages)
    for _round in range(rounds):
        if remaining <= 1e-9:
            remaining = 0.0
            break
        if channels == 1:
            expected_success = 1.0 if remaining <= 1.0 else 0.0
        else:
            expected_success = remaining * (
                (channels - 1.0) / channels) ** max(remaining - 1.0, 0.0)
        expected_success = min(expected_success, remaining)
        remaining -= expected_success
    ratio = max(0.0, min(1.0, (messages - remaining) / float(messages)))
    full_success_proxy = 1.0 if remaining < 0.5 else 0.0
    return ratio, full_success_proxy, rounds


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_frame(timestamp, sender_boxes, args, rate_mbps):
    nonzero = [(cav_id, boxes) for cav_id, boxes in sender_boxes.items()
               if boxes > 0]
    broadcast_payloads = [
        message_bytes(boxes, args.box_bytes, args.message_overhead_bytes)
        for _cav_id, boxes in nonzero
    ]
    unicast_payloads = []
    for _cav_id, boxes in nonzero:
        payload = message_bytes(
            boxes, args.box_bytes, args.message_overhead_bytes)
        unicast_payloads.extend([payload] * max(args.num_cavs - 1, 0))

    broadcast_bytes = sum(broadcast_payloads)
    unicast_bytes = sum(unicast_payloads)
    broadcast_packets = sum(packet_count(p, args.mtu_bytes)
                            for p in broadcast_payloads)
    unicast_packets = sum(packet_count(p, args.mtu_bytes)
                          for p in unicast_payloads)
    broadcast_ms = greedy_scheduled_ms(
        broadcast_payloads, args.mtu_bytes, args.packet_overhead_bytes,
        rate_mbps, args.subchannels, args.mac_gap_ms)
    unicast_ms = greedy_scheduled_ms(
        unicast_payloads, args.mtu_bytes, args.packet_overhead_bytes,
        rate_mbps, args.subchannels, args.mac_gap_ms)
    broadcast_ra_ratio, broadcast_ra_full, rounds = random_access_success(
        len(broadcast_payloads), args.subchannels, args.deadline_ms,
        args.contention_round_ms)
    unicast_ra_ratio, unicast_ra_full, _ = random_access_success(
        len(unicast_payloads), args.subchannels, args.deadline_ms,
        args.contention_round_ms)

    frame_seconds = args.deadline_ms / 1000.0
    return {
        'timestamp': timestamp,
        'senders_with_boxes': len(nonzero),
        'total_boxes': sum(boxes for _cav_id, boxes in nonzero),
        'broadcast_messages': len(broadcast_payloads),
        'broadcast_packets': broadcast_packets,
        'broadcast_bytes': broadcast_bytes,
        'broadcast_mbps': '%.6f' % (
            broadcast_bytes * 8.0 / frame_seconds / 1e6),
        'broadcast_scheduled_ms': '%.6f' % broadcast_ms,
        'broadcast_deadline_ok': int(broadcast_ms <= args.deadline_ms),
        'broadcast_random_access_delivery_ratio': '%.6f' % broadcast_ra_ratio,
        'broadcast_random_access_full_success_prob': '%.6f' %
        broadcast_ra_full,
        'alltoall_messages': len(unicast_payloads),
        'alltoall_packets': unicast_packets,
        'alltoall_bytes': unicast_bytes,
        'alltoall_mbps': '%.6f' % (
            unicast_bytes * 8.0 / frame_seconds / 1e6),
        'alltoall_scheduled_ms': '%.6f' % unicast_ms,
        'alltoall_deadline_ok': int(unicast_ms <= args.deadline_ms),
        'alltoall_random_access_delivery_ratio': '%.6f' % unicast_ra_ratio,
        'alltoall_random_access_full_success_prob': '%.6f' %
        unicast_ra_full,
        'contention_rounds': rounds,
    }


def build_summary(frame_rows, args, rate_mbps):
    numeric_fields = [
        'senders_with_boxes', 'total_boxes', 'broadcast_messages',
        'broadcast_packets', 'broadcast_bytes', 'broadcast_mbps',
        'broadcast_scheduled_ms', 'broadcast_deadline_ok',
        'broadcast_random_access_delivery_ratio',
        'broadcast_random_access_full_success_prob',
        'alltoall_messages', 'alltoall_packets', 'alltoall_bytes',
        'alltoall_mbps', 'alltoall_scheduled_ms', 'alltoall_deadline_ok',
        'alltoall_random_access_delivery_ratio',
        'alltoall_random_access_full_success_prob',
    ]
    summary = {
        'frames': len(frame_rows),
        'num_cavs': args.num_cavs,
        'deadline_ms': args.deadline_ms,
        'box_bytes': args.box_bytes,
        'message_overhead_bytes': args.message_overhead_bytes,
        'packet_overhead_bytes': args.packet_overhead_bytes,
        'mtu_bytes': args.mtu_bytes,
        'total_bandwidth_mhz': args.total_bandwidth_mhz,
        'subchannels': args.subchannels,
        'spectral_efficiency_bpshz': args.spectral_efficiency,
        'per_subchannel_rate_mbps': '%.6f' % rate_mbps,
    }
    for field in numeric_fields:
        values = [float(row[field]) for row in frame_rows]
        summary[field + '_mean'] = '%.6f' % (
            sum(values) / float(max(len(values), 1)))
        summary[field + '_p95'] = '%.6f' % percentile(values, 95)
        summary[field + '_max'] = '%.6f' % (max(values) if values else 0.0)
    return summary


def write_markdown(path, summary):
    with open(path, 'w') as stream:
        stream.write('# Late-Fusion Prediction-Box Communication Budget\n\n')
        stream.write('This artifact estimates whether all-CAV pure late '
                     'fusion can be naturally limited by a 100 ms V2V '
                     'communication deadline.\n\n')
        stream.write('| Scenario | Mean Mbps | Max Mbps | Mean scheduled ms | '
                     'Deadline OK mean | Random-access full success |\n')
        stream.write('| --- | ---: | ---: | ---: | ---: | ---: |\n')
        stream.write('| Broadcast one message per sender | %s | %s | %s | %s | %s |\n' % (
            summary['broadcast_mbps_mean'],
            summary['broadcast_mbps_max'],
            summary['broadcast_scheduled_ms_mean'],
            summary['broadcast_deadline_ok_mean'],
            summary['broadcast_random_access_full_success_prob_mean']))
        stream.write('| All-to-all unicast | %s | %s | %s | %s | %s |\n\n' % (
            summary['alltoall_mbps_mean'],
            summary['alltoall_mbps_max'],
            summary['alltoall_scheduled_ms_mean'],
            summary['alltoall_deadline_ok_mean'],
            summary['alltoall_random_access_full_success_prob_mean']))
        stream.write('Interpretation: scheduled deadline results model an '
                     'ideal channel assignment. Random-access results are a '
                     'coarse collision proxy where each outstanding message '
                     'chooses a random subchannel each contention round.\n')


def main():
    args = parse_args()
    if args.subchannels <= 0:
        raise ValueError('--subchannels must be positive')
    os.makedirs(args.output_dir, exist_ok=True)

    frame_boxes = read_frame_box_counts(args.trace_csv)
    rate_mbps = (
        args.total_bandwidth_mhz * args.spectral_efficiency /
        float(args.subchannels))
    rows = []
    for timestamp in sorted(frame_boxes):
        rows.append(summarize_frame(
            timestamp, frame_boxes[timestamp], args, rate_mbps))
    summary = build_summary(rows, args, rate_mbps)

    frame_path = os.path.join(args.output_dir, 'late_box_frame_budget.csv')
    summary_path = os.path.join(args.output_dir, 'late_box_summary.csv')
    notes_path = os.path.join(args.output_dir, 'late_box_budget.md')
    frame_fields = [
        'timestamp', 'senders_with_boxes', 'total_boxes',
        'broadcast_messages', 'broadcast_packets', 'broadcast_bytes',
        'broadcast_mbps', 'broadcast_scheduled_ms',
        'broadcast_deadline_ok',
        'broadcast_random_access_delivery_ratio',
        'broadcast_random_access_full_success_prob',
        'alltoall_messages', 'alltoall_packets', 'alltoall_bytes',
        'alltoall_mbps', 'alltoall_scheduled_ms',
        'alltoall_deadline_ok',
        'alltoall_random_access_delivery_ratio',
        'alltoall_random_access_full_success_prob',
        'contention_rounds',
    ]
    write_csv(frame_path, rows, frame_fields)
    write_csv(summary_path, [summary], list(summary.keys()))
    write_markdown(notes_path, summary)

    print('frames=%s' % summary['frames'])
    print('broadcast mean/max Mbps=%s/%s scheduled_ms=%s deadline_ok=%s' % (
        summary['broadcast_mbps_mean'],
        summary['broadcast_mbps_max'],
        summary['broadcast_scheduled_ms_mean'],
        summary['broadcast_deadline_ok_mean']))
    print('alltoall mean/max Mbps=%s/%s scheduled_ms=%s deadline_ok=%s' % (
        summary['alltoall_mbps_mean'],
        summary['alltoall_mbps_max'],
        summary['alltoall_scheduled_ms_mean'],
        summary['alltoall_deadline_ok_mean']))
    print('outputs=%s,%s,%s' % (frame_path, summary_path, notes_path))


if __name__ == '__main__':
    main()
