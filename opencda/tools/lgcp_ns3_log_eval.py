# -*- coding: utf-8 -*-
"""
Summarize NS3 delivery logs for LGCP upload plans.

The NS3 CARLA bridge prints successful callbacks as ``cam_received`` JSON
messages. This utility aligns those visible callbacks with an LGCP
``upload_plan.csv`` and reports a conservative bridge-observed delivery
summary. It also parses text PHY decode diagnostics when the NS3 build prints
``PSCCH_DECODE_*`` and ``PSSCH_DECODE_*`` lines.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict, deque


CAM_PATTERN = re.compile(r'SendMsgToCarla: (\{.*?\}), send_to_carla_fd')
PSCCH_PATTERN = re.compile(
    r'\[PSCCH_DECODE_(OK|FAIL)\]\s+txRnti=(\d+)\s+dstL2Id=(\d+)')
PSSCH_PATTERN = re.compile(r'\[PSSCH_DECODE_(OK|FAIL)\].*')
RLC_PATTERN = re.compile(r'\[NRSL_RLC_(TX|RX|DROP)\].*')
REQUEST_PHY_PATTERN = re.compile(r'\[NRSL_(PHY|HARQ)_EVENT\].*')
TRANSFER_FRAME_PATTERN = re.compile(r'Processing transfer_requests with')
FIELD_PATTERN = re.compile(r'([A-Za-z][A-Za-z0-9_]*)=([^\s]+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP NS3 bridge-observed delivery logs.')
    parser.add_argument('--ns3-stdout', required=True,
                        help='Path to NS3 stdout log.')
    parser.add_argument('--upload-plan', required=True,
                        help='Path to LGCP upload_plan.csv.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for CSV summaries.')
    parser.add_argument('--rsu-node-id', type=int, default=21,
                        help='Positive NS3 node id used for RSU.')
    parser.add_argument('--frame-interval-ms', type=float, default=100.0,
                        help='Replay frame interval in milliseconds.')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Number of plan frames to evaluate. Use 0 for all.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from in upload_plan.csv.')
    parser.add_argument('--frame-step', type=int, default=1,
                        help='Evaluate every Nth selected plan frame.')
    return parser.parse_args()


def endpoint_to_node(value, rsu_node_id):
    value = str(value)
    if value.upper() == 'RSU':
        return str(int(rsu_node_id))
    return value


def read_upload_plan(path, rsu_node_id, max_frames, start_index, frame_step):
    rows = []
    all_timestamps = []
    with open(path, newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream), start=1):
            if row['timestamp'] not in all_timestamps:
                all_timestamps.append(row['timestamp'])
            row = dict(row)
            row['plan_index'] = index
            row['source_node'] = endpoint_to_node(row['source_id'], rsu_node_id)
            row['target_node'] = endpoint_to_node(row['target_id'], rsu_node_id)
            row['bytes'] = int(float(row['bytes']))
            rows.append(row)

    selected = all_timestamps[start_index:] if max_frames == 0 else \
        all_timestamps[start_index:start_index + max_frames]
    timestamps = selected[::max(1, frame_step)]
    selected_set = set(timestamps)
    timestamp_to_frame = {
        timestamp: frame_index
        for frame_index, timestamp in enumerate(timestamps)
    }
    rows = [row for row in rows if row['timestamp'] in selected_set]
    frame_pkt_counters = Counter()
    for row in rows:
        row['frame_index'] = timestamp_to_frame[row['timestamp']]
        frame_pkt_counters[row['timestamp']] += 1
        row['frame_pkt_id'] = frame_pkt_counters[row['timestamp']]
        row['request_id'] = int(row.get('pkt_id') or row['plan_index'])
    return rows, timestamps


def parse_cam_received(path, frame_interval_ms):
    records = []
    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line_no, line in enumerate(stream, start=1):
            match = CAM_PATTERN.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if payload.get('type') != 'cam_received':
                continue
            send_ms = float(payload.get('send_timestamp', 0.0))
            receive_ms = float(payload.get('receive_timestamp', 0.0))
            frame_index = int(round(send_ms / frame_interval_ms)) \
                if frame_interval_ms > 0 else 0
            records.append({
                'line_no': line_no,
                'request_id': int(float(payload.get('request_id', 0))),
                'source_node': str(payload.get('sender_id', '')),
                'target_node': str(payload.get('receiver_id', '')),
                'bytes': int(float(payload.get('packet_size', 0))),
                'send_timestamp_ms': send_ms,
                'receive_timestamp_ms': receive_ms,
                'delay_ms': receive_ms - send_ms,
                'frame_index': frame_index,
                'is_last_packet': payload.get('is_last_packet', ''),
            })
    return records


def parse_value(value):
    value = str(value).strip().rstrip(',;')
    try:
        if any(char in value for char in ('.', 'e', 'E')):
            return float(value)
        return int(value)
    except ValueError:
        return value


def fields_from_line(line):
    return {
        key: parse_value(value)
        for key, value in FIELD_PATTERN.findall(line)
    }


def normalize_reason(fields, status):
    if status == 'OK':
        return ''
    return str(fields.get('reason', 'decode_fail'))


def parse_phy_decodes(path):
    records = []
    pending_pscch = None
    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line_no, line in enumerate(stream, start=1):
            pssch_match = PSSCH_PATTERN.search(line)
            if pssch_match:
                status = pssch_match.group(1)
                fields = fields_from_line(line)
                records.append({
                    'line_no': line_no,
                    'channel': 'PSSCH',
                    'status': status,
                    'tx_rnti': fields.get('txRnti', ''),
                    'dst_l2_id': fields.get('dstL2Id', ''),
                    'harq_id': fields.get('harqId', ''),
                    'tb_size': fields.get('tbSize', ''),
                    'tbler': fields.get('tbler', ''),
                    'sinr_avg': fields.get('sinrAvg', ''),
                    'sinr_min': fields.get('sinrMin', ''),
                    'reason': normalize_reason(fields, status),
                    'frame': fields.get('frame', ''),
                    'subframe': fields.get('subframe', ''),
                    'slot': fields.get('slot', ''),
                })
                pending_pscch = None
                continue

            pscch_match = PSCCH_PATTERN.search(line)
            if pscch_match:
                pending_pscch = {
                    'line_no': line_no,
                    'channel': 'PSCCH',
                    'status': pscch_match.group(1),
                    'tx_rnti': int(pscch_match.group(2)),
                    'dst_l2_id': int(pscch_match.group(3)),
                }
                continue

            if pending_pscch is not None and 'rbStart=' in line:
                fields = fields_from_line(line)
                status = pending_pscch['status']
                record = dict(pending_pscch)
                record.update({
                    'harq_id': '',
                    'tb_size': '',
                    'tbler': fields.get('tbler', ''),
                    'sinr_avg': fields.get('sinrAvg', ''),
                    'sinr_min': fields.get('sinrMin', ''),
                    'reason': normalize_reason(fields, status),
                    'frame': fields.get('frame', ''),
                    'subframe': fields.get('subframe', ''),
                    'slot': fields.get('slot', ''),
                })
                records.append(record)
                pending_pscch = None
    return records


def parse_rlc_events(path):
    records = []
    frame_index = -1
    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line_no, line in enumerate(stream, start=1):
            if TRANSFER_FRAME_PATTERN.search(line):
                frame_index += 1
                continue
            match = RLC_PATTERN.search(line)
            if not match:
                continue
            fields = fields_from_line(line.replace('\x00', '0'))
            records.append({
                'line_no': line_no,
                'frame_index': frame_index,
                'event': match.group(1),
                'rnti': fields.get('rnti', ''),
                'lcid': fields.get('lcid', ''),
                'source_node': str(fields.get('srcL2Id', '')),
                'target_node': str(fields.get('dstL2Id', '')),
                'request_id': int(fields.get('request_id', 0) or 0),
                'sn': fields.get('sn', ''),
                'fi': fields.get('fi', ''),
                'size': fields.get('size', ''),
                'harq_id': fields.get('harqId', ''),
                'bwp_id': fields.get('bwpId', ''),
                'layer': fields.get('layer', ''),
                'vr_ur': fields.get('vrUr', ''),
                'vr_uh': fields.get('vrUh', ''),
            })
    return records


def split_request_ids(fields):
    if 'request_id' in fields:
        request_id = int(fields.get('request_id') or 0)
        return [request_id] if request_id > 0 else [0]
    if 'request_ids' not in fields:
        return [0]
    values = str(fields.get('request_ids') or '')
    request_ids = []
    for value in values.split(','):
        value = value.strip()
        if not value:
            continue
        try:
            request_ids.append(int(value))
        except ValueError:
            continue
    return request_ids or [0]


def first_present(fields, names, default=''):
    for name in names:
        if name in fields:
            return fields[name]
    return default


def parse_phy_harq_request_events(path):
    records = []
    frame_index = -1
    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line_no, line in enumerate(stream, start=1):
            if TRANSFER_FRAME_PATTERN.search(line):
                frame_index += 1
                continue
            match = REQUEST_PHY_PATTERN.search(line)
            if not match:
                continue
            fields = fields_from_line(line.replace('\x00', '0'))
            for request_id in split_request_ids(fields):
                records.append({
                    'line_no': line_no,
                    'frame_index': int(fields.get('frame_index', frame_index)),
                    'trace_layer': match.group(1),
                    'event': str(fields.get('event', '')),
                    'request_id': int(request_id),
                    'source_node': str(first_present(
                        fields, ('sender_l2_id', 'srcL2Id', 'txRnti'))),
                    'target_node': str(first_present(
                        fields, ('receiver_l2_id', 'dstL2Id'))),
                    'tb_size': fields.get('tb_size', fields.get('tbSize', '')),
                    'subchannel_start': fields.get('subchannel_start', ''),
                    'subchannel_num': fields.get('subchannel_num', ''),
                    'slot': fields.get('slot', ''),
                    'harq_id': fields.get('harq_id', fields.get('harqId', '')),
                    'sinr': fields.get('sinr', fields.get('sinrAvg', '')),
                    'tbler': fields.get('tbler', ''),
                    'reason': fields.get('reason', ''),
                    'raw_event': line.strip(),
                })
    return records


def build_match_queues(plan_rows):
    queues = defaultdict(deque)
    for row in plan_rows:
        pkt_key = (
            'pkt_id',
            int(row['frame_index']),
            int(row['request_id']),
        )
        legacy_pkt_key = (
            'pkt_id',
            int(row['frame_index']),
            int(row['frame_pkt_id']),
        )
        key = (
            'endpoint',
            int(row['frame_index']),
            row['source_node'],
            row['target_node'],
            int(row['bytes']),
        )
        queues[pkt_key].append(row)
        queues[legacy_pkt_key].append(row)
        queues[key].append(row)
    return queues


def align_records(cam_records, plan_rows, timestamps):
    queues = build_match_queues(plan_rows)
    aligned = []
    for record in cam_records:
        request_id = int(record.get('request_id', 0))
        pkt_key = (
            'pkt_id',
            int(record['frame_index']),
            request_id,
        )
        endpoint_key = (
            'endpoint',
            int(record['frame_index']),
            record['source_node'],
            record['target_node'],
            int(record['bytes']),
        )
        match_method = ''
        if request_id > 0 and queues.get(pkt_key):
            plan = queues[pkt_key].popleft()
            match_method = 'frame_request_id'
        elif queues.get(endpoint_key):
            plan = queues[endpoint_key].popleft()
            match_method = 'frame_endpoint_bytes'
        else:
            plan = None
        output = dict(record)
        if plan is None:
            output.update({
                'timestamp': timestamps[record['frame_index']]
                if 0 <= record['frame_index'] < len(timestamps) else '',
                'area_id': '',
                'upload_type': 'unmatched',
                'plan_index': '',
                'pkt_id': '',
                'matched': 0,
                'match_method': '',
            })
        else:
            output.update({
                'timestamp': plan['timestamp'],
                'area_id': plan['area_id'],
                'upload_type': plan.get('upload_type', ''),
                'plan_index': plan['plan_index'],
                'pkt_id': plan.get('request_id', ''),
                'matched': 1,
                'match_method': match_method,
            })
        aligned.append(output)
    return aligned


def safe_mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, ratio):
    values = sorted(values)
    if not values:
        return 0.0
    index = int(round((len(values) - 1) * ratio))
    return values[index]


def aggregate(plan_rows, aligned_records, timestamps):
    planned_by_frame = Counter(row['frame_index'] for row in plan_rows)
    planned_bytes_by_frame = Counter()
    planned_by_type = Counter(row.get('upload_type', '') for row in plan_rows)
    planned_bytes_by_type = Counter()
    for row in plan_rows:
        planned_bytes_by_frame[row['frame_index']] += row['bytes']
        planned_bytes_by_type[row.get('upload_type', '')] += row['bytes']

    observed_by_frame = Counter(row['frame_index'] for row in aligned_records)
    observed_bytes_by_frame = Counter()
    delays_by_frame = defaultdict(list)
    observed_by_type = Counter()
    observed_bytes_by_type = Counter()
    delays_by_type = defaultdict(list)
    for row in aligned_records:
        frame_index = int(row['frame_index'])
        upload_type = row.get('upload_type', 'unmatched')
        observed_bytes_by_frame[frame_index] += int(row['bytes'])
        delays_by_frame[frame_index].append(float(row['delay_ms']))
        observed_by_type[upload_type] += 1
        observed_bytes_by_type[upload_type] += int(row['bytes'])
        delays_by_type[upload_type].append(float(row['delay_ms']))

    frame_summary = []
    for frame_index, timestamp in enumerate(timestamps):
        planned = planned_by_frame[frame_index]
        observed = observed_by_frame[frame_index]
        frame_summary.append({
            'frame_index': frame_index,
            'timestamp': timestamp,
            'planned_requests': planned,
            'observed_cam_received': observed,
            'bridge_observed_delivery_ratio': observed / planned
            if planned else 0.0,
            'planned_bytes': planned_bytes_by_frame[frame_index],
            'observed_bytes': observed_bytes_by_frame[frame_index],
            'avg_delay_ms': safe_mean(delays_by_frame[frame_index]),
            'p95_delay_ms': percentile(delays_by_frame[frame_index], 0.95),
            'max_delay_ms': max(delays_by_frame[frame_index])
            if delays_by_frame[frame_index] else 0.0,
        })

    type_summary = []
    for upload_type in sorted(set(planned_by_type) | set(observed_by_type)):
        planned = planned_by_type[upload_type]
        observed = observed_by_type[upload_type]
        type_summary.append({
            'upload_type': upload_type,
            'planned_requests': planned,
            'observed_cam_received': observed,
            'bridge_observed_delivery_ratio': observed / planned
            if planned else 0.0,
            'planned_bytes': planned_bytes_by_type[upload_type],
            'observed_bytes': observed_bytes_by_type[upload_type],
            'avg_delay_ms': safe_mean(delays_by_type[upload_type]),
            'p95_delay_ms': percentile(delays_by_type[upload_type], 0.95),
            'max_delay_ms': max(delays_by_type[upload_type])
            if delays_by_type[upload_type] else 0.0,
        })

    delays = [float(row['delay_ms']) for row in aligned_records]
    planned_total = len(plan_rows)
    observed_total = len(aligned_records)
    summary = [{
        'planned_requests': planned_total,
        'observed_cam_received': observed_total,
        'matched_cam_received': sum(int(row['matched']) for row in aligned_records),
        'bridge_observed_delivery_ratio': observed_total / planned_total
        if planned_total else 0.0,
        'planned_bytes': sum(row['bytes'] for row in plan_rows),
        'observed_bytes': sum(int(row['bytes']) for row in aligned_records),
        'avg_delay_ms': safe_mean(delays),
        'p95_delay_ms': percentile(delays, 0.95),
        'max_delay_ms': max(delays) if delays else 0.0,
        'note': 'bridge-observed cam_received callbacks, not PHY-layer trace',
    }]
    return frame_summary, type_summary, summary


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, '')
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def aggregate_phy(phy_records):
    grouped = defaultdict(list)
    for row in phy_records:
        grouped[(row['channel'], row['status'], row.get('reason', ''))].append(row)

    summary = []
    total_by_channel = Counter(row['channel'] for row in phy_records)
    for (channel, status, reason), rows in sorted(grouped.items()):
        total = total_by_channel[channel]
        summary.append({
            'channel': channel,
            'status': status,
            'reason': reason,
            'count': len(rows),
            'channel_ratio': len(rows) / total if total else 0.0,
            'avg_sinr': safe_mean(numeric_values(rows, 'sinr_avg')),
            'avg_sinr_min': safe_mean(numeric_values(rows, 'sinr_min')),
            'avg_tbler': safe_mean(numeric_values(rows, 'tbler')),
        })
    return summary


def enrich_rlc_records(rlc_records, plan_rows, timestamps):
    plan_by_frame_pkt = {
        (int(row['frame_index']), int(row['request_id'])): row
        for row in plan_rows
    }
    plan_by_frame_legacy_pkt = {
        (int(row['frame_index']), int(row['frame_pkt_id'])): row
        for row in plan_rows
    }
    pkt_counts = Counter(int(row['request_id']) for row in plan_rows)
    plan_by_unique_pkt = {
        int(row['request_id']): row
        for row in plan_rows
        if pkt_counts[int(row['request_id'])] == 1
    }
    enriched = []
    for record in rlc_records:
        output = dict(record)
        plan = plan_by_frame_pkt.get(
            (int(record['frame_index']), int(record['request_id'])))
        if plan is None:
            plan = plan_by_frame_legacy_pkt.get(
                (int(record['frame_index']), int(record['request_id'])))
        if plan is None:
            plan = plan_by_unique_pkt.get(int(record['request_id']))
        if plan is None:
            output.update({
                'timestamp': timestamps[record['frame_index']]
                if 0 <= record['frame_index'] < len(timestamps) else '',
                'area_id': '',
                'upload_type': 'unmatched',
                'plan_index': '',
                'pkt_id': '',
                'matched': 0,
            })
        else:
            output.update({
                'timestamp': plan['timestamp'],
                'frame_index': plan['frame_index'],
                'area_id': plan['area_id'],
                'upload_type': plan.get('upload_type', ''),
                'plan_index': plan['plan_index'],
                'pkt_id': plan.get('request_id', ''),
                'matched': 1,
            })
        enriched.append(output)
    return enriched


def enrich_request_events(records, plan_rows, timestamps):
    plan_by_frame_pkt = {
        (int(row['frame_index']), int(row['request_id'])): row
        for row in plan_rows
    }
    plan_by_frame_legacy_pkt = {
        (int(row['frame_index']), int(row['frame_pkt_id'])): row
        for row in plan_rows
    }
    pkt_counts = Counter(int(row['request_id']) for row in plan_rows)
    plan_by_unique_pkt = {
        int(row['request_id']): row
        for row in plan_rows
        if pkt_counts[int(row['request_id'])] == 1
    }
    enriched = []
    for record in records:
        output = dict(record)
        plan = plan_by_frame_pkt.get(
            (int(record['frame_index']), int(record['request_id'])))
        if plan is None:
            plan = plan_by_frame_legacy_pkt.get(
                (int(record['frame_index']), int(record['request_id'])))
        if plan is None:
            plan = plan_by_unique_pkt.get(int(record['request_id']))
        if plan is None:
            output.update({
                'timestamp': timestamps[record['frame_index']]
                if 0 <= record['frame_index'] < len(timestamps) else '',
                'area_id': '',
                'upload_type': 'unmatched',
                'plan_index': '',
                'pkt_id': '',
                'matched': 0,
            })
        else:
            output.update({
                'timestamp': plan['timestamp'],
                'frame_index': plan['frame_index'],
                'area_id': plan['area_id'],
                'upload_type': plan.get('upload_type', ''),
                'plan_index': plan['plan_index'],
                'pkt_id': plan.get('request_id', ''),
                'matched': 1,
            })
        enriched.append(output)
    return enriched


def aggregate_rlc(rlc_records, plan_rows):
    planned_total = len(plan_rows)
    event_counts = Counter(row['event'] for row in rlc_records)
    matched_counts = Counter(row['event'] for row in rlc_records
                             if int(row.get('matched', 0)))
    by_request = {}
    for row in plan_rows:
        request_id = int(row['request_id'])
        key = (int(row['frame_index']), request_id)
        by_request[key] = {
            'frame_index': row['frame_index'],
            'timestamp': row.get('timestamp', ''),
            'request_id': request_id,
            'source_node': row['source_node'],
            'target_node': row['target_node'],
            'area_id': row.get('area_id', ''),
            'upload_type': row.get('upload_type', ''),
            'plan_index': row.get('plan_index', ''),
            'planned_bytes': row.get('bytes', 0),
            'tx_events': 0,
            'rx_events': 0,
            'drop_events': 0,
            'tx_bytes': 0,
            'rx_bytes': 0,
            'drop_bytes': 0,
            'matched': 1,
        }

    for row in rlc_records:
        key = (int(row['frame_index']), int(row['request_id']))
        item = by_request.setdefault(key, {
            'frame_index': row['frame_index'],
            'timestamp': row.get('timestamp', ''),
            'request_id': row['request_id'],
            'source_node': row['source_node'],
            'target_node': row['target_node'],
            'area_id': row.get('area_id', ''),
            'upload_type': row.get('upload_type', ''),
            'plan_index': row.get('plan_index', ''),
            'planned_bytes': '',
            'tx_events': 0,
            'rx_events': 0,
            'drop_events': 0,
            'tx_bytes': 0,
            'rx_bytes': 0,
            'drop_bytes': 0,
            'matched': row.get('matched', 0),
        })
        size = int(row.get('size') or 0)
        if row['event'] == 'TX':
            item['tx_events'] += 1
            item['tx_bytes'] += size
        elif row['event'] == 'RX':
            item['rx_events'] += 1
            item['rx_bytes'] += size
        elif row['event'] == 'DROP':
            item['drop_events'] += 1
            item['drop_bytes'] += size

    request_rows = sorted(by_request.values(),
                          key=lambda item: (int(item['frame_index']),
                                            int(item['request_id'])))
    for item in request_rows:
        item['has_rlc_tx'] = int(item['tx_events'] > 0)
        item['has_rlc_rx'] = int(item['rx_events'] > 0)
        item['has_rlc_drop'] = int(item['drop_events'] > 0)
        item['rlc_complete'] = int(
            item['tx_events'] > 0 and
            item['rx_events'] >= item['tx_events'] and
            item['drop_events'] == 0)
        item['rlc_partial'] = int(
            item['rx_events'] > 0 and not item['rlc_complete'])
        item['rlc_no_rx'] = int(
            item['tx_events'] > 0 and item['rx_events'] == 0)

    any_tx_requests = [row for row in request_rows if row['has_rlc_tx']]
    any_rx_requests = [row for row in request_rows if row['has_rlc_rx']]
    any_drop_requests = [row for row in request_rows if row['has_rlc_drop']]
    complete_requests = [row for row in request_rows if row['rlc_complete']]
    partial_requests = [row for row in request_rows if row['rlc_partial']]
    no_rx_requests = [row for row in request_rows if row['rlc_no_rx']]
    summary = [{
        'planned_requests': planned_total,
        'rlc_tx_events': event_counts['TX'],
        'rlc_rx_events': event_counts['RX'],
        'rlc_drop_events': event_counts['DROP'],
        'matched_rlc_tx_events': matched_counts['TX'],
        'matched_rlc_rx_events': matched_counts['RX'],
        'matched_rlc_drop_events': matched_counts['DROP'],
        'requests_with_any_rlc_tx': len(any_tx_requests),
        'requests_with_any_rlc_rx': len(any_rx_requests),
        'requests_with_any_rlc_drop': len(any_drop_requests),
        'rlc_complete_requests': len(complete_requests),
        'rlc_partial_requests': len(partial_requests),
        'rlc_no_rx_requests': len(no_rx_requests),
        'request_any_rlc_rx_ratio': len(any_rx_requests) / planned_total
        if planned_total else 0.0,
        'request_rlc_complete_ratio': len(complete_requests) / planned_total
        if planned_total else 0.0,
        'request_rlc_partial_ratio': len(partial_requests) / planned_total
        if planned_total else 0.0,
        'note': 'RLC complete means RX segment/event count covers TX segment/event count for the request_id and no RLC DROP was observed; cam_received remains the application-visible completion signal',
    }]
    return summary, request_rows


def build_request_lifecycle(plan_rows, aligned_records, rlc_records,
                            phy_request_records):
    lifecycle = {}
    for row in plan_rows:
        request_id = int(row['request_id'])
        key = (int(row['frame_index']), request_id)
        lifecycle[key] = {
            'frame_index': row['frame_index'],
            'timestamp': row.get('timestamp', ''),
            'request_id': request_id,
            'source_node': row['source_node'],
            'target_node': row['target_node'],
            'area_id': row.get('area_id', ''),
            'upload_type': row.get('upload_type', ''),
            'planned_bytes': row.get('bytes', 0),
            'rlc_tx_count': 0,
            'rlc_rx_count': 0,
            'rlc_drop_count': 0,
            'phy_schedule_count': 0,
            'pscch_ok_count': 0,
            'pscch_fail_count': 0,
            'pssch_ok_count': 0,
            'pssch_fail_count': 0,
            'harq_ack_count': 0,
            'harq_nack_count': 0,
            'harq_timeout_count': 0,
            'cam_received': 0,
            'terminal_state': 'planned_only',
        }

    for row in rlc_records:
        if not int(row.get('matched', 0)):
            continue
        key = (int(row['frame_index']), int(row['request_id']))
        item = lifecycle.get(key)
        if item is None:
            continue
        if row['event'] == 'TX':
            item['rlc_tx_count'] += 1
        elif row['event'] == 'RX':
            item['rlc_rx_count'] += 1
        elif row['event'] == 'DROP':
            item['rlc_drop_count'] += 1

    for row in phy_request_records:
        if not int(row.get('matched', 0)):
            continue
        key = (int(row['frame_index']), int(row['request_id']))
        item = lifecycle.get(key)
        if item is None:
            continue
        event = str(row.get('event', '')).upper()
        if event == 'PHY_SCHEDULE':
            item['phy_schedule_count'] += 1
        elif event == 'PSCCH_DECODE_OK':
            item['pscch_ok_count'] += 1
        elif event == 'PSCCH_DECODE_FAIL':
            item['pscch_fail_count'] += 1
        elif event == 'PSSCH_DECODE_OK':
            item['pssch_ok_count'] += 1
        elif event == 'PSSCH_DECODE_FAIL':
            item['pssch_fail_count'] += 1
        elif event == 'HARQ_ACK':
            item['harq_ack_count'] += 1
        elif event == 'HARQ_NACK':
            item['harq_nack_count'] += 1
        elif event == 'HARQ_TIMEOUT':
            item['harq_timeout_count'] += 1

    for row in aligned_records:
        if not int(row.get('matched', 0)):
            continue
        key = (int(row['frame_index']), int(row.get('pkt_id') or 0))
        item = lifecycle.get(key)
        if item is not None:
            item['cam_received'] = 1

    for item in lifecycle.values():
        if item['cam_received']:
            item['terminal_state'] = 'application_received'
        elif item['rlc_rx_count'] > 0:
            item['terminal_state'] = 'rlc_rx_only'
        elif item['harq_ack_count'] > 0:
            item['terminal_state'] = 'harq_ack_only'
        elif item['pssch_ok_count'] > 0:
            item['terminal_state'] = 'pssch_ok_only'
        elif item['pssch_fail_count'] > 0:
            item['terminal_state'] = 'pssch_fail'
        elif item['harq_nack_count'] > 0:
            item['terminal_state'] = 'harq_nack'
        elif item['harq_timeout_count'] > 0:
            item['terminal_state'] = 'harq_timeout'
        elif item['pscch_fail_count'] > 0:
            item['terminal_state'] = 'pscch_fail'
        elif item['phy_schedule_count'] > 0:
            item['terminal_state'] = 'phy_scheduled_only'
        elif item['rlc_tx_count'] > 0:
            item['terminal_state'] = 'rlc_tx_no_rx'

    return sorted(lifecycle.values(),
                  key=lambda item: (int(item['frame_index']),
                                    int(item['request_id'])))


def aggregate_request_lifecycle(lifecycle_rows):
    total = len(lifecycle_rows)
    terminal_counts = Counter(row['terminal_state'] for row in lifecycle_rows)
    return [{
        'planned_requests': total,
        'requests_with_rlc_tx': sum(1 for row in lifecycle_rows
                                    if row['rlc_tx_count'] > 0),
        'requests_with_phy_schedule': sum(1 for row in lifecycle_rows
                                          if row['phy_schedule_count'] > 0),
        'requests_with_pssch_ok': sum(1 for row in lifecycle_rows
                                      if row['pssch_ok_count'] > 0),
        'requests_with_pssch_fail': sum(1 for row in lifecycle_rows
                                        if row['pssch_fail_count'] > 0),
        'requests_with_harq_ack': sum(1 for row in lifecycle_rows
                                      if row['harq_ack_count'] > 0),
        'requests_with_harq_nack': sum(1 for row in lifecycle_rows
                                       if row['harq_nack_count'] > 0),
        'requests_with_rlc_rx': sum(1 for row in lifecycle_rows
                                    if row['rlc_rx_count'] > 0),
        'requests_with_cam_received': sum(1 for row in lifecycle_rows
                                          if row['cam_received']),
        'terminal_application_received': terminal_counts['application_received'],
        'terminal_rlc_rx_only': terminal_counts['rlc_rx_only'],
        'terminal_pssch_fail': terminal_counts['pssch_fail'],
        'terminal_harq_nack': terminal_counts['harq_nack'],
        'terminal_harq_timeout': terminal_counts['harq_timeout'],
        'terminal_pscch_fail': terminal_counts['pscch_fail'],
        'terminal_rlc_tx_no_rx': terminal_counts['rlc_tx_no_rx'],
        'terminal_planned_only': terminal_counts['planned_only'],
        'note': 'PHY/HARQ counts are populated only when NS3 emits NRSL_PHY_EVENT or NRSL_HARQ_EVENT request-level logs',
    }]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plan_rows, timestamps = read_upload_plan(
        args.upload_plan,
        args.rsu_node_id,
        args.max_frames,
        args.start_index,
        args.frame_step)
    cam_records = parse_cam_received(args.ns3_stdout, args.frame_interval_ms)
    phy_records = parse_phy_decodes(args.ns3_stdout)
    rlc_records = parse_rlc_events(args.ns3_stdout)
    phy_request_records = parse_phy_harq_request_events(args.ns3_stdout)
    aligned_records = align_records(cam_records, plan_rows, timestamps)
    enriched_rlc_records = enrich_rlc_records(rlc_records, plan_rows, timestamps)
    enriched_phy_request_records = enrich_request_events(
        phy_request_records,
        plan_rows,
        timestamps)
    frame_summary, type_summary, summary = aggregate(
        plan_rows,
        aligned_records,
        timestamps)
    phy_summary = aggregate_phy(phy_records)
    rlc_summary, rlc_by_request = aggregate_rlc(
        enriched_rlc_records,
        plan_rows)
    request_lifecycle = build_request_lifecycle(
        plan_rows,
        aligned_records,
        enriched_rlc_records,
        enriched_phy_request_records)
    lifecycle_summary = aggregate_request_lifecycle(request_lifecycle)

    write_csv(os.path.join(args.output_dir, 'cam_received_records.csv'),
              aligned_records)
    write_csv(os.path.join(args.output_dir, 'delivery_by_frame.csv'),
              frame_summary)
    write_csv(os.path.join(args.output_dir, 'delivery_by_type.csv'),
              type_summary)
    write_csv(os.path.join(args.output_dir, 'delivery_summary.csv'), summary)
    write_csv(os.path.join(args.output_dir, 'phy_decode_events.csv'),
              phy_records)
    write_csv(os.path.join(args.output_dir, 'phy_decode_summary.csv'),
              phy_summary)
    write_csv(os.path.join(args.output_dir, 'rlc_events.csv'),
              enriched_rlc_records)
    write_csv(os.path.join(args.output_dir, 'rlc_summary.csv'),
              rlc_summary)
    write_csv(os.path.join(args.output_dir, 'rlc_by_request.csv'),
              rlc_by_request)
    write_csv(os.path.join(args.output_dir, 'phy_harq_request_events.csv'),
              enriched_phy_request_records)
    write_csv(os.path.join(args.output_dir, 'request_lifecycle.csv'),
              request_lifecycle)
    write_csv(os.path.join(args.output_dir, 'request_lifecycle_summary.csv'),
              lifecycle_summary)

    row = summary[0]
    phy_failures = sum(1 for item in phy_records if item['status'] == 'FAIL')
    rlc_row = rlc_summary[0]
    print('planned_requests=%s observed_cam_received=%s '
          'bridge_observed_delivery_ratio=%.6f avg_delay_ms=%.3f '
          'p95_delay_ms=%.3f max_delay_ms=%.3f phy_decode_events=%s '
          'phy_decode_failures=%s phy_harq_request_events=%s '
          'rlc_tx_events=%s rlc_rx_events=%s' % (
              row['planned_requests'],
              row['observed_cam_received'],
              row['bridge_observed_delivery_ratio'],
              row['avg_delay_ms'],
              row['p95_delay_ms'],
              row['max_delay_ms'],
              len(phy_records),
              phy_failures,
              len(enriched_phy_request_records),
              rlc_row['rlc_tx_events'],
              rlc_row['rlc_rx_events']))


if __name__ == '__main__':
    main()
