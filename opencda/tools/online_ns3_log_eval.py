# -*- coding: utf-8 -*-
"""Summarize online OpenCDA+NS3 upload lifecycle logs.

The online CARLA run currently does not persist the transfer request plan as a
CSV.  This helper therefore evaluates the OpenCDA application-visible upload
state: partial fragment accumulation, eventual completion, and repeated
incomplete polling lines.
"""

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter, defaultdict


INCOMPLETE_PATTERN = re.compile(
    r'cav\s+(?P<src>\d+)\s+data upload to\s+(?P<dst>\d+)\s+'
    r'incomplete \(NS3 mode\)\. Received size:\s+'
    r'(?P<received>\d+)\s+bytes, expected size:\s+'
    r'(?P<expected>\d+)\s+bytes\.')
SUCCESS_PATTERN = re.compile(
    r'cav\s+(?P<src>\d+)\s+has uploaded its data to\s+(?P<dst>\d+)\s+'
    r'via network at\s+(?P<slot>\d+)\.')
CAM_PATTERN = re.compile(r'cam_received')
PSCCH_FAIL_PATTERN = re.compile(r'PSCCH_DECODE_FAIL')
PSSCH_FAIL_PATTERN = re.compile(r'PSSCH_DECODE_FAIL')
OVERLAP_PATTERN = re.compile(r'reason=decoded_overlap')
MANUAL_ADD_PATTERN = re.compile(r'MANUAL_CMD_ADD')
MANUAL_REJECT_PATTERN = re.compile(r'MANUAL_CMD_REJECT')
SYNC_REQUEST_PATTERN = re.compile(r'Received sync_request')
SYNC_ACK_PATTERN = re.compile(r'Sent sync_ack')
CP_EVAL_PATTERN = re.compile(r'CP_EVAL_FRAME\b')
CP_SUBMIT_PATTERN = re.compile(r'CP_SUBMIT_FRAME\b')
CP_WAIT_PATTERN = re.compile(
    r'CP_WAIT_FRAME\s+ego=(?P<ego>\d+)\s+slot=(?P<slot>\S+)\s+'
    r'uploaded=(?P<uploaded>\d+)/(?P<needed>\d+)')
CP_EGO_DID_PATTERN = re.compile(r'CP_EGO_DID_CP\b')
AP_COUNTER_PATTERN = re.compile(r'cp counter:\s*(?P<count>\d+)')
AP_RESULT_PATTERN = re.compile(
    r'Average Precision at IOU 0\.3 is (?P<ap30>[0-9.]+).*'
    r'Average Precision at IOU 0\.5 is (?P<ap50>[0-9.]+).*'
    r'Average Precision at IOU 0\.7 is (?P<ap70>[0-9.]+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate online OpenCDA+NS3 application upload logs.')
    parser.add_argument('--opencda-stdout', required=True,
                        help='Path to OpenCDA stdout log.')
    parser.add_argument('--ns3-stdout', default='',
                        help='Optional NS3 stdout log for coarse counters.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for CSV/JSON summaries.')
    return parser.parse_args()


def new_episode(src, dst, episode_index):
    return {
            'source_id': src,
            'target_id': dst,
            'episode_index': episode_index,
            'incomplete_lines': 0,
            'first_incomplete_line': '',
            'last_incomplete_line': '',
            'first_success_line': '',
            'success_lines': 0,
            'success_slot_min': '',
            'success_slot_max': '',
            'expected_bytes_min': '',
            'expected_bytes_max': '',
            'received_bytes_max': 0,
            'missing_bytes_at_max': '',
            'terminal_state': 'unknown',
        }


def update_expected(row, expected):
    if row['expected_bytes_min'] == '':
        row['expected_bytes_min'] = expected
        row['expected_bytes_max'] = expected
        return
    row['expected_bytes_min'] = min(int(row['expected_bytes_min']), expected)
    row['expected_bytes_max'] = max(int(row['expected_bytes_max']), expected)


def parse_opencda(path):
    episodes = []
    active_by_link = {}
    episode_counts = defaultdict(int)
    total_incomplete_lines = 0
    total_success_lines = 0

    def start_episode(src, dst):
        key = (src, dst)
        episode_counts[key] += 1
        row = new_episode(src, dst, episode_counts[key])
        active_by_link[key] = row
        episodes.append(row)
        return row

    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line_no, line in enumerate(stream, start=1):
            match = INCOMPLETE_PATTERN.search(line)
            if match:
                src = int(match.group('src'))
                dst = int(match.group('dst'))
                received = int(match.group('received'))
                expected = int(match.group('expected'))
                key = (src, dst)
                row = active_by_link.get(key)
                if row is None or row['terminal_state'] == 'application_complete':
                    row = start_episode(src, dst)
                row['incomplete_lines'] += 1
                total_incomplete_lines += 1
                if row['first_incomplete_line'] == '':
                    row['first_incomplete_line'] = line_no
                row['last_incomplete_line'] = line_no
                row['received_bytes_max'] = max(
                    int(row['received_bytes_max']), received)
                update_expected(row, expected)
                continue

            match = SUCCESS_PATTERN.search(line)
            if match:
                src = int(match.group('src'))
                dst = int(match.group('dst'))
                slot = int(match.group('slot'))
                key = (src, dst)
                row = active_by_link.get(key)
                if row is None or row['terminal_state'] == 'application_complete':
                    row = start_episode(src, dst)
                row['success_lines'] += 1
                total_success_lines += 1
                if row['first_success_line'] == '':
                    row['first_success_line'] = line_no
                row['terminal_state'] = 'application_complete'
                if row['success_slot_min'] == '':
                    row['success_slot_min'] = slot
                    row['success_slot_max'] = slot
                else:
                    row['success_slot_min'] = min(
                        int(row['success_slot_min']), slot)
                    row['success_slot_max'] = max(
                        int(row['success_slot_max']), slot)

    for row in episodes:
        expected_max = row['expected_bytes_max']
        if expected_max != '':
            row['missing_bytes_at_max'] = max(
                int(expected_max) - int(row['received_bytes_max']), 0)
        if int(row['success_lines']) > 0:
            row['terminal_state'] = 'application_complete'
        elif int(row['incomplete_lines']) > 0:
            row['terminal_state'] = 'application_partial'
        else:
            row['terminal_state'] = 'unknown'

    rows = list(episodes)
    state_counts = Counter(row['terminal_state'] for row in rows)
    summary = {
        'unique_links_observed': len(set(
            (row['source_id'], row['target_id']) for row in rows)),
        'upload_episodes_observed': len(rows),
        'application_complete_episodes': state_counts['application_complete'],
        'application_partial_episodes': state_counts['application_partial'],
        'incomplete_log_lines': total_incomplete_lines,
        'success_log_lines': total_success_lines,
        'duplicate_incomplete_lines': total_incomplete_lines -
                                      state_counts['application_partial'],
    }
    return rows, summary


def parse_cp_and_comm(path):
    summary = {
        'cp_eval_frames': 0,
        'cp_submit_frames': 0,
        'cp_wait_frames': 0,
        'cp_ego_did_cp_lines': 0,
        'cp_counter': '',
        'ap_30': '',
        'ap_50': '',
        'ap_70': '',
        'comm_total_volume_bytes': '',
        'comm_try_volume_bytes': '',
        'comm_duration_s': '',
        'comm_total_payload_mbps': '',
        'comm_try_payload_mbps': '',
        'comm_total_slots': '',
    }
    wait_rows = []
    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line_no, line in enumerate(stream, start=1):
            if CP_EVAL_PATTERN.search(line):
                summary['cp_eval_frames'] += 1
            if CP_SUBMIT_PATTERN.search(line):
                summary['cp_submit_frames'] += 1
            if CP_EGO_DID_PATTERN.search(line):
                summary['cp_ego_did_cp_lines'] += 1
            match = CP_WAIT_PATTERN.search(line)
            if match:
                summary['cp_wait_frames'] += 1
                wait_rows.append({
                    'line': line_no,
                    'ego': int(match.group('ego')),
                    'slot': match.group('slot'),
                    'uploaded': int(match.group('uploaded')),
                    'needed': int(match.group('needed')),
                })
            match = AP_COUNTER_PATTERN.search(line)
            if match:
                summary['cp_counter'] = int(match.group('count'))
            match = AP_RESULT_PATTERN.search(line)
            if match:
                summary['ap_30'] = float(match.group('ap30'))
                summary['ap_50'] = float(match.group('ap50'))
                summary['ap_70'] = float(match.group('ap70'))
            if "'traffic_distribution'" in line and "'historical'" in line:
                payload = line[line.find("{"):].strip()
                try:
                    report = ast.literal_eval(payload.replace('nan', 'None'))
                except (SyntaxError, ValueError):
                    continue
                traffic = report.get('traffic_distribution', {})
                hist = report.get('historical', {})
                summary['comm_total_volume_bytes'] = float(
                    traffic.get('total_vol(Bytes)',
                                hist.get('total_volume_bytes', 0.0)))
                summary['comm_try_volume_bytes'] = float(
                    traffic.get('try_volume', 0.0))
                summary['comm_duration_s'] = float(
                    traffic.get('duration_s', hist.get('duration_s', 0.0)))
                summary['comm_total_payload_mbps'] = float(
                    traffic.get('total_payload_mbps',
                                hist.get('total_payload_mbps', 0.0)))
                summary['comm_try_payload_mbps'] = float(
                    traffic.get('try_payload_mbps',
                                hist.get('try_payload_mbps', 0.0)))
                summary['comm_total_slots'] = int(hist.get('total_slots', 0))

    wait_by_ego = Counter(row['ego'] for row in wait_rows)
    summary['cp_wait_by_ego'] = dict(sorted(wait_by_ego.items()))
    if (summary['comm_total_slots'] and
            (summary['comm_duration_s'] == '' or summary['comm_duration_s'] == 0)):
        # Compatibility with logs produced before duration_s was added.
        duration_s = summary['comm_total_slots'] * 0.1
        summary['comm_duration_s'] = duration_s
        total = float(summary['comm_total_volume_bytes'] or 0.0)
        tried = float(summary['comm_try_volume_bytes'] or 0.0)
        summary['comm_total_payload_mbps'] = (
            total * 8.0 / duration_s / 1e6 if duration_s > 0 else 0.0)
        summary['comm_try_payload_mbps'] = (
            tried * 8.0 / duration_s / 1e6 if duration_s > 0 else 0.0)
    return wait_rows, summary


def count_pattern(path, pattern):
    if not path:
        return 0
    count = 0
    with open(path, encoding='utf-8', errors='ignore') as stream:
        for line in stream:
            if pattern.search(line):
                count += 1
    return count


def parse_ns3_counts(path):
    if not path:
        return {}
    return {
        'ns3_sync_request': count_pattern(path, SYNC_REQUEST_PATTERN),
        'ns3_sync_ack': count_pattern(path, SYNC_ACK_PATTERN),
        'manual_cmd_add': count_pattern(path, MANUAL_ADD_PATTERN),
        'manual_cmd_reject': count_pattern(path, MANUAL_REJECT_PATTERN),
        'cam_received_lines': count_pattern(path, CAM_PATTERN),
        'pscch_decode_fail': count_pattern(path, PSCCH_FAIL_PATTERN),
        'pssch_decode_fail': count_pattern(path, PSSCH_FAIL_PATTERN),
        'decoded_overlap_fail': count_pattern(path, OVERLAP_PATTERN),
    }


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
    lifecycle_rows, summary = parse_opencda(args.opencda_stdout)
    wait_rows, cp_comm_summary = parse_cp_and_comm(args.opencda_stdout)
    summary.update(cp_comm_summary)
    summary.update(parse_ns3_counts(args.ns3_stdout))
    write_csv(os.path.join(args.output_dir, 'online_upload_lifecycle.csv'),
              lifecycle_rows)
    write_csv(os.path.join(args.output_dir, 'online_cp_wait_rows.csv'),
              wait_rows)
    with open(os.path.join(args.output_dir, 'online_upload_summary.json'), 'w') \
            as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
