# -*- coding: utf-8 -*-
"""
Profile detector-side compute for SGCP experiment traces.

This utility serves two related purposes:

1. Calibrate one OpenCOOD detector forward on a real dumped frame with module
   forward hooks. The hook-based counter currently covers Conv2d,
   ConvTranspose2d and Linear layers and reports multiply-add as two FLOPs.
2. Convert SGCP/offline-inference trace CSVs into per-method compute profiles:
   detector calls per frame, input points, predicted boxes, payload, and
   calibrated GFLOPs per frame.

The trace-derived rows are intentionally protocol-aware. For example, pure late
fusion has one detector call per CAV, while SGCP with inter-cluster NMS has one
detector call per cluster head. This makes the computation side of the
AP-Mbps-compute tradeoff explicit without changing perception evaluation.
"""

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build compute/GFLOPs profiles from SGCP trace CSVs.')
    parser.add_argument(
        '--method',
        action='append',
        default=[],
        metavar='LABEL=TRACE.csv',
        help='Trace CSV to summarize. Can be repeated.')
    parser.add_argument(
        '--metrics-csv',
        action='append',
        default=[],
        help='Optional experiment table CSVs used to attach AP/Mbps columns.')
    parser.add_argument(
        '--calibration-json',
        default=None,
        help='Calibration JSON produced by --calibrate-forward.')
    parser.add_argument(
        '--dense-calibration-json',
        default=None,
        help='Optional second calibration JSON with a denser point cloud. '
             'When provided together with --calibration-json, the tool '
             'estimates per-call GFLOPs as fixed Conv/Deconv cost plus a '
             'linear point-count model for VFE Linear FLOPs.')
    parser.add_argument(
        '--calibrated-gflops-per-forward',
        type=float,
        default=None,
        help='Detector GFLOPs per forward. Overrides --calibration-json.')
    parser.add_argument(
        '--frame-interval-s',
        type=float,
        default=0.1,
        help='Perception cycle duration. Defaults to 0.1 s.')
    parser.add_argument(
        '--output-csv',
        default=None,
        help='Output compute profile CSV.')
    parser.add_argument(
        '--summary-md',
        default=None,
        help='Output Markdown summary.')

    parser.add_argument(
        '--calibrate-forward',
        action='store_true',
        help='Run one real OpenCOOD forward and estimate FLOPs with hooks.')
    parser.add_argument('--dataset-root', default=None)
    parser.add_argument('--scenario-id', default=None)
    parser.add_argument('--timestamp', default=None)
    parser.add_argument('--ego-cav-id', default=None)
    parser.add_argument('--cav-count', type=int, default=None)
    parser.add_argument('--cav-ids', default=None)
    parser.add_argument('--fusion-method', default='early')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument(
        '--calibration-output',
        default=None,
        help='Path for calibration JSON.')
    return parser.parse_args()


def parse_label_path(value):
    if '=' not in value:
        raise ValueError('--method must use LABEL=TRACE.csv: %s' % value)
    label, path = value.split('=', 1)
    label = label.strip()
    path = path.strip().strip('"')
    if not label or not path:
        raise ValueError('--method must use LABEL=TRACE.csv: %s' % value)
    return label, path


def read_json_cell(value, default):
    if value is None or value == '':
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def parse_semicolon_ids(value):
    if not value:
        return []
    output = []
    for item in str(value).split(';'):
        item = item.strip()
        if not item:
            continue
        try:
            output.append(int(item))
        except ValueError:
            output.append(item)
    return output


def to_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    if value is None or value == '':
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * pct / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(values[lower])
    return float(values[lower] * (upper - rank) + values[upper] * (rank - lower))


def load_metrics(paths):
    metrics = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, newline='') as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                label = row.get('label') or row.get('method') or row.get('name')
                if not label:
                    continue
                metrics[label] = row
    return metrics


def build_input_adjusted_model(calibration, dense_calibration):
    if not calibration or not dense_calibration:
        return None
    try:
        point_a = float(calibration['input_points'])
        point_b = float(dense_calibration['input_points'])
        linear_a = float(
            calibration.get('flops_by_module_type', {}).get('Linear', 0.0))
        linear_b = float(
            dense_calibration.get(
                'flops_by_module_type', {}).get('Linear', 0.0))
        fixed_a = (
            float(calibration.get(
                'flops_by_module_type', {}).get('Conv2d', 0.0)) +
            float(calibration.get(
                'flops_by_module_type', {}).get('ConvTranspose2d', 0.0)))
    except (KeyError, TypeError, ValueError):
        return None
    if point_a == point_b:
        return None
    slope = (linear_b - linear_a) / (point_b - point_a)
    intercept = linear_a - slope * point_a
    return {
        'fixed_flops': fixed_a,
        'linear_intercept_flops': intercept,
        'linear_flops_per_point': slope,
    }


def adjusted_gflops_for_points(point_count, model):
    linear_flops = (
        model['linear_intercept_flops'] +
        model['linear_flops_per_point'] * float(point_count))
    return (model['fixed_flops'] + max(0.0, linear_flops)) / 1e9


def summarize_trace(label, trace_path, metric_row, calibrated_gflops,
                    input_adjusted_model,
                    frame_interval_s):
    with open(trace_path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    active_rows = [row for row in rows if not row.get('skipped')]
    timestamps = sorted({row.get('timestamp', '') for row in rows})
    frame_count = len(timestamps) if timestamps else 0
    detector_calls_total = len(active_rows)
    calls_per_frame = (
        detector_calls_total / float(frame_count) if frame_count else 0.0)

    rows_by_frame = defaultdict(list)
    for row in active_rows:
        rows_by_frame[row.get('timestamp', '')].append(row)

    source_counts = []
    uploaded_source_counts = []
    selected_grids = []
    input_points = []
    pred_boxes = []
    comm_bytes = []
    comm_time_ms = []
    for row in active_rows:
        source_ids = parse_semicolon_ids(row.get('source_cav_ids', ''))
        uploaded_ids = parse_semicolon_ids(row.get('uploaded_source_ids', ''))
        point_counts = read_json_cell(row.get('point_counts_json'), {})
        selected_grid_counts = read_json_cell(
            row.get('selected_grid_counts_json'), {})
        source_counts.append(len(source_ids))
        uploaded_source_counts.append(len(uploaded_ids))
        selected_grids.append(
            sum(to_int(value) for value in selected_grid_counts.values()))
        input_points.append(sum(to_int(value) for value in point_counts.values()))
        pred_boxes.append(to_int(row.get('pred_boxes'), 0))
        comm_bytes.append(to_float(row.get('communication_bytes'), 0.0))
        comm_time_ms.append(to_float(row.get('frame_comm_time_ms'), 0.0))

    calls_per_frame_values = [len(rows_by_frame[timestamp])
                              for timestamp in timestamps]
    pred_boxes_per_frame_values = [
        sum(to_int(row.get('pred_boxes'), 0)
            for row in rows_by_frame[timestamp])
        for timestamp in timestamps
    ]
    input_points_per_frame_values = [
        sum(
            sum(to_int(value) for value in read_json_cell(
                row.get('point_counts_json'), {}).values())
            for row in rows_by_frame[timestamp])
        for timestamp in timestamps
    ]
    adjusted_gflops_per_frame_values = []
    if input_adjusted_model is not None:
        for timestamp in timestamps:
            frame_gflops = 0.0
            for row in rows_by_frame[timestamp]:
                point_counts = read_json_cell(row.get('point_counts_json'), {})
                row_points = sum(
                    to_int(value) for value in point_counts.values())
                frame_gflops += adjusted_gflops_for_points(
                    row_points,
                    input_adjusted_model)
            adjusted_gflops_per_frame_values.append(frame_gflops)

    raw_lidar_mbps = to_float(
        metric_row.get('raw_lidar_mbps') if metric_row else None,
        default=(
            sum(comm_bytes) * 8.0 /
            (max(frame_count * frame_interval_s, 1e-9) * 1e6)))
    total_mbps = to_float(
        metric_row.get('total_mbps') if metric_row else None,
        default=raw_lidar_mbps)

    gflops_per_frame = ''
    gflops_per_second = ''
    if calibrated_gflops is not None:
        gflops_per_frame = calls_per_frame * calibrated_gflops
        gflops_per_second = gflops_per_frame / frame_interval_s

    return {
        'label': label,
        'trace_path': trace_path,
        'ap_03': '' if not metric_row else metric_row.get('ap_03', ''),
        'ap_05': '' if not metric_row else metric_row.get('ap_05', ''),
        'ap_07': '' if not metric_row else metric_row.get('ap_07', ''),
        'raw_lidar_mbps': '%.6f' % raw_lidar_mbps,
        'total_mbps': '%.6f' % total_mbps,
        'frame_count': frame_count,
        'trace_rows': len(rows),
        'detector_calls_total': detector_calls_total,
        'detector_calls_per_frame': '%.3f' % calls_per_frame,
        'detector_calls_per_frame_p95': '%.3f' % percentile(
            calls_per_frame_values, 95),
        'mean_source_cavs_per_call': '%.3f' % mean(source_counts),
        'mean_uploaded_cavs_per_call': '%.3f' % mean(uploaded_source_counts),
        'mean_selected_grids_per_call': '%.3f' % mean(selected_grids),
        'mean_input_points_per_call': '%.1f' % mean(input_points),
        'mean_input_points_per_frame': '%.1f' % mean(
            input_points_per_frame_values),
        'mean_pred_boxes_per_call': '%.3f' % mean(pred_boxes),
        'mean_pred_boxes_per_frame': '%.3f' % mean(
            pred_boxes_per_frame_values),
        'mean_raw_bytes_per_call': '%.1f' % mean(comm_bytes),
        'mean_frame_comm_time_ms_per_call_row': '%.3f' % mean(comm_time_ms),
        'calibrated_gflops_per_forward': (
            '' if calibrated_gflops is None else '%.6f' % calibrated_gflops),
        'profiled_detector_gflops_per_frame': (
            '' if gflops_per_frame == '' else '%.6f' % gflops_per_frame),
        'profiled_detector_gflops_per_second_at_10hz': (
            '' if gflops_per_second == '' else '%.6f' % gflops_per_second),
        'input_adjusted_detector_gflops_per_frame': (
            '' if not adjusted_gflops_per_frame_values else
            '%.6f' % mean(adjusted_gflops_per_frame_values)),
    }


class FlopCounter(object):
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.flops = 0
        self.by_type = defaultdict(float)

    def __enter__(self):
        import torch.nn as nn

        def conv_hook(module, inputs, output):
            if output is None:
                return
            output_shape = tuple(output.shape)
            if len(output_shape) < 4:
                return
            batch, out_channels, out_h, out_w = output_shape[:4]
            in_channels = module.in_channels
            groups = module.groups
            kernel_h, kernel_w = module.kernel_size
            macs = (
                batch * out_channels * out_h * out_w *
                (in_channels / float(groups)) * kernel_h * kernel_w)
            flops = 2.0 * macs
            if module.bias is not None:
                flops += batch * out_channels * out_h * out_w
            self.flops += flops
            self.by_type[type(module).__name__] += flops

        def linear_hook(module, inputs, output):
            if output is None:
                return
            if not hasattr(output, 'shape') or len(output.shape) == 0:
                return
            output_elements = int(output.numel())
            output_vectors = output_elements / float(module.out_features)
            flops = 2.0 * output_vectors * module.in_features * (
                module.out_features)
            if module.bias is not None:
                flops += output_elements
            self.flops += flops
            self.by_type[type(module).__name__] += flops

        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(linear_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def calibrate_forward(args):
    if not args.dataset_root:
        raise ValueError('--dataset-root is required with --calibrate-forward')

    from opencda.core.common.offline_dataset import OPV2VFrameDataset
    from opencda.core.ml_libs.opencood_manager import OpenCOODManager
    from opencda.tools.offline_inference import (
        load_coperception_params,
        run_opencood_inference,
        select_cav_ids,
    )

    dataset = OPV2VFrameDataset(args.dataset_root)
    scenario_id = args.scenario_id or next(iter(dataset.scenarios.keys()))
    timestamp = args.timestamp or dataset.scenarios[scenario_id]['timestamps'][0]
    params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(params)
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=args.ego_cav_id,
        cav_ids=select_cav_ids(
            dataset,
            scenario_id,
            ego_cav_id=args.ego_cav_id,
            cav_count=args.cav_count,
            cav_ids=args.cav_ids))
    ego = next(cav for cav in frame.values() if cav['ego'])
    point_count = int(sum(cav['lidar_np'].shape[0] for cav in frame.values()))
    with FlopCounter(manager.model) as counter:
        run_opencood_inference(
            manager,
            frame,
            ego['params']['lidar_pose'],
            debug_output=False)

    result = {
        'dataset_root': args.dataset_root,
        'scenario_id': scenario_id,
        'timestamp': timestamp,
        'ego_cav_id': args.ego_cav_id,
        'cav_count': len(frame),
        'fusion_method': args.fusion_method,
        'coperception_yaml': args.coperception_yaml,
        'input_points': point_count,
        'flop_policy': 'Conv2d/ConvTranspose2d/Linear hooks; multiply-add=2 FLOPs',
        'profiled_flops_per_forward': counter.flops,
        'profiled_gflops_per_forward': counter.flops / 1e9,
        'flops_by_module_type': {
            key: value for key, value in sorted(counter.by_type.items())
        },
    }
    if args.calibration_output:
        output_dir = os.path.dirname(os.path.abspath(args.calibration_output))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(args.calibration_output, 'w') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
    return result


def write_csv(path, rows):
    if not path or not rows:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, rows, calibration):
    if not path:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    columns = [
        'label',
        'ap_03',
        'ap_05',
        'ap_07',
        'total_mbps',
        'detector_calls_per_frame',
        'mean_source_cavs_per_call',
        'mean_input_points_per_frame',
        'mean_pred_boxes_per_frame',
        'profiled_detector_gflops_per_frame',
        'input_adjusted_detector_gflops_per_frame',
    ]
    with open(path, 'w') as stream:
        stream.write('# SGCP Profiled GFLOPs Summary\n\n')
        stream.write(
            'This file summarizes detector-side compute from offline trace '
            'CSVs. GFLOPs are calibrated from one real OpenCOOD forward when '
            'a calibration JSON is provided; otherwise the table still reports '
            'forward-equivalent compute and input-size diagnostics.\n\n')
        if calibration:
            stream.write('## Calibration\n\n')
            stream.write('- fusion_method: `%s`\n' % calibration.get(
                'fusion_method', ''))
            stream.write('- scenario/timestamp: `%s/%s`\n' % (
                calibration.get('scenario_id', ''),
                calibration.get('timestamp', '')))
            stream.write('- CAVs in calibration forward: `%s`\n' %
                         calibration.get('cav_count', ''))
            stream.write('- input points: `%s`\n' %
                         calibration.get('input_points', ''))
            stream.write('- profiled detector GFLOPs/forward: `%.6f`\n' %
                         float(calibration.get(
                             'profiled_gflops_per_forward', 0.0)))
            stream.write('- FLOP policy: %s\n\n' %
                         calibration.get('flop_policy', ''))
        stream.write('## Compute Table\n\n')
        stream.write('| ' + ' | '.join(columns) + ' |\n')
        stream.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in rows:
            stream.write('| ' + ' | '.join(
                str(row.get(column, '')) for column in columns) + ' |\n')
        stream.write('\n## Notes\n\n')
        stream.write(
            '- `detector_calls_per_frame` is the number of OpenCOOD detector '
            'forwards represented by the trace in one 100 ms perception '
            'cycle.\n')
        stream.write(
            '- Pure late/global box baselines can have high AP because many '
            'CAVs run local detection; SGCP reduces this by evaluating only '
            'cluster heads while still ingesting selected member point clouds.\n')
        stream.write(
            '- `mean_input_points_per_frame` is trace-derived and therefore '
            'captures fused point-cloud size, not just the number of receivers.\n')
        stream.write(
            '- `input_adjusted_detector_gflops_per_frame`, when present, uses '
            'singleton and dense/full calibrations to model point-dependent VFE '
            'Linear FLOPs on top of fixed BEV Conv/Deconv FLOPs.\n')
    return path


def load_calibration(args):
    if args.calibrated_gflops_per_forward is not None:
        return {
            'profiled_gflops_per_forward':
                args.calibrated_gflops_per_forward,
            'flop_policy': 'user-provided calibrated GFLOPs/forward',
        }
    if not args.calibration_json:
        return None
    with open(args.calibration_json) as stream:
        return json.load(stream)


def load_dense_calibration(args):
    if not args.dense_calibration_json:
        return None
    with open(args.dense_calibration_json) as stream:
        return json.load(stream)


def main():
    args = parse_args()
    calibration = None
    if args.calibrate_forward:
        calibration = calibrate_forward(args)
        print(json.dumps(calibration, indent=2, sort_keys=True))
    loaded_calibration = load_calibration(args)
    if loaded_calibration is not None:
        calibration = loaded_calibration
    dense_calibration = load_dense_calibration(args)
    input_adjusted_model = build_input_adjusted_model(
        calibration,
        dense_calibration)
    calibrated_gflops = None
    if calibration is not None and calibration.get(
            'profiled_gflops_per_forward') is not None:
        calibrated_gflops = float(
            calibration['profiled_gflops_per_forward'])

    metrics = load_metrics(args.metrics_csv)
    rows = []
    for method in args.method:
        label, trace_path = parse_label_path(method)
        metric_row = metrics.get(label, {})
        rows.append(summarize_trace(
            label,
            trace_path,
            metric_row,
            calibrated_gflops,
            input_adjusted_model,
            args.frame_interval_s))
    if rows:
        write_csv(args.output_csv, rows)
        write_markdown(args.summary_md, rows, calibration)
        print('compute_profile_rows=%d' % len(rows))
        if args.output_csv:
            print('output_csv=%s' % args.output_csv)
        if args.summary_md:
            print('summary_md=%s' % args.summary_md)


if __name__ == '__main__':
    main()
