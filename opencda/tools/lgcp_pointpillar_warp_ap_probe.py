# -*- coding: utf-8 -*-
"""
Evaluate coordinate-warped LGCP PointPillar canvas against reference-frame GT.

This is a smoke probe: it feeds a warped RSU/reference feature canvas through
the PointPillar backbone and detection heads, uses the OpenCOOD postprocessor
to decode boxes, and computes AP against the reference CAV frame GT.
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.lgcp_pointpillar_rsu_head_probe import run_head
from opencda.tools.offline_inference import load_coperception_params
from opencood.utils import eval_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate LGCP warped PointPillar canvas AP.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--warped-root', required=True)
    parser.add_argument('--warped-frame-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-cav-id', default='1')
    parser.add_argument('--fusion-method', default='intermediate_attentive')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--frame-file-column', default='warped_frame_file')
    parser.add_argument('--canvas-key', default='warped_canvas')
    parser.add_argument('--score-threshold', type=float, default=None)
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shape_string(tensor):
    if tensor is None:
        return ''
    return 'x'.join(str(int(item)) for item in tensor.shape)


def build_reference_batch(manager, dataset, scenario_id, timestamp,
                          reference_cav_id):
    frame = dataset.load_frame(
        scenario_id,
        timestamp,
        ego_cav_id=reference_cav_id,
        cav_ids=[reference_cav_id])
    ego = next(cav for cav in frame.values() if cav['ego'])
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        ego['params']['lidar_pose'])
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    return manager.to_device(output_dict)


def decode_predictions(manager, batch_data, psm, rm, score_threshold):
    post_processor = manager.opencood_dataset.post_processor
    original_threshold = post_processor.params['target_args'][
        'score_threshold']
    if score_threshold is not None:
        post_processor.params['target_args']['score_threshold'] = (
            score_threshold)
    try:
        output_dict = {'ego': {'psm': psm, 'rm': rm}}
        pred_box_tensor, pred_score, gt_box_tensor = (
            manager.opencood_dataset.post_process(batch_data, output_dict))
        return pred_box_tensor, pred_score, gt_box_tensor
    finally:
        post_processor.params['target_args']['score_threshold'] = (
            original_threshold)


def empty_result_stat():
    return {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }


def ap_or_zero(result_stat, iou):
    if result_stat[iou]['gt'] == 0 or not result_stat[iou]['tp']:
        return 0.0
    ap, _mrec, _mpre = eval_utils.calculate_ap(result_stat, iou)
    return float(ap)


def to_float(value):
    return '%.6f' % float(value)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    dataset = OPV2VFrameDataset(args.dataset_root)

    frame_rows = []
    result_stat = empty_result_stat()
    for row in read_csv(args.warped_frame_manifest):
        frame_path = os.path.join(args.warped_root, row[args.frame_file_column])
        canvas = np.load(frame_path)[args.canvas_key]
        spatial_features_2d, psm, rm = run_head(manager, canvas)
        batch_data = build_reference_batch(
            manager,
            dataset,
            args.scenario_id,
            row['timestamp'],
            args.reference_cav_id)
        pred_box_tensor, pred_score, gt_box_tensor = decode_predictions(
            manager,
            batch_data,
            psm,
            rm,
            args.score_threshold)
        for iou in (0.3, 0.5, 0.7):
            eval_utils.calculate_tp_fp(
                pred_box_tensor,
                pred_score,
                gt_box_tensor,
                result_stat,
                iou)
        pred_count = 0 if pred_box_tensor is None else int(
            pred_box_tensor.shape[0])
        score_max = 0.0 if pred_score is None or pred_score.numel() == 0 else (
            float(torch.max(pred_score).detach().cpu()))
        frame_rows.append(OrderedDict({
            'timestamp': row['timestamp'],
            'reference_cav_id': args.reference_cav_id,
            'source_frame_file': row[args.frame_file_column],
            'canvas_key': args.canvas_key,
            'input_canvas_shape': shape_string(canvas),
            'backbone_spatial_features_2d_shape':
                shape_string(spatial_features_2d),
            'psm_shape': shape_string(psm),
            'rm_shape': shape_string(rm),
            'pred_boxes': pred_count,
            'gt_boxes': int(gt_box_tensor.shape[0]),
            'score_max': to_float(score_max),
        }))

    ap_rows = [OrderedDict({
        'frames': len(frame_rows),
        'ap_03': to_float(ap_or_zero(result_stat, 0.3)),
        'ap_05': to_float(ap_or_zero(result_stat, 0.5)),
        'ap_07': to_float(ap_or_zero(result_stat, 0.7)),
        'gt_total': result_stat[0.5]['gt'],
        'pred_samples': len(result_stat[0.5]['tp']),
    })]
    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'warp_ap_frame_summary.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    write_csv(os.path.join(args.output_dir, 'warp_ap_summary.csv'),
              list(ap_rows[0].keys()),
              ap_rows)
    config = {
        'dataset_root': os.path.abspath(args.dataset_root),
        'scenario_id': args.scenario_id,
        'warped_root': os.path.abspath(args.warped_root),
        'warped_frame_manifest': os.path.abspath(args.warped_frame_manifest),
        'reference_cav_id': args.reference_cav_id,
        'fusion_method': args.fusion_method,
        'frame_file_column': args.frame_file_column,
        'canvas_key': args.canvas_key,
        'score_threshold': args.score_threshold,
        'note': 'Single-frame warped feature AP smoke; not final paper result.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Coordinate-Warp AP Probe\n\n')
        stream.write('This smoke evaluates warped feature canvas predictions ')
        stream.write('against reference-frame GT. It is not a calibrated final ')
        stream.write('model-level AP result.\n\n')
        stream.write('- frames: `%s`\n' % ap_rows[0]['frames'])
        stream.write('- AP@0.5: `%s`\n' % ap_rows[0]['ap_05'])
        stream.write('- AP@0.7: `%s`\n' % ap_rows[0]['ap_07'])

    print('Wrote LGCP coordinate-warp AP probe to %s' % args.output_dir)
    print('frames=%s AP@0.5=%s AP@0.7=%s pred_samples=%s gt=%s' % (
        ap_rows[0]['frames'],
        ap_rows[0]['ap_05'],
        ap_rows[0]['ap_07'],
        ap_rows[0]['pred_samples'],
        ap_rows[0]['gt_total']))


if __name__ == '__main__':
    main()
