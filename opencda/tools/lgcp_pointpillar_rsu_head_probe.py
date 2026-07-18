# -*- coding: utf-8 -*-
"""
Probe PointPillar detection heads on assembled LGCP RSU feature canvases.

The input is produced by lgcp_pointpillar_rsu_feature_assembly.py. This tool
feeds the assembled scatter canvas through the loaded OpenCOOD backbone and
classification/regression heads, then optionally calls the voxel postprocessor.
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Probe detection heads on LGCP RSU feature canvas.')
    parser.add_argument('--rsu-root', required=True)
    parser.add_argument('--rsu-frame-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_attentive')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--score-threshold', type=float, default=None)
    parser.add_argument('--frame-file-column', default='rsu_frame_file')
    parser.add_argument('--canvas-key', default='rsu_canvas')
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
    return 'x'.join(str(int(item)) for item in tensor.shape)


def to_float(value):
    return '%.6f' % float(value)


def run_head(manager, canvas):
    spatial_features = torch.from_numpy(canvas.astype(np.float32))
    spatial_features = spatial_features.to(manager.device)
    record_len = torch.tensor([spatial_features.shape[0]],
                              dtype=torch.int64,
                              device=manager.device)
    batch_dict = {
        'spatial_features': spatial_features,
        'record_len': record_len,
    }
    with torch.no_grad():
        batch_dict = manager.model.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        psm = manager.model.cls_head(spatial_features_2d)
        rm = manager.model.reg_head(spatial_features_2d)
    return spatial_features_2d, psm, rm


def postprocess(manager, psm, rm, score_threshold):
    post_processor = manager.opencood_dataset.post_processor
    original_threshold = post_processor.params['target_args'][
        'score_threshold']
    if score_threshold is not None:
        post_processor.params['target_args']['score_threshold'] = (
            score_threshold)
    try:
        anchor_box = post_processor.generate_anchor_box()
        device = psm.device
        data_dict = {
            'ego': {
                'anchor_box': torch.from_numpy(anchor_box).to(device),
                'transformation_matrix': torch.eye(4, device=device),
            }
        }
        output_dict = {
            'ego': {
                'psm': psm,
                'rm': rm,
            }
        }
        pred_box_tensor, pred_score = post_processor.post_process(
            data_dict,
            output_dict)
        if pred_box_tensor is None:
            return 0, 0, 0.0, ''
        score_np = pred_score.detach().cpu().numpy()
        return (
            int(pred_box_tensor.shape[0]),
            int(score_np.size),
            float(score_np.max()) if score_np.size else 0.0,
            ';'.join('%.6f' % item for item in score_np[:10]))
    finally:
        post_processor.params['target_args']['score_threshold'] = (
            original_threshold)


def top_anchor_rows(timestamp, psm, top_k):
    prob = torch.sigmoid(psm.detach().cpu()).numpy()[0]
    flat = prob.reshape(-1)
    if flat.size == 0:
        return []
    top_k = min(top_k, flat.size)
    indexes = np.argpartition(-flat, top_k - 1)[:top_k]
    indexes = indexes[np.argsort(-flat[indexes])]
    channels, height, width = prob.shape
    rows = []
    for rank, index in enumerate(indexes, start=1):
        channel, y, x = np.unravel_index(index, (channels, height, width))
        rows.append(OrderedDict({
            'timestamp': timestamp,
            'rank': rank,
            'anchor_channel': int(channel),
            'feature_x': int(x),
            'feature_y': int(y),
            'score': to_float(flat[index]),
        }))
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)

    rows = read_csv(args.rsu_frame_manifest)
    head_rows = []
    top_rows = []
    for row in rows:
        frame_path = os.path.join(args.rsu_root, row[args.frame_file_column])
        data = np.load(frame_path)
        canvas = data[args.canvas_key]
        spatial_features_2d, psm, rm = run_head(manager, canvas)
        prob = torch.sigmoid(psm.detach().cpu())
        pred_count, score_count, post_max_score, post_scores = postprocess(
            manager,
            psm,
            rm,
            args.score_threshold)
        head_rows.append(OrderedDict({
            'timestamp': row['timestamp'],
            'source_rsu_frame_file': row[args.frame_file_column],
            'canvas_key': args.canvas_key,
            'input_canvas_shape': shape_string(canvas),
            'backbone_spatial_features_2d_shape':
                shape_string(spatial_features_2d),
            'psm_shape': shape_string(psm),
            'rm_shape': shape_string(rm),
            'score_min': to_float(torch.min(prob)),
            'score_mean': to_float(torch.mean(prob)),
            'score_max': to_float(torch.max(prob)),
            'score_p95': to_float(torch.quantile(prob.flatten(), 0.95)),
            'score_threshold_used': (
                args.score_threshold if args.score_threshold is not None
                else manager.opencood_dataset.post_processor.params[
                    'target_args']['score_threshold']),
            'postprocess_pred_boxes': pred_count,
            'postprocess_scores': score_count,
            'postprocess_max_score': to_float(post_max_score),
            'postprocess_top_scores': post_scores,
        }))
        top_rows.extend(top_anchor_rows(row['timestamp'], psm, args.top_k))

    if head_rows:
        write_csv(os.path.join(args.output_dir, 'rsu_head_probe_summary.csv'),
                  list(head_rows[0].keys()),
                  head_rows)
    if top_rows:
        write_csv(os.path.join(args.output_dir, 'rsu_head_top_scores.csv'),
                  list(top_rows[0].keys()),
                  top_rows)
    config = {
        'rsu_root': os.path.abspath(args.rsu_root),
        'rsu_frame_manifest': os.path.abspath(args.rsu_frame_manifest),
        'fusion_method': args.fusion_method,
        'top_k': args.top_k,
        'score_threshold': args.score_threshold,
        'frame_file_column': args.frame_file_column,
        'canvas_key': args.canvas_key,
        'note': 'Detection-head feasibility probe; no GT/AP evaluation.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP PointPillar RSU Head Probe\n\n')
        stream.write('This run feeds assembled RSU scatter canvases through ')
        stream.write('the PointPillar backbone and detection heads. It also ')
        stream.write('attempts voxel postprocessing without GT/AP.\n\n')
        stream.write('- rows: `%d`\n' % len(head_rows))
        if head_rows:
            stream.write('- psm shape: `%s`\n' % head_rows[0]['psm_shape'])
            stream.write('- rm shape: `%s`\n' % head_rows[0]['rm_shape'])
            stream.write('- postprocess boxes: `%s`\n' %
                         head_rows[0]['postprocess_pred_boxes'])

    print('Wrote LGCP RSU head probe to %s' % args.output_dir)
    if head_rows:
        print('rows=%d psm=%s rm=%s post_boxes=%s score_max=%s' % (
            len(head_rows),
            head_rows[0]['psm_shape'],
            head_rows[0]['rm_shape'],
            head_rows[0]['postprocess_pred_boxes'],
            head_rows[0]['score_max']))


if __name__ == '__main__':
    main()
