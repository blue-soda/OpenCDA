# -*- coding: utf-8 -*-
"""
Minimal train smoke for LGCP V2X-ViT RSU aggregation.

This validates the trainable path after the feature-packet probe:
area point slices -> compressed leader packets -> explicit RSU query/head
training. It is intentionally small and isolated from SGCP.
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.models.lgcp_v2xvit_rsu import LgcpV2XRsu
from opencood.utils import box_utils

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.lgcp_pointpillar_rsu_bev_fusion import (
    build_area_leader_points,
    build_gt_batch,
    build_planned_area_bounds,
    calculate_ap_safe,
    filter_boxes_to_planned_areas,
    grouped_by_timestamp,
    load_frame_for_reference,
    parse_members,
    postprocess_predictions,
    read_csv,
    resolve_reference_pose,
    selected_timestamps,
    unique_strings,
    update_stats,
    write_csv,
)
from opencda.tools.lgcp_v2xvit_feature_probe import (
    encode_backbone_features,
    shape_string,
)
from opencda.tools.lgcp_v2xvit_rsu_detection_probe import (
    assemble_latent_packets,
    load_lidar_range,
)
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train-smoke LGCP V2X-ViT RSU wrapper.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--scenario-id', required=True)
    parser.add_argument('--assignment-plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_v2xvit')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--reference-cav-id', default='1')
    parser.add_argument('--reference-pose', nargs=6, type=float, default=None)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=1)
    parser.add_argument('--max-areas-per-frame', type=int, default=2)
    parser.add_argument('--grid-size-x', type=float, default=10.0)
    parser.add_argument('--grid-size-y', type=float, default=6.0)
    parser.add_argument('--crop-halo-cells', type=int, default=1)
    parser.add_argument('--packet-mode', choices=['crop', 'full'],
                        default='crop')
    parser.add_argument('--query-mode',
                        choices=['input', 'mean', 'zero',
                                 'learnable_channel'],
                        default='learnable_channel')
    parser.add_argument('--freeze-mode',
                        choices=['query_only', 'query_heads', 'heads',
                                 'query_fusion_heads', 'none'],
                        default='query_heads')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-train-steps', type=int, default=1)
    parser.add_argument('--val-start-index', type=int, default=None)
    parser.add_argument('--val-max-frames', type=int, default=0)
    parser.add_argument('--max-val-steps', type=int, default=0)
    parser.add_argument('--eval-ap', action='store_true')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=0.05)
    parser.add_argument('--ap-score-thresholds', default='',
                        help='Comma-separated validation AP thresholds. '
                             'Overrides --postprocess-score-threshold.')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--save-checkpoint', action='store_true')
    return parser.parse_args()


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def load_hypes(checkpoint_dir):
    return yaml_utils.load_yaml(
        None,
        argparse.Namespace(model_dir=checkpoint_dir))


def apply_freeze_mode(model, freeze_mode):
    if freeze_mode == 'none':
        return
    for param in model.parameters():
        param.requires_grad = False
    if freeze_mode in ('query_only', 'query_heads'):
        if hasattr(model, 'rsu_query_channel'):
            model.rsu_query_channel.requires_grad = True
    if freeze_mode in ('query_heads', 'heads'):
        for param in model.cls_head.parameters():
            param.requires_grad = True
        for param in model.reg_head.parameters():
            param.requires_grad = True
    if freeze_mode == 'query_fusion_heads':
        if hasattr(model, 'rsu_query_channel'):
            model.rsu_query_channel.requires_grad = True
        for param in model.fusion_net.parameters():
            param.requires_grad = True
        for param in model.cls_head.parameters():
            param.requires_grad = True
        for param in model.reg_head.parameters():
            param.requires_grad = True


def trainable_parameter_count(model):
    return sum(param.numel() for param in model.parameters()
               if param.requires_grad)


def validation_ap_thresholds(args):
    if args.ap_score_thresholds:
        values = []
        for raw_value in args.ap_score_thresholds.split(','):
            raw_value = raw_value.strip()
            if raw_value:
                values.append(float(raw_value))
        if not values:
            raise ValueError('--ap-score-thresholds was set but empty.')
        return values
    return [args.postprocess_score_threshold]


def make_result_stat():
    return {
        0.3: {'tp': [], 'fp': [], 'gt': 0},
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0},
    }


def move_label_to_device(label_dict, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in label_dict.items()
    }


def generate_label_dict(post_processor, gt_box_tensor):
    max_num = int(post_processor.params.get('max_num', 100))
    order = post_processor.params.get('order', 'hwl')
    gt_corners = gt_box_tensor.detach().cpu().numpy().astype(np.float32)
    gt_centers = np.zeros((max_num, 7), dtype=np.float32)
    mask = np.zeros((max_num,), dtype=np.float32)
    if gt_corners.shape[0] > 0:
        centers = box_utils.corner_to_center(gt_corners, order=order)
        valid_count = min(max_num, centers.shape[0])
        gt_centers[:valid_count] = centers[:valid_count]
        mask[:valid_count] = 1
    label = post_processor.generate_label(
        gt_box_center=gt_centers,
        anchors=post_processor.generate_anchor_box(),
        mask=mask)
    return post_processor.collate_batch([label])


def create_model(args, hypes, manager):
    model_args = dict(hypes['model']['args'])
    model_args['query_mode'] = args.query_mode
    model = LgcpV2XRsu(model_args)
    model.load_state_dict(manager.model.state_dict(), strict=False)
    apply_freeze_mode(model, args.freeze_mode)
    return model


def build_training_sample(args, manager, dataset, grouped, timestamp,
                          lidar_range):
    reference_pose, reference_label = resolve_reference_pose(
        dataset,
        args.scenario_id,
        timestamp,
        args.reference_cav_id,
        args.reference_pose)
    frame_plan = grouped[timestamp]
    if args.max_areas_per_frame:
        frame_plan = frame_plan[:args.max_areas_per_frame]
    required_cavs = [args.reference_cav_id]
    for row in frame_plan:
        required_cavs.append(str(row['leader_id']))
        required_cavs.extend(parse_members(row['group_members']))
    frame = load_frame_for_reference(
        dataset,
        args.scenario_id,
        timestamp,
        unique_strings(required_cavs),
        reference_pose,
        args.reference_cav_id)

    leader_packets = []
    point_batches = []
    for row in frame_plan:
        packet = build_area_leader_points(
            frame,
            row,
            reference_pose,
            args.grid_size_x,
            args.grid_size_y)
        leader_packets.append((row, packet))
        point_batches.append(packet['points_reference'])

    _scatter, _shrink, compressed, valid_indices = encode_backbone_features(
        manager,
        point_batches)
    if compressed is None:
        return None
    valid_packets = [leader_packets[index] for index in valid_indices]
    assembled = assemble_latent_packets(
        args,
        compressed,
        valid_packets,
        reference_pose,
        lidar_range)
    if isinstance(assembled, tuple):
        assembled_latent, crop_cells = assembled
    else:
        assembled_latent = assembled
        crop_cells = int(assembled_latent.shape[0] *
                         assembled_latent.shape[2] *
                         assembled_latent.shape[3])

    gt_batch = build_gt_batch(
        manager,
        dataset,
        args.scenario_id,
        timestamp,
        reference_pose,
        args.reference_cav_id)
    gt_box_tensor = manager.opencood_dataset.post_processor.generate_gt_bbx(
        gt_batch)
    planned_bounds = build_planned_area_bounds(
        frame_plan,
        args.grid_size_x,
        args.grid_size_y)
    gt_box_tensor, _ = filter_boxes_to_planned_areas(
        gt_box_tensor,
        None,
        reference_pose,
        planned_bounds)
    return {
        'timestamp': timestamp,
        'reference': reference_label,
        'compressed_features': assembled_latent,
        'leader_record_len': torch.tensor(
            [assembled_latent.shape[0]], dtype=torch.int64),
        'gt_box_tensor': gt_box_tensor,
        'reference_pose': reference_pose,
        'planned_bounds': planned_bounds,
        'planned_areas': len(frame_plan),
        'valid_leader_features': len(valid_packets),
        'crop_cells': crop_cells,
    }


def sample_to_batch(sample, device):
    return {
        'compressed_features': sample['compressed_features'].to(device),
        'leader_record_len': sample['leader_record_len'].to(device),
    }


def append_trace_row(rows, phase, epoch, step, sample, loss, output,
                     grad_norm='', pred_count='', score_count=''):
    rows.append(OrderedDict({
        'phase': phase,
        'epoch': epoch,
        'step': step,
        'timestamp': sample['timestamp'],
        'reference': sample['reference'],
        'planned_areas': sample['planned_areas'],
        'valid_leader_features': sample['valid_leader_features'],
        'crop_cells': sample['crop_cells'],
        'gt_boxes': int(sample['gt_box_tensor'].shape[0]),
        'loss': '%.9f' % float(loss.detach().cpu()),
        'query_grad_norm': grad_norm,
        'pred_boxes': pred_count,
        'pred_scores': score_count,
        'compressed_shape': shape_string(sample['compressed_features']),
        'psm_shape': shape_string(output['psm']),
        'rm_shape': shape_string(output['rm']),
    }))


def run_train(args, model, criterion, optimizer, manager, dataset, grouped,
              timestamps, lidar_range, post_processor, device):
    rows = []
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for timestamp in timestamps:
            sample = build_training_sample(
                args,
                manager,
                dataset,
                grouped,
                timestamp,
                lidar_range)
            if sample is None:
                continue
            batch = sample_to_batch(sample, device)
            label_dict = generate_label_dict(
                post_processor,
                sample['gt_box_tensor'])
            label_dict = move_label_to_device(label_dict, device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, label_dict)
            loss.backward()
            grad_norm = ''
            if hasattr(model, 'rsu_query_channel'):
                grad = model.rsu_query_channel.grad
                if grad is not None:
                    grad_norm = '%.9f' % float(grad.norm().detach().cpu())
            optimizer.step()
            append_trace_row(
                rows,
                'train',
                epoch,
                step,
                sample,
                loss,
                output,
                grad_norm=grad_norm)
            step += 1
            if args.max_train_steps and step >= args.max_train_steps:
                return rows
    return rows


def run_val(args, model, criterion, manager, dataset, grouped, timestamps,
            lidar_range, post_processor, device):
    rows = []
    thresholds = validation_ap_thresholds(args) if args.eval_ap else []
    result_stats = {
        threshold: make_result_stat()
        for threshold in thresholds
    }
    model.eval()
    limit = len(timestamps)
    if args.max_val_steps:
        limit = min(limit, args.max_val_steps)
    with torch.no_grad():
        for step, timestamp in enumerate(timestamps[:limit]):
            sample = build_training_sample(
                args,
                manager,
                dataset,
                grouped,
                timestamp,
                lidar_range)
            if sample is None:
                continue
            batch = sample_to_batch(sample, device)
            label_dict = generate_label_dict(
                post_processor,
                sample['gt_box_tensor'])
            label_dict = move_label_to_device(label_dict, device)
            output = model(batch)
            loss = criterion(output, label_dict)
            primary_pred_box_tensor = None
            primary_pred_score = None
            if args.eval_ap:
                gt_box_tensor = sample['gt_box_tensor'].to(device)
                for threshold_index, threshold in enumerate(thresholds):
                    pred_box_tensor, pred_score = postprocess_predictions(
                        manager,
                        output['psm'],
                        output['rm'],
                        threshold)
                    pred_box_tensor, pred_score = filter_boxes_to_planned_areas(
                        pred_box_tensor,
                        pred_score,
                        sample['reference_pose'],
                        sample['planned_bounds'])
                    update_stats(
                        result_stats[threshold],
                        pred_box_tensor,
                        pred_score,
                        gt_box_tensor)
                    if threshold_index == 0:
                        primary_pred_box_tensor = pred_box_tensor
                        primary_pred_score = pred_score
            pred_count = 0 if primary_pred_box_tensor is None else int(
                primary_pred_box_tensor.shape[0])
            score_count = 0 if primary_pred_score is None else int(
                primary_pred_score.shape[0])
            append_trace_row(
                rows,
                'val',
                '',
                step,
                sample,
                loss,
                output,
                pred_count=pred_count,
                score_count=score_count)
    ap_summary_rows = None
    if args.eval_ap:
        ap_summary_rows = []
        for threshold in thresholds:
            result_stat = result_stats[threshold]
            ap_summary_rows.append(OrderedDict({
                'val_samples_evaluated': limit,
                'score_threshold': threshold,
                'gt_boxes': result_stat[0.5]['gt'],
                'pred_samples': len(result_stat[0.5]['tp']),
                'ap_03': calculate_ap_safe(result_stat, 0.3),
                'ap_05': calculate_ap_safe(result_stat, 0.5),
                'ap_07': calculate_ap_safe(result_stat, 0.7),
            }))
    return rows, ap_summary_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    assignment_rows = read_csv(args.assignment_plan)
    grouped = grouped_by_timestamp(assignment_rows)
    timestamps = selected_timestamps(
        grouped,
        args.start_index,
        args.max_frames)
    val_timestamps = []
    if args.val_start_index is not None or args.val_max_frames:
        val_start_index = args.val_start_index
        if val_start_index is None:
            val_start_index = args.start_index + len(timestamps)
        val_timestamps = selected_timestamps(
            grouped,
            val_start_index,
            args.val_max_frames)

    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    checkpoint_dir = coperception_params['models'][args.fusion_method]
    hypes = load_hypes(checkpoint_dir)
    manager = OpenCOODManager(coperception_params)
    lidar_range = load_lidar_range(checkpoint_dir)
    dataset = OPV2VFrameDataset(args.dataset_root)
    post_processor = manager.opencood_dataset.post_processor

    model = create_model(args, hypes, manager).to(device)
    criterion = PointPillarLoss(hypes['loss']['args'])
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr)

    train_rows = run_train(
        args,
        model,
        criterion,
        optimizer,
        manager,
        dataset,
        grouped,
        timestamps,
        lidar_range,
        post_processor,
        device)
    val_rows = []
    ap_summary_rows = None
    if val_timestamps:
        val_rows, ap_summary_rows = run_val(
            args,
            model,
            criterion,
            manager,
            dataset,
            grouped,
            val_timestamps,
            lidar_range,
            post_processor,
            device)
    rows = train_rows + val_rows

    if rows:
        write_csv(os.path.join(args.output_dir, 'loss_trace.csv'),
                  list(rows[0].keys()),
                  rows)
    summary = OrderedDict({
        'checkpoint_dir': os.path.abspath(checkpoint_dir),
        'query_mode': args.query_mode,
        'freeze_mode': args.freeze_mode,
        'packet_mode': args.packet_mode,
        'device': str(device),
        'epochs': args.epochs,
        'max_train_steps': args.max_train_steps,
        'max_val_steps': args.max_val_steps,
        'train_steps': len(train_rows),
        'val_steps': len(val_rows),
        'trainable_parameters': trainable_parameter_count(model),
        'train_final_loss': train_rows[-1]['loss'] if train_rows else '',
        'val_final_loss': val_rows[-1]['loss'] if val_rows else '',
        'query_final_grad_norm': (
            train_rows[-1]['query_grad_norm'] if train_rows else ''),
    })
    write_csv(os.path.join(args.output_dir, 'train_summary.csv'),
              list(summary.keys()),
              [summary])
    if ap_summary_rows is not None:
        write_csv(os.path.join(args.output_dir, 'val_ap_summary.csv'),
                  list(ap_summary_rows[0].keys()),
                  ap_summary_rows)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False)
    if args.save_checkpoint:
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, 'lgcp_v2xvit_rsu_smoke.pth'))
    print('Wrote LGCP V2X-ViT RSU train smoke outputs to %s' %
          args.output_dir)
    print('train_final_loss=%s trainable_parameters=%s query_grad_norm=%s' % (
        summary['train_final_loss'],
        summary['trainable_parameters'],
        summary['query_final_grad_norm']))
    if ap_summary_rows is not None:
        best_row = max(
            ap_summary_rows,
            key=lambda row: float(row['ap_05']) if row['ap_05'] != '' else -1)
        print('best_val_ap_threshold=%s val_ap_03=%s val_ap_05=%s '
              'val_ap_07=%s' % (
                  best_row['score_threshold'],
                  best_row['ap_03'],
                  best_row['ap_05'],
                  best_row['ap_07']))


if __name__ == '__main__':
    main()
