# -*- coding: utf-8 -*-
"""
Minimal LGCP RSU BEV training smoke.

This is intentionally a small, explicit training loop for the sparse samples
exported by ``lgcp_rsu_bev_training_sample_export``. It validates the training
path before integrating a larger OpenCOOD YAML/dataset registry workflow.
"""

import argparse
import csv
import os
from collections import OrderedDict

import torch
import yaml

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.models.lgcp_rsu_bev_attentive import LgcpRsuBevAttentive
from opencood.tools import train_utils
from opencood.utils import eval_utils

from opencda.core.ml_libs.lgcp_rsu_bev_dataset import LGCPRSUBevSparseDataset
from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.offline_inference import load_coperception_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train-smoke LGCP RSU BEV attentive wrapper.')
    parser.add_argument('--train-root', required=True)
    parser.add_argument('--val-root', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-method', default='intermediate_attentive')
    parser.add_argument('--coperception-yaml', default=None)
    parser.add_argument('--checkpoint-dir',
                        default='opencood/logs/pointpillar_attentive_fusion')
    parser.add_argument('--query-mode',
                        choices=['input', 'mean', 'zero',
                                 'learnable_channel'],
                        default='learnable_channel')
    parser.add_argument('--dataset-query-mode',
                        choices=['mean', 'zero', 'first_leader'],
                        default='mean')
    parser.add_argument('--freeze-mode',
                        choices=['query_only', 'query_heads',
                                 'heads', 'none'],
                        default='query_heads')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-train-steps', type=int, default=1)
    parser.add_argument('--max-val-steps', type=int, default=1)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--eval-ap', action='store_true',
                        help='Also run validation postprocess/AP smoke.')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=None)
    parser.add_argument('--ap-score-thresholds', default='',
                        help='Comma-separated validation AP thresholds. '
                             'Overrides --postprocess-score-threshold when '
                             'set, e.g. "0.005,0.01,0.02".')
    parser.add_argument('--save-checkpoint', action='store_true')
    return parser.parse_args()


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def load_hypes(checkpoint_dir):
    return yaml_utils.load_yaml(
        None,
        argparse.Namespace(model_dir=checkpoint_dir))


def create_model(args, hypes, manager):
    model_args = dict(hypes['model']['args'])
    model_args['query_mode'] = args.query_mode
    model = LgcpRsuBevAttentive(model_args)
    # Reuse matching attentive backbone/head parameters when possible.
    model.load_state_dict(manager.model.state_dict(), strict=False)
    apply_freeze_mode(model, args.freeze_mode)
    return model


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


def trainable_parameter_count(model):
    return sum(param.numel() for param in model.parameters()
               if param.requires_grad)


def format_float(value):
    return '%.6f' % float(value)


def move_batch_to_device(batch, device):
    output = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        elif isinstance(value, dict):
            output[key] = move_batch_to_device(value, device)
        else:
            output[key] = value
    return output


def postprocess_predictions(post_processor, output, device, score_threshold):
    original_threshold = post_processor.params['target_args'][
        'score_threshold']
    if score_threshold is not None:
        post_processor.params['target_args']['score_threshold'] = (
            score_threshold)
    try:
        anchor_box = post_processor.generate_anchor_box()
        data_dict = {
            'ego': {
                'anchor_box': torch.from_numpy(anchor_box).to(device),
                'transformation_matrix': torch.eye(4, device=device),
            }
        }
        output_dict = {'ego': output}
        return post_processor.post_process(data_dict, output_dict)
    finally:
        post_processor.params['target_args']['score_threshold'] = (
            original_threshold)


def update_ap_stats(result_stat, pred_box_tensor, pred_score, gt_box_tensor):
    if pred_box_tensor is None or pred_score is None:
        pred_box_tensor = gt_box_tensor.new_zeros((0, 8, 3))
        pred_score = gt_box_tensor.new_zeros((0,))
    for iou in (0.3, 0.5, 0.7):
        eval_utils.calculate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat,
            iou)


def calculate_ap_safe(result_stat, iou):
    if result_stat[iou]['gt'] == 0:
        return ''
    stat = {
        iou: {
            'tp': list(result_stat[iou]['tp']),
            'fp': list(result_stat[iou]['fp']),
            'gt': result_stat[iou]['gt'],
        }
    }
    ap, _mrec, _mpre = eval_utils.calculate_ap(stat, iou)
    return format_float(ap)


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


def run_train(args, model, criterion, dataset, optimizer, device):
    rows = []
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for index in range(len(dataset)):
            sample = dataset[index]
            batch = dataset.collate_batch([sample])['ego']
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['label_dict'])
            loss.backward()
            optimizer.step()
            row = OrderedDict({
                'phase': 'train',
                'epoch': epoch,
                'step': step,
                'sample_index': index,
                'timestamp': sample['timestamp'],
                'loss': '%.9f' % float(loss.detach().cpu()),
                'psm_shape': 'x'.join(str(item) for item in output['psm'].shape),
                'rm_shape': 'x'.join(str(item) for item in output['rm'].shape),
                'pred_boxes': '',
                'pred_scores': '',
                'gt_boxes': int(sample['gt_boxes'].shape[0]),
            })
            rows.append(row)
            step += 1
            if args.max_train_steps and step >= args.max_train_steps:
                return rows
    return rows


def run_val(args, model, criterion, dataset, device, post_processor=None):
    rows = []
    thresholds = validation_ap_thresholds(args) if args.eval_ap else []
    result_stats = {
        threshold: make_result_stat()
        for threshold in thresholds
    }
    model.eval()
    limit = len(dataset)
    if args.max_val_steps:
        limit = min(limit, args.max_val_steps)
    with torch.no_grad():
        for index in range(limit):
            sample = dataset[index]
            batch = dataset.collate_batch([sample])['ego']
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            loss = criterion(output, batch['label_dict'])
            primary_pred_box_tensor = None
            primary_pred_score = None
            if args.eval_ap:
                gt_box_tensor = torch.from_numpy(sample['gt_boxes']).to(device)
                for threshold_index, threshold in enumerate(thresholds):
                    pred_box_tensor, pred_score = postprocess_predictions(
                        post_processor,
                        output,
                        device,
                        threshold)
                    update_ap_stats(
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
            rows.append(OrderedDict({
                'phase': 'val',
                'epoch': '',
                'step': index,
                'sample_index': index,
                'timestamp': sample['timestamp'],
                'loss': '%.9f' % float(loss.detach().cpu()),
                'psm_shape': 'x'.join(str(item) for item in output['psm'].shape),
                'rm_shape': 'x'.join(str(item) for item in output['rm'].shape),
                'pred_boxes': pred_count,
                'pred_scores': score_count,
                'gt_boxes': int(sample['gt_boxes'].shape[0]),
            }))
    ap_summary_rows = None
    if args.eval_ap:
        ap_summary_rows = []
        for threshold in thresholds:
            result_stat = result_stats[threshold]
            ap_summary_rows.append(OrderedDict({
                'val_samples_evaluated': limit,
                'score_threshold': (
                    threshold if threshold is not None else ''),
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
    hypes = load_hypes(args.checkpoint_dir)
    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    post_processor = manager.opencood_dataset.post_processor

    train_dataset = LGCPRSUBevSparseDataset(
        args.train_root,
        post_processor=post_processor,
        query_mode=args.dataset_query_mode,
        return_dense=True)
    val_dataset = None
    if args.val_root:
        val_dataset = LGCPRSUBevSparseDataset(
            args.val_root,
            post_processor=post_processor,
            query_mode=args.dataset_query_mode,
            return_dense=True)

    model = create_model(args, hypes, manager).to(device)
    criterion = PointPillarLoss(hypes['loss']['args'])
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr)

    train_rows = run_train(
        args,
        model,
        criterion,
        train_dataset,
        optimizer,
        device)
    val_rows = []
    ap_summary_rows = None
    if val_dataset is not None:
        val_rows, ap_summary_rows = run_val(
            args,
            model,
            criterion,
            val_dataset,
            device,
            post_processor=post_processor)
    rows = train_rows + val_rows
    if rows:
        write_csv(os.path.join(args.output_dir, 'loss_trace.csv'),
                  list(rows[0].keys()),
                  rows)
    summary = OrderedDict({
        'train_root': os.path.abspath(args.train_root),
        'val_root': os.path.abspath(args.val_root) if args.val_root else '',
        'query_mode': args.query_mode,
        'dataset_query_mode': args.dataset_query_mode,
        'freeze_mode': args.freeze_mode,
        'device': str(device),
        'epochs': args.epochs,
        'max_train_steps': args.max_train_steps,
        'max_val_steps': args.max_val_steps,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset) if val_dataset is not None else 0,
        'trainable_parameters': trainable_parameter_count(model),
        'train_final_loss': train_rows[-1]['loss'] if train_rows else '',
        'val_final_loss': val_rows[-1]['loss'] if val_rows else '',
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
                   os.path.join(args.output_dir, 'lgcp_rsu_bev_smoke.pth'))
    print('Wrote LGCP RSU BEV train smoke outputs to %s' % args.output_dir)
    print('train_final_loss=%s val_final_loss=%s trainable_parameters=%s' % (
        summary['train_final_loss'],
        summary['val_final_loss'],
        summary['trainable_parameters']))
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
