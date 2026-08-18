# -*- coding: utf-8 -*-
"""Probe the runtime CAV-count limit of the vendored Where2comm fusion path."""

import argparse
import os
import time

import torch

from opencda.core.ml_libs.opencood_manager import OpenCOODManager
from opencda.tools.lgcp_where2comm_area_mask_eval import (
    enable_external_mask_semantics,
    load_coperception_params,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run synthetic Where2comm multiscale fusion with different '
                    'record_len values.')
    parser.add_argument('--coperception-yaml',
                        default='opencda/scenario_testing/config_yaml/'
                                'enable_coperception.yaml')
    parser.add_argument('--fusion-method',
                        default='intermediate_where2comm')
    parser.add_argument('--dataset-root',
                        default='D:/Data/Carla')
    parser.add_argument('--cav-counts',
                        default='5,8,10,13,16,20,24,32',
                        help='Comma-separated total CAV counts, including ego.')
    parser.add_argument('--device',
                        choices=['manager', 'cpu', 'cuda'],
                        default='manager')
    parser.add_argument('--height0', type=int, default=96)
    parser.add_argument('--width0', type=int, default=352)
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def identity_affine(batch_size, agent_count, device):
    matrix = torch.zeros(
        (batch_size, agent_count, agent_count, 2, 3),
        dtype=torch.float32,
        device=device)
    matrix[..., 0, 0] = 1.0
    matrix[..., 1, 1] = 1.0
    return matrix


def tensor_mb(tensor):
    return tensor.numel() * tensor.element_size() / (1024.0 * 1024.0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    params['_dataset_root_override'] = args.dataset_root
    manager = OpenCOODManager(params)
    enable_external_mask_semantics(manager.model, 'replace')
    model = manager.model.eval()
    if args.device == 'cpu':
        device = torch.device('cpu')
        model = model.to(device)
    elif args.device == 'cuda':
        device = torch.device('cuda')
        model = model.to(device)
    else:
        device = manager.device

    cav_counts = [
        int(item) for item in args.cav_counts.split(',') if item.strip()
    ]
    scale_shapes = [
        (64, args.height0, args.width0),
        (128, args.height0 // 2, args.width0 // 2),
        (256, args.height0 // 4, args.width0 // 4),
    ]
    print('device,total_cav,ok,elapsed_s,feature_mb,error')
    for cav_count in cav_counts:
        start = time.time()
        feature_mb = 0.0
        error = ''
        ok = True
        try:
            record_len = torch.tensor([cav_count],
                                      dtype=torch.int64,
                                      device=device)
            affine = identity_affine(1, cav_count, device)
            psm = torch.rand(
                (cav_count, 2, args.height0, args.width0),
                dtype=torch.float32,
                device=device)
            with torch.no_grad():
                for scale_index, fuse_module in enumerate(model.fusion_net):
                    channels, height, width = scale_shapes[scale_index]
                    feature = torch.rand(
                        (cav_count, channels, height, width),
                        dtype=torch.float32,
                        device=device)
                    mask = torch.ones(
                        (cav_count, 1, height, width),
                        dtype=torch.float32,
                        device=device)
                    feature_mb += tensor_mb(feature)
                    out, _comm_rate = fuse_module(
                        feature,
                        psm,
                        record_len,
                        affine,
                        None,
                        external_comm_mask=mask)
                    feature_mb += tensor_mb(out)
                    del feature, mask, out
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        except Exception as exc:  # pylint: disable=broad-except
            ok = False
            error = type(exc).__name__ + ':' + str(exc).replace(',', ';')
        elapsed = time.time() - start
        print('%s,%d,%s,%.6f,%.3f,%s' % (
            str(device),
            cav_count,
            str(ok).lower(),
            elapsed,
            feature_mb,
            error))
        if not ok:
            break


if __name__ == '__main__':
    main()
