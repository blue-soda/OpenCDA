# -*- coding: utf-8 -*-
"""
Fuse exported LGCP PointPillar feature slices at the leader side.

The input is produced by lgcp_pointpillar_feature_slice_export.py. This tool
performs a minimal leader-local fusion over the CAV dimension of each scatter
slice and writes leader feature slices plus a manifest for later RSU assembly.
"""

import argparse
import csv
import os
from collections import OrderedDict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fuse LGCP PointPillar feature slices at leaders.')
    parser.add_argument('--slice-root', required=True)
    parser.add_argument('--feature-slice-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fusion-methods', default='mean,max',
                        help='Comma-separated scatter fusion methods.')
    parser.add_argument('--dtype', choices=['float32', 'float16'],
                        default='float16')
    parser.add_argument('--keep-model-fused', action='store_true',
                        help='Copy model fused slice into leader output npz.')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_methods(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def to_dtype(array, dtype_name):
    if dtype_name == 'float16':
        return array.astype(np.float16)
    return array.astype(np.float32)


def fuse_scatter(scatter, method):
    if method == 'mean':
        return np.mean(scatter.astype(np.float32), axis=0, keepdims=True)
    if method == 'max':
        return np.max(scatter, axis=0, keepdims=True)
    if method == 'sum':
        return np.sum(scatter.astype(np.float32), axis=0, keepdims=True)
    raise ValueError('Unsupported fusion method: %s' % method)


def safe_name(row):
    return '%s_area%s_leader%s.npz' % (
        row['timestamp'],
        str(row['area_id']).replace('/', '_').replace('\\', '_'),
        row['leader_id'])


def shape_string(array):
    return 'x'.join(str(int(item)) for item in array.shape)


def save_npz(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_slice_dir = os.path.join(args.output_dir, 'leader_slices')
    methods = parse_methods(args.fusion_methods)
    if not methods:
        raise ValueError('At least one fusion method is required.')

    rows = read_csv(args.feature_slice_manifest)
    manifest_rows = []
    for row in rows:
        input_path = os.path.join(args.slice_root, row['slice_file'])
        data = np.load(input_path)
        if 'scatter' not in data:
            continue
        scatter = data['scatter']
        payload = {
            'timestamp': np.asarray(row['timestamp']),
            'area_id': np.asarray(row['area_id']),
            'leader_id': np.asarray(str(row['leader_id'])),
            'group_members': np.asarray(row['group_members']),
            'source_slice_file': np.asarray(row['slice_file']),
            'scatter_bounds_xyxy': data['scatter_bounds_xyxy'],
        }

        fused_elements = 0
        for method in methods:
            fused = to_dtype(fuse_scatter(scatter, method), args.dtype)
            payload['leader_scatter_%s' % method] = fused
            fused_elements += int(fused.size)

        model_fused_shape = ''
        if args.keep_model_fused and 'fused' in data:
            model_fused = to_dtype(data['fused'], args.dtype)
            payload['model_fused_reference'] = model_fused
            payload['model_fused_bounds_xyxy'] = data['fused_bounds_xyxy']
            fused_elements += int(model_fused.size)
            model_fused_shape = shape_string(model_fused)

        rel_path = os.path.join('leader_slices', safe_name(row))
        compressed_bytes = save_npz(
            os.path.join(args.output_dir, rel_path),
            payload)
        bytes_per_value = 2 if args.dtype == 'float16' else 4
        first_method = methods[0]
        first_key = 'leader_scatter_%s' % first_method
        manifest_rows.append(OrderedDict({
            'scenario_id': row['scenario_id'],
            'timestamp': row['timestamp'],
            'area_id': row['area_id'],
            'leader_id': row['leader_id'],
            'group_members': row['group_members'],
            'group_size': row['group_size'],
            'leader_slice_file': rel_path.replace('\\', '/'),
            'source_slice_file': row['slice_file'],
            'fusion_methods': ';'.join(methods),
            'dtype': args.dtype,
            'source_scatter_shape': row['scatter_slice_shape'],
            'leader_scatter_shape': shape_string(payload[first_key]),
            'model_fused_reference_shape': model_fused_shape,
            'uncompressed_bytes': fused_elements * bytes_per_value,
            'compressed_npz_bytes': compressed_bytes,
        }))

    manifest_path = os.path.join(args.output_dir, 'leader_feature_manifest.csv')
    if manifest_rows:
        write_csv(manifest_path, list(manifest_rows[0].keys()), manifest_rows)
    summary = summarize(manifest_rows)
    write_csv(os.path.join(args.output_dir, 'leader_feature_summary.csv'),
              list(summary.keys()),
              [summary])

    config = {
        'slice_root': os.path.abspath(args.slice_root),
        'feature_slice_manifest': os.path.abspath(
            args.feature_slice_manifest),
        'fusion_methods': methods,
        'dtype': args.dtype,
        'keep_model_fused': args.keep_model_fused,
        'note': 'Leader-local feature fusion smoke; no RSU assembly yet.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP PointPillar Leader Feature Fusion\n\n')
        stream.write('This run fuses per-CAV scatter slices into leader-local ')
        stream.write('feature slices. It does not assemble RSU global maps.\n\n')
        stream.write('- rows: `%s`\n' % summary['rows'])
        stream.write('- compressed bytes: `%s`\n' %
                     summary['compressed_npz_bytes'])
        stream.write('- uncompressed bytes: `%s`\n' %
                     summary['uncompressed_bytes'])

    print('Wrote LGCP leader feature fusion to %s' % args.output_dir)
    print('rows=%s compressed_npz_bytes=%s uncompressed_bytes=%s' % (
        summary['rows'],
        summary['compressed_npz_bytes'],
        summary['uncompressed_bytes']))


def summarize(rows):
    if not rows:
        return OrderedDict({
            'rows': 0,
            'compressed_npz_bytes': 0,
            'uncompressed_bytes': 0,
            'mean_compressed_npz_bytes': 0.0,
            'mean_uncompressed_bytes': 0.0,
        })
    compressed = sum(int(row['compressed_npz_bytes']) for row in rows)
    uncompressed = sum(int(row['uncompressed_bytes']) for row in rows)
    return OrderedDict({
        'rows': len(rows),
        'compressed_npz_bytes': compressed,
        'uncompressed_bytes': uncompressed,
        'mean_compressed_npz_bytes': '%.6f' % (compressed / len(rows)),
        'mean_uncompressed_bytes': '%.6f' % (uncompressed / len(rows)),
    })


if __name__ == '__main__':
    main()
