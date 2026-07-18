# -*- coding: utf-8 -*-
"""
Assemble leader-local LGCP PointPillar feature slices at the RSU.

The input is produced by lgcp_pointpillar_leader_feature_fusion.py. This tool
places leader-local scatter feature slices back onto a common PointPillar BEV
canvas and averages overlapping area cells.
"""

import argparse
import csv
import os
from collections import OrderedDict, defaultdict

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='Assemble LGCP leader feature slices at the RSU.')
    parser.add_argument('--leader-root', required=True)
    parser.add_argument('--leader-feature-manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--feature-key', default='leader_scatter_mean')
    parser.add_argument('--canvas-height', type=int, default=200)
    parser.add_argument('--canvas-width', type=int, default=704)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--dtype', choices=['float32', 'float16'],
                        default='float16')
    return parser.parse_args()


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_dtype(array, dtype_name):
    if dtype_name == 'float16':
        return array.astype(np.float16)
    return array.astype(np.float32)


def shape_string(array):
    return 'x'.join(str(int(item)) for item in array.shape)


def save_npz(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)
    return os.path.getsize(path)


def grouped_by_timestamp(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['timestamp']].append(row)
    return OrderedDict(sorted(grouped.items()))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_slice_dir = os.path.join(args.output_dir, 'rsu_frames')

    rows = read_csv(args.leader_feature_manifest)
    frame_rows = []
    for timestamp, timestamp_rows in grouped_by_timestamp(rows).items():
        accum = np.zeros(
            (1, args.channels, args.canvas_height, args.canvas_width),
            dtype=np.float32)
        counts = np.zeros(
            (1, 1, args.canvas_height, args.canvas_width),
            dtype=np.float32)
        used_rows = 0
        skipped_rows = 0
        for row in timestamp_rows:
            path = os.path.join(args.leader_root, row['leader_slice_file'])
            data = np.load(path)
            if args.feature_key not in data:
                skipped_rows += 1
                continue
            feature = data[args.feature_key].astype(np.float32)
            bounds = [int(item) for item in data['scatter_bounds_xyxy']]
            x0, x1, y0, y1 = bounds
            x0 = max(0, min(args.canvas_width, x0))
            x1 = max(0, min(args.canvas_width, x1))
            y0 = max(0, min(args.canvas_height, y0))
            y1 = max(0, min(args.canvas_height, y1))
            if x1 <= x0 or y1 <= y0:
                skipped_rows += 1
                continue
            height = min(y1 - y0, feature.shape[2])
            width = min(x1 - x0, feature.shape[3])
            accum[:, :, y0:y0 + height, x0:x0 + width] += (
                feature[:, :, :height, :width])
            counts[:, :, y0:y0 + height, x0:x0 + width] += 1.0
            used_rows += 1

        nonzero_mask = counts > 0
        canvas = np.zeros_like(accum)
        canvas[nonzero_mask.repeat(args.channels, axis=1)] = (
            accum[nonzero_mask.repeat(args.channels, axis=1)] /
            counts.repeat(args.channels, axis=1)[
                nonzero_mask.repeat(args.channels, axis=1)])
        coverage_cells = int(np.count_nonzero(counts[0, 0]))
        overlap_cells = int(np.count_nonzero(counts[0, 0] > 1))
        max_overlap = int(np.max(counts)) if coverage_cells else 0
        rel_path = os.path.join('rsu_frames', '%s_%s.npz' % (
            timestamp,
            args.feature_key))
        payload = {
            'timestamp': np.asarray(timestamp),
            'feature_key': np.asarray(args.feature_key),
            'rsu_canvas': to_dtype(canvas, args.dtype),
            'coverage_count': counts.astype(np.uint16),
        }
        compressed_bytes = save_npz(
            os.path.join(args.output_dir, rel_path),
            payload)
        uncompressed_bytes = (
            canvas.size * (2 if args.dtype == 'float16' else 4) +
            counts.size * 2)
        frame_rows.append(OrderedDict({
            'timestamp': timestamp,
            'rsu_frame_file': rel_path.replace('\\', '/'),
            'feature_key': args.feature_key,
            'dtype': args.dtype,
            'canvas_shape': shape_string(canvas),
            'input_rows': len(timestamp_rows),
            'used_rows': used_rows,
            'skipped_rows': skipped_rows,
            'coverage_cells': coverage_cells,
            'coverage_ratio': '%.6f' % (
                coverage_cells / float(args.canvas_height * args.canvas_width)),
            'overlap_cells': overlap_cells,
            'max_overlap': max_overlap,
            'uncompressed_bytes': uncompressed_bytes,
            'compressed_npz_bytes': compressed_bytes,
        }))

    if frame_rows:
        write_csv(os.path.join(args.output_dir, 'rsu_feature_frame_manifest.csv'),
                  list(frame_rows[0].keys()),
                  frame_rows)
    summary = summarize(frame_rows)
    write_csv(os.path.join(args.output_dir, 'rsu_feature_summary.csv'),
              list(summary.keys()),
              [summary])
    config = {
        'leader_root': os.path.abspath(args.leader_root),
        'leader_feature_manifest': os.path.abspath(
            args.leader_feature_manifest),
        'feature_key': args.feature_key,
        'canvas_height': args.canvas_height,
        'canvas_width': args.canvas_width,
        'channels': args.channels,
        'dtype': args.dtype,
        'note': 'RSU feature assembly smoke; no detection head/AP yet.',
    }
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP PointPillar RSU Feature Assembly\n\n')
        stream.write('This run places leader-local scatter feature slices ')
        stream.write('onto a common RSU BEV canvas and averages overlaps. ')
        stream.write('It does not run a detection head.\n\n')
        stream.write('- frames: `%s`\n' % summary['frames'])
        stream.write('- compressed bytes: `%s`\n' %
                     summary['compressed_npz_bytes'])
        stream.write('- mean coverage ratio: `%s`\n' %
                     summary['mean_coverage_ratio'])

    print('Wrote LGCP RSU feature assembly to %s' % args.output_dir)
    print('frames=%s compressed_npz_bytes=%s mean_coverage_ratio=%s' % (
        summary['frames'],
        summary['compressed_npz_bytes'],
        summary['mean_coverage_ratio']))


def summarize(rows):
    if not rows:
        return OrderedDict({
            'frames': 0,
            'compressed_npz_bytes': 0,
            'uncompressed_bytes': 0,
            'mean_coverage_ratio': 0.0,
            'mean_overlap_cells': 0.0,
            'max_overlap': 0,
        })
    compressed = sum(int(row['compressed_npz_bytes']) for row in rows)
    uncompressed = sum(int(row['uncompressed_bytes']) for row in rows)
    coverage = sum(float(row['coverage_ratio']) for row in rows) / len(rows)
    overlap = sum(int(row['overlap_cells']) for row in rows) / len(rows)
    max_overlap = max(int(row['max_overlap']) for row in rows)
    return OrderedDict({
        'frames': len(rows),
        'compressed_npz_bytes': compressed,
        'uncompressed_bytes': uncompressed,
        'mean_coverage_ratio': '%.6f' % coverage,
        'mean_overlap_cells': '%.6f' % overlap,
        'max_overlap': max_overlap,
    })


if __name__ == '__main__':
    main()
