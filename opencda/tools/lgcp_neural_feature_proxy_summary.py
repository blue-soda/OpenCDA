# -*- coding: utf-8 -*-
"""
Summarize LGCP neural feature hierarchy proxy results.

This tool intentionally separates feature-path feasibility, communication
bytes, coverage, and low-AP boundary results. It should not be used to claim
paper-level model AP for the current uncalibrated feature warp.
"""

import argparse
import csv
import os
from collections import OrderedDict

import yaml


DEFAULT_PATHS = {
    'raw_slice_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_carla_feature_slice_area23_11f/'
        'feature_slice_summary.csv',
    'feature_slice_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_feature_slice_export_area23_1f/'
        'feature_slice_summary.csv',
    'leader_feature_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_leader_feature_fusion_area23_1f/'
        'leader_feature_summary.csv',
    'rsu_assembly_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_rsu_feature_assembly_area23_1f/'
        'rsu_feature_summary.csv',
    'nearest_warp_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_coordinate_warp_assembly_area23_1f_ref1/'
        'coordinate_warp_summary.csv',
    'nearest_head_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_coordinate_warp_head_probe_area23_1f_ref1/'
        'rsu_head_probe_summary.csv',
    'nearest_ap_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_coordinate_warp_ap_probe_area23_1f_ref1/'
        'warp_ap_summary.csv',
    'bilinear_warp_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_coordinate_warp_bilinear_assembly_area23_1f_ref1/'
        'coordinate_warp_summary.csv',
    'bilinear_head_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_coordinate_warp_bilinear_head_probe_area23_1f_ref1/'
        'rsu_head_probe_summary.csv',
    'bilinear_ap_summary':
        'docs/doc_workspace/LGCP/experiments/hierarchy_plan/'
        '20260718_lgcp_pointpillar_coordinate_warp_bilinear_ap_probe_area23_1f_ref1/'
        'warp_ap_summary.csv',
    'area_slice_accounting':
        'docs/doc_workspace/LGCP/experiments/ablation/'
        '20260718_lgcp_local_to_global_ablation_alignment/'
        'unified_area_slice_accounting_summary.csv',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create LGCP neural feature proxy summary tables.')
    parser.add_argument('--output-dir', required=True)
    for key, value in DEFAULT_PATHS.items():
        parser.add_argument('--%s' % key.replace('_', '-'), default=value)
    return parser.parse_args()


def read_first(path):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('CSV has no rows: %s' % path)
    return rows[0]


def read_rows(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def f(row, key, default=0.0):
    value = row.get(key, '')
    if value == '' or value is None:
        return default
    return float(value)


def ratio(value, denom):
    if denom <= 0:
        return ''
    return '%.6f' % (value / denom)


def fmt(value):
    if value == '' or value is None:
        return ''
    return '%.6f' % float(value)


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def comm_aware_area23_bytes(rows):
    for row in rows:
        if (row.get('area_plan') == 'area23' and
                row.get('method') == 'comm_aware_topk_10'):
            return f(row, 'area_slice_bytes_per_frame')
    return 0.0


def add_row(rows, raw_member_bytes, comm_area_bytes, **values):
    bytes_per_frame = float(values.get('bytes_per_frame') or 0.0)
    row = OrderedDict({
        'stage': values.get('stage', ''),
        'scope': values.get('scope', ''),
        'frames': values.get('frames', ''),
        'rows_or_areas': values.get('rows_or_areas', ''),
        'bytes_per_frame': fmt(bytes_per_frame) if bytes_per_frame else '',
        'mean_bytes_per_area': values.get('mean_bytes_per_area', ''),
        'ratio_vs_raw_member_area23': ratio(bytes_per_frame,
                                            raw_member_bytes),
        'ratio_vs_comm_aware_area23_slice': ratio(bytes_per_frame,
                                                  comm_area_bytes),
        'coverage_ratio': values.get('coverage_ratio', ''),
        'sample_ratio': values.get('sample_ratio', ''),
        'head_score_max': values.get('head_score_max', ''),
        'pred_boxes': values.get('pred_boxes', ''),
        'ap_05': values.get('ap_05', ''),
        'ap_07': values.get('ap_07', ''),
        'paper_safe_use': values.get('paper_safe_use', ''),
        'source': values.get('source', ''),
    })
    rows.append(row)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    paths = {key: getattr(args, key) for key in DEFAULT_PATHS}
    raw = read_first(paths['raw_slice_summary'])
    feature = read_first(paths['feature_slice_summary'])
    leader = read_first(paths['leader_feature_summary'])
    rsu = read_first(paths['rsu_assembly_summary'])
    nearest_warp = read_first(paths['nearest_warp_summary'])
    nearest_head = read_first(paths['nearest_head_summary'])
    nearest_ap = read_first(paths['nearest_ap_summary'])
    bilinear_warp = read_first(paths['bilinear_warp_summary'])
    bilinear_head = read_first(paths['bilinear_head_summary'])
    bilinear_ap = read_first(paths['bilinear_ap_summary'])
    area_accounting = read_rows(paths['area_slice_accounting'])

    raw_member_bytes = f(raw, 'member_upload_bytes_mean')
    comm_area_bytes = comm_aware_area23_bytes(area_accounting)

    rows = []
    add_row(
        rows, raw_member_bytes, comm_area_bytes,
        stage='raw_member_area_slice',
        scope='area23_11f_mean',
        frames=raw['frames'],
        rows_or_areas=raw['areas_mean'],
        bytes_per_frame=raw_member_bytes,
        paper_safe_use='baseline raw member-upload byte reference',
        source=paths['raw_slice_summary'])
    add_row(
        rows, raw_member_bytes, comm_area_bytes,
        stage='flat_comm_aware_area_slice',
        scope='area23_11f_mean',
        frames='11',
        rows_or_areas='10_agents_on_23_areas',
        bytes_per_frame=comm_area_bytes,
        paper_safe_use='strong flat area-slice byte reference',
        source=paths['area_slice_accounting'])
    add_row(
        rows, raw_member_bytes, comm_area_bytes,
        stage='pointpillar_feature_crop',
        scope='top23_1f',
        frames='1',
        rows_or_areas=feature['rows'],
        bytes_per_frame=f(feature, 'compressed_npz_bytes'),
        mean_bytes_per_area=feature['mean_compressed_npz_bytes'],
        paper_safe_use='feature crop feasibility and unoptimized byte proxy',
        source=paths['feature_slice_summary'])
    add_row(
        rows, raw_member_bytes, comm_area_bytes,
        stage='leader_scatter_fusion',
        scope='top23_1f',
        frames='1',
        rows_or_areas=leader['rows'],
        bytes_per_frame=f(leader, 'compressed_npz_bytes'),
        mean_bytes_per_area=leader['mean_compressed_npz_bytes'],
        paper_safe_use='leader-local deterministic feature fusion smoke',
        source=paths['leader_feature_summary'])
    add_row(
        rows, raw_member_bytes, comm_area_bytes,
        stage='rsu_index_canvas',
        scope='top23_1f',
        frames=rsu['frames'],
        rows_or_areas='23',
        bytes_per_frame=f(rsu, 'compressed_npz_bytes'),
        coverage_ratio=rsu['mean_coverage_ratio'],
        paper_safe_use='RSU canvas interface smoke; not coordinate-valid AP',
        source=paths['rsu_assembly_summary'])
    add_row(
        rows, raw_member_bytes, comm_area_bytes,
        stage='coordinate_warp_nearest',
        scope='top23_1f_ref1',
        frames=nearest_warp['frames'],
        rows_or_areas='23',
        bytes_per_frame=f(nearest_warp, 'compressed_npz_bytes'),
        coverage_ratio=nearest_warp['mean_coverage_ratio'],
        sample_ratio=nearest_warp['mean_sample_ratio'],
        head_score_max=nearest_head['score_max'],
        pred_boxes=nearest_head['postprocess_pred_boxes'],
        ap_05=nearest_ap['ap_05'],
        ap_07=nearest_ap['ap_07'],
        paper_safe_use='negative AP boundary for uncalibrated nearest warp',
        source=paths['nearest_ap_summary'])
    add_row(
        rows, raw_member_bytes, comm_area_bytes,
        stage='coordinate_warp_bilinear',
        scope='top23_1f_ref1',
        frames=bilinear_warp['frames'],
        rows_or_areas='23',
        bytes_per_frame=f(bilinear_warp, 'compressed_npz_bytes'),
        coverage_ratio=bilinear_warp['mean_coverage_ratio'],
        sample_ratio=bilinear_warp['mean_sample_ratio'],
        head_score_max=bilinear_head['score_max'],
        pred_boxes=bilinear_head['postprocess_pred_boxes'],
        ap_05=bilinear_ap['ap_05'],
        ap_07=bilinear_ap['ap_07'],
        paper_safe_use='negative AP boundary for uncalibrated bilinear warp',
        source=paths['bilinear_ap_summary'])

    csv_path = os.path.join(args.output_dir,
                            'neural_feature_proxy_summary.csv')
    write_csv(csv_path, list(rows[0].keys()), rows)

    config = dict(paths)
    config['raw_member_area23_bytes_per_frame'] = raw_member_bytes
    config['comm_aware_area23_slice_bytes_per_frame'] = comm_area_bytes
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    with open(os.path.join(args.output_dir, 'notes.md'), 'w') as stream:
        stream.write('# LGCP Neural Feature Proxy Summary\n\n')
        stream.write('This table is a paper-boundary artifact. It separates ')
        stream.write('feature-path feasibility from AP claims.\n\n')
        stream.write('- raw member area-slice bytes/frame: `%.6f`\n' %
                     raw_member_bytes)
        stream.write('- comm-aware flat area-slice bytes/frame: `%.6f`\n' %
                     comm_area_bytes)
        stream.write('- PointPillar feature crop compressed bytes/frame: `%s`\n' %
                     feature['compressed_npz_bytes'])
        stream.write('- bilinear warp AP@0.5/AP@0.7: `%s / %s`\n' %
                     (bilinear_ap['ap_05'], bilinear_ap['ap_07']))
        stream.write('\nThe current uncalibrated neural feature path should be ')
        stream.write('used as coverage/byte feasibility evidence only.\n')

    print('Wrote LGCP neural feature proxy summary to %s' % args.output_dir)
    print('feature_crop_ratio_vs_raw=%.6f bilinear_ap05=%s' % (
        f(feature, 'compressed_npz_bytes') / raw_member_bytes,
        bilinear_ap['ap_05']))


if __name__ == '__main__':
    main()
