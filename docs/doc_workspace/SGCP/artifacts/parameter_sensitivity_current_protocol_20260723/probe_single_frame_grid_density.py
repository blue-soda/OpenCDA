# -*- coding: utf-8 -*-
"""Probe one frame/CAV LiDAR grid-density distribution."""

import argparse

import numpy as np

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import OfflineCavWorld
from opencda.tools.offline_inference import load_protocol


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default=r'D:\Data\Carla')
    parser.add_argument('--scenario-id', default='2026_07_15_01_26_56')
    parser.add_argument('--timestamp', default='000060')
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--cav-ids', default='1,4,11,13,15')
    parser.add_argument('--thresholds', default='0.5,1,2,3,5,8,10,15,20')
    return parser.parse_args()


def threshold_values(text):
    return [float(item.strip()) for item in text.split(',') if item.strip()]


def summarize(vid, lidar, thresholds):
    densities = np.asarray(list(lidar.grid_density_dict.values()),
                           dtype=np.float64)
    nonzero = densities[densities > 0]
    local_points = getattr(lidar, 'local_data').shape[0]
    print('\nCAV %s' % vid)
    print('local_points=%d grid_total=%d nonzero_grids=%d nonzero_ratio=%.4f' %
          (local_points, len(densities), len(nonzero),
           len(nonzero) / float(max(len(densities), 1))))
    if len(nonzero) == 0:
        return
    print('all_density mean=%.4f p95=%.4f p99=%.4f max=%.4f' % (
        float(np.mean(densities)),
        float(np.percentile(densities, 95)),
        float(np.percentile(densities, 99)),
        float(np.max(densities))))
    print('nonzero_density mean=%.4f p50=%.4f p75=%.4f p90=%.4f '
          'p95=%.4f p99=%.4f max=%.4f' % (
              float(np.mean(nonzero)),
              float(np.percentile(nonzero, 50)),
              float(np.percentile(nonzero, 75)),
              float(np.percentile(nonzero, 90)),
              float(np.percentile(nonzero, 95)),
              float(np.percentile(nonzero, 99)),
              float(np.max(nonzero))))
    print('| rho_th | high grids | high / nonzero |')
    print('| ---: | ---: | ---: |')
    for threshold in thresholds:
        high = int(np.sum(densities >= threshold))
        print('| %g | %d | %.4f |' %
              (threshold, high, high / float(max(len(nonzero), 1))))
    print('top20 grid densities:')
    for grid_id, density in sorted(
            lidar.grid_density_dict.items(),
            key=lambda item: item[1],
            reverse=True)[:20]:
        if density <= 0:
            break
        print('  %s: %.4f' % (grid_id, density))


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)
    protocol = load_protocol(dataset, args.scenario_id)
    frame = dataset.load_frame(
        args.scenario_id,
        args.timestamp,
        ego_cav_id=args.ego_cav_id)
    world = OfflineCavWorld(frame, ego_id=args.ego_cav_id,
                            protocol=protocol)
    thresholds = threshold_values(args.thresholds)
    print('scenario=%s timestamp=%s ego=%s thresholds=%s' % (
        args.scenario_id,
        args.timestamp,
        args.ego_cav_id,
        ','.join('%g' % item for item in thresholds)))
    for cav_id in [item.strip() for item in args.cav_ids.split(',')
                   if item.strip()]:
        vm = world.get_vehicle_managers().get(cav_id)
        if vm is None:
            try:
                vm = world.get_vehicle_managers().get(int(cav_id))
            except ValueError:
                vm = None
        if vm is None:
            print('\nCAV %s not found' % cav_id)
            continue
        summarize(cav_id, vm.perception_manager.lidar, thresholds)


if __name__ == '__main__':
    main()
