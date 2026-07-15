# -*- coding: utf-8 -*-
"""
Utilities for loading OPV2V-style dumped frames without running CARLA.
"""

from collections import OrderedDict
import os
import sys

import numpy as np
import yaml

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))
_OPENCOOD_ROOT = os.path.join(_REPO_ROOT, 'opencood')
if _OPENCOOD_ROOT not in sys.path:
    sys.path.insert(0, _OPENCOOD_ROOT)

from opencood.utils.transformation_utils import x1_to_x2
try:
    from opencood.utils.pcd_utils import pcd_to_np
except ModuleNotFoundError:
    def pcd_to_np(pcd_file):
        """Read an ASCII PCD file without requiring open3d."""
        with open(pcd_file, 'r') as pcd:
            lines = pcd.readlines()

        data_start = None
        for index, line in enumerate(lines):
            if line.strip().lower().startswith('data'):
                data_start = index + 1
                break
        if data_start is None:
            raise ValueError('PCD DATA header not found: %s' % pcd_file)

        points = np.loadtxt(lines[data_start:], dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] < 4:
            intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
            points = np.hstack((points[:, :3], intensity))
        return points[:, :4].astype(np.float32)


class OPV2VFrameDataset(object):
    """
    Load a CARLA/OpenCDA data dump laid out like OPV2V.

    Expected layout:
        root/scenario_id/data_protocol.yaml
        root/scenario_id/cav_id/000000.pcd
        root/scenario_id/cav_id/000000.yaml
    """

    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self.scenarios = self._discover_scenarios()

    def _discover_scenarios(self):
        scenarios = OrderedDict()
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(self.root_dir)

        for scenario_id in sorted(os.listdir(self.root_dir)):
            scenario_path = os.path.join(self.root_dir, scenario_id)
            if not os.path.isdir(scenario_path):
                continue

            cav_ids = [
                name for name in sorted(os.listdir(scenario_path))
                if os.path.isdir(os.path.join(scenario_path, name))
            ]
            if not cav_ids:
                continue

            timestamps = None
            for cav_id in cav_ids:
                cav_path = os.path.join(scenario_path, cav_id)
                cav_timestamps = {
                    os.path.splitext(name)[0]
                    for name in os.listdir(cav_path)
                    if name.endswith('.yaml')
                }
                timestamps = cav_timestamps if timestamps is None else \
                    timestamps.intersection(cav_timestamps)

            scenarios[scenario_id] = {
                'path': scenario_path,
                'cav_ids': cav_ids,
                'timestamps': sorted(timestamps or [])
            }

        return scenarios

    def iter_frames(self, scenario_id=None):
        scenario_ids = [scenario_id] if scenario_id else self.scenarios.keys()
        for sid in scenario_ids:
            for timestamp in self.scenarios[sid]['timestamps']:
                yield sid, timestamp

    def load_frame(self, scenario_id, timestamp, ego_cav_id=None,
                   cav_ids=None, add_transformation=True):
        scenario = self.scenarios[scenario_id]
        selected_cav_ids = cav_ids or scenario['cav_ids']
        ego_cav_id = str(ego_cav_id or selected_cav_ids[0])
        ego_lidar_pose = None
        frame = OrderedDict()

        for cav_id in selected_cav_ids:
            cav_id = str(cav_id)
            cav_path = os.path.join(scenario['path'], cav_id)
            yaml_path = os.path.join(cav_path, timestamp + '.yaml')
            pcd_path = os.path.join(cav_path, timestamp + '.pcd')
            if not os.path.exists(yaml_path) or not os.path.exists(pcd_path):
                continue

            params = self._load_yaml(yaml_path)
            is_ego = cav_id == ego_cav_id
            if is_ego:
                ego_lidar_pose = params['lidar_pose']

            frame[self._normalize_cav_id(cav_id)] = OrderedDict({
                'ego': is_ego,
                'time_delay': 0,
                'params': params,
                'lidar_np': pcd_to_np(pcd_path)
            })

        if add_transformation:
            if ego_lidar_pose is None:
                raise ValueError(
                    'ego_cav_id %s is missing from frame %s/%s' %
                    (ego_cav_id, scenario_id, timestamp))
            for cav_content in frame.values():
                cav_pose = cav_content['params']['lidar_pose']
                cav_content['params']['transformation_matrix'] = \
                    x1_to_x2(cav_pose, ego_lidar_pose)
                cav_content['params']['gt_transformation_matrix'] = \
                    cav_content['params']['transformation_matrix']
                cav_content['params']['spatial_correction_matrix'] = \
                    x1_to_x2(ego_lidar_pose, ego_lidar_pose)

        return frame

    @staticmethod
    def _normalize_cav_id(cav_id):
        try:
            return int(cav_id)
        except ValueError:
            return cav_id

    @staticmethod
    def _load_yaml(yaml_path):
        with open(yaml_path, 'r') as stream:
            return yaml.load(stream, Loader=yaml.Loader)
