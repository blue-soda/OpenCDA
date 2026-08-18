# -*- coding: utf-8 -*-
"""
Dataset helpers for LGCP RSU BEV sparse training samples.

The sample exporter stores sparse PointPillar scatter cells. This helper can
inspect sparse samples cheaply, or reconstruct dense BEV tensors when a
training wrapper needs the original AttBEVBackbone input.
"""

import csv
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.utils import box_utils


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def as_scalar(value):
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return value


class LGCPRSUBevSparseDataset(Dataset):
    """
    Read sparse RSU BEV training samples exported by
    ``opencda.tools.lgcp_rsu_bev_training_sample_export``.

    Parameters
    ----------
    root_dir : str
        Directory containing ``sample_manifest.csv`` and ``samples/*.npz``.
    post_processor : object, optional
        OpenCOOD postprocessor used to generate PointPillar label_dict. If not
        provided, samples expose GT boxes but no training label_dict.
    query_mode : str
        ``mean``, ``zero``, or ``first_leader``.
    return_dense : bool
        Reconstruct dense BEV tensors. Keep this false for light inspection.
    """

    def __init__(self, root_dir, post_processor=None, query_mode='mean',
                 return_dense=False):
        self.root_dir = root_dir
        self.manifest_path = os.path.join(root_dir, 'sample_manifest.csv')
        self.rows = read_csv(self.manifest_path)
        self.post_processor = post_processor
        self.query_mode = query_mode
        self.return_dense = return_dense
        if self.query_mode not in ('mean', 'zero', 'first_leader'):
            raise ValueError('Unsupported query_mode: %s' % self.query_mode)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample_path = os.path.join(
            self.root_dir,
            row['sample_file'].replace('/', os.sep))
        data = np.load(sample_path, allow_pickle=False)
        sample = OrderedDict({
            'scenario_id': str(as_scalar(data['scenario_id'])),
            'timestamp': str(as_scalar(data['timestamp'])),
            'reference_label': str(as_scalar(data['reference_label'])),
            'reference_pose': data['reference_pose'].astype(np.float32),
            'dense_shape': data['dense_shape'].astype(np.int32),
            'sparse_indices': data['sparse_indices'].astype(np.int64),
            'sparse_features': data['sparse_features'].astype(np.float32),
            'area_ids': data['area_ids'].astype(str),
            'leader_ids': data['leader_ids'].astype(str),
            'group_sizes': data['group_sizes'].astype(np.int64),
            'gt_boxes': data['gt_boxes'].astype(np.float32),
            'gt_scope': str(as_scalar(data['gt_scope'])),
            'sample_file': row['sample_file'],
        })
        sample['record_len_value'] = self.record_len_value(sample)
        if self.return_dense:
            leader_features = self.reconstruct_dense(sample)
            sample['leader_features'] = leader_features
            sample['spatial_features'] = self.make_query_stack(
                leader_features)
            sample['record_len'] = torch.tensor(
                [sample['spatial_features'].shape[0]],
                dtype=torch.int64)
        if self.post_processor is not None:
            sample['label_dict'] = self.generate_label_dict(sample)
            sample['anchor_box'] = self.post_processor.generate_anchor_box()
        return sample

    def record_len_value(self, sample):
        leaders = int(sample['dense_shape'][0])
        if self.query_mode == 'first_leader':
            return leaders
        return leaders + 1

    @staticmethod
    def reconstruct_dense(sample):
        dense_shape = tuple(int(item) for item in sample['dense_shape'])
        dense = np.zeros(dense_shape, dtype=np.float32)
        indices = sample['sparse_indices']
        values = sample['sparse_features']
        if indices.shape[0] > 0:
            dense[indices[:, 0], :, indices[:, 1], indices[:, 2]] = values
        return torch.from_numpy(dense)

    def make_query_stack(self, leader_features):
        if self.query_mode == 'first_leader':
            return leader_features
        if self.query_mode == 'mean':
            query = torch.mean(leader_features, dim=0, keepdim=True)
        else:
            query = torch.zeros_like(leader_features[:1])
        return torch.cat([query, leader_features], dim=0)

    def generate_label_dict(self, sample):
        max_num = int(self.post_processor.params.get('max_num', 100))
        order = self.post_processor.params.get('order', 'hwl')
        gt_corners = sample['gt_boxes']
        gt_centers = np.zeros((max_num, 7), dtype=np.float32)
        mask = np.zeros((max_num,), dtype=np.float32)
        if gt_corners.shape[0] > 0:
            centers = box_utils.corner_to_center(gt_corners, order=order)
            valid_count = min(max_num, centers.shape[0])
            gt_centers[:valid_count] = centers[:valid_count]
            mask[:valid_count] = 1
        anchor_box = self.post_processor.generate_anchor_box()
        return self.post_processor.generate_label(
            gt_box_center=gt_centers,
            anchors=anchor_box,
            mask=mask)

    def collate_batch(self, samples):
        if not samples:
            raise ValueError('Cannot collate an empty sample list.')
        if any('spatial_features' not in sample for sample in samples):
            raise ValueError('Set return_dense=True before collating.')
        spatial_features = torch.cat(
            [sample['spatial_features'] for sample in samples],
            dim=0)
        leader_features = torch.cat(
            [sample['leader_features'] for sample in samples],
            dim=0)
        record_len = torch.tensor(
            [sample['record_len_value'] for sample in samples],
            dtype=torch.int64)
        leader_record_len = torch.tensor(
            [int(sample['dense_shape'][0]) for sample in samples],
            dtype=torch.int64)
        output = {
            'ego': {
                'spatial_features': spatial_features,
                'leader_features': leader_features,
                'record_len': record_len,
                'leader_record_len': leader_record_len,
                'metadata': [
                    {
                        'scenario_id': sample['scenario_id'],
                        'timestamp': sample['timestamp'],
                        'sample_file': sample['sample_file'],
                    }
                    for sample in samples
                ],
            }
        }
        if self.post_processor is not None:
            label_dicts = [sample['label_dict'] for sample in samples]
            output['ego']['label_dict'] = self.post_processor.collate_batch(
                label_dicts)
        return output
