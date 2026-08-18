# -*- coding: utf-8 -*-
"""
LGCP RSU BEV attentive aggregation wrapper.

This model starts from PointPillar scatter BEV features exported by LGCP
area-leader packets. It intentionally skips PillarVFE and PointPillarScatter:
the input is already a BEV scatter tensor.
"""

import torch
import torch.nn as nn

from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class LgcpRsuBevAttentive(nn.Module):
    def __init__(self, args):
        super(LgcpRsuBevAttentive, self).__init__()
        self.query_mode = args.get('query_mode', 'input')
        if self.query_mode not in (
                'input', 'mean', 'zero', 'learnable_channel'):
            raise ValueError('Unsupported query_mode: %s' % self.query_mode)

        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)
        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)
        if self.query_mode == 'learnable_channel':
            self.rsu_query_channel = nn.Parameter(torch.zeros(1, 64, 1, 1))

    def forward(self, data_dict):
        if self.query_mode == 'input':
            spatial_features = data_dict['spatial_features']
            record_len = data_dict['record_len']
        else:
            spatial_features, record_len = self.build_query_stack(data_dict)

        batch_dict = {
            'spatial_features': spatial_features,
            'record_len': record_len,
        }
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        return {'psm': psm, 'rm': rm}

    def build_query_stack(self, data_dict):
        leader_features = data_dict['leader_features']
        leader_record_len = data_dict['leader_record_len']
        split_features = torch.split(
            leader_features,
            [int(item) for item in leader_record_len.detach().cpu().tolist()],
            dim=0)
        stacks = []
        record_len = []
        for features in split_features:
            if self.query_mode == 'mean':
                query = torch.mean(features, dim=0, keepdim=True)
            elif self.query_mode == 'zero':
                query = torch.zeros_like(features[:1])
            elif self.query_mode == 'learnable_channel':
                query = self.rsu_query_channel.expand(
                    1,
                    -1,
                    features.shape[2],
                    features.shape[3])
            else:
                raise ValueError('Unsupported query_mode: %s' %
                                 self.query_mode)
            stack = torch.cat([query, features], dim=0)
            stacks.append(stack)
            record_len.append(stack.shape[0])
        return (
            torch.cat(stacks, dim=0),
            torch.tensor(record_len, dtype=torch.int64,
                         device=leader_features.device))
