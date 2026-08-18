# -*- coding: utf-8 -*-
"""
LGCP RSU wrapper for V2X-ViT compressed leader features.

The input is already the leader-to-RSU feature packet used by LGCP:
PointPillar backbone -> shrink -> NaiveCompressor encoder. This wrapper
decodes the packet, adds an explicit RSU/global query when requested, reuses
V2XTransformer for aggregation, then applies the original detection heads.
"""

import torch
import torch.nn as nn

from opencood.models.fuse_modules.v2xvit_basic import V2XTransformer
from opencood.models.sub_modules.naive_compress import NaiveCompressor


class LgcpV2XViTRsu(nn.Module):
    def __init__(self, args):
        super(LgcpV2XViTRsu, self).__init__()
        self.max_cav = args.get('max_cav', 5)
        self.query_mode = args.get('query_mode', 'learnable_channel')
        if self.query_mode not in ('input', 'mean', 'zero',
                                   'learnable_channel'):
            raise ValueError('Unsupported query_mode: %s' % self.query_mode)

        self.compression = args.get('compression', 0) > 0
        if self.compression:
            self.naive_compressor = NaiveCompressor(256, args['compression'])
        self.fusion_net = V2XTransformer(args['transformer'])
        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)
        if self.query_mode == 'learnable_channel':
            self.rsu_query_channel = nn.Parameter(torch.zeros(1, 256, 1, 1))

    def forward(self, data_dict):
        if 'decoded_features' in data_dict:
            leader_features = data_dict['decoded_features']
        elif 'compressed_features' in data_dict:
            if not self.compression:
                leader_features = data_dict['compressed_features']
            else:
                leader_features = self.naive_compressor.decoder(
                    data_dict['compressed_features'])
        else:
            raise KeyError('Expected decoded_features or compressed_features.')

        leader_record_len = data_dict.get('leader_record_len')
        if leader_record_len is None:
            leader_record_len = torch.tensor(
                [leader_features.shape[0]],
                dtype=torch.int64,
                device=leader_features.device)
        split_features = torch.split(
            leader_features,
            [int(item) for item in leader_record_len.detach().cpu().tolist()],
            dim=0)

        stacks = []
        prior_rows = []
        record_len = []
        for features in split_features:
            stack, prior = self.build_query_stack(features)
            stacks.append(stack)
            prior_rows.append(prior)
            record_len.append(stack.shape[0])

        max_len = max(record_len)
        padded_stacks = []
        padded_priors = []
        mask_rows = []
        for stack, prior, length in zip(stacks, prior_rows, record_len):
            pad = max_len - length
            if pad:
                stack = torch.cat(
                    [stack, torch.zeros(
                        pad,
                        stack.shape[1],
                        stack.shape[2],
                        stack.shape[3],
                        dtype=stack.dtype,
                        device=stack.device)],
                    dim=0)
                prior = torch.cat(
                    [prior, torch.zeros(
                        pad,
                        prior.shape[1],
                        dtype=prior.dtype,
                        device=prior.device)],
                    dim=0)
            padded_stacks.append(stack)
            padded_priors.append(prior)
            mask_rows.append([1] * length + [0] * pad)

        feature = torch.stack(padded_stacks, dim=0)
        prior_encoding = torch.stack(padded_priors, dim=0)
        mask = torch.as_tensor(mask_rows, dtype=torch.bool,
                               device=leader_features.device)
        spatial_correction_matrix = data_dict.get('spatial_correction_matrix')
        if spatial_correction_matrix is None:
            eye = torch.eye(
                4, dtype=leader_features.dtype,
                device=leader_features.device)
            spatial_correction_matrix = eye.view(1, 1, 4, 4).repeat(
                feature.shape[0], max_len, 1, 1)

        prior = prior_encoding.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, 1, feature.shape[3], feature.shape[4])
        fusion_input = torch.cat([feature, prior], dim=2)
        fusion_input = fusion_input.permute(0, 1, 3, 4, 2)
        fused_feature = self.fusion_net(
            fusion_input,
            mask,
            spatial_correction_matrix)
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        return {
            'psm': self.cls_head(fused_feature),
            'rm': self.reg_head(fused_feature),
            'fused_feature': fused_feature,
        }

    def build_query_stack(self, features):
        prior = torch.zeros(
            features.shape[0],
            3,
            dtype=features.dtype,
            device=features.device)
        if self.query_mode == 'input':
            return features, prior
        if self.query_mode == 'mean':
            query = torch.mean(features, dim=0, keepdim=True)
        elif self.query_mode == 'zero':
            query = torch.zeros_like(features[:1])
        elif self.query_mode == 'learnable_channel':
            query = self.rsu_query_channel.expand(
                1, -1, features.shape[2], features.shape[3])
        else:
            raise ValueError('Unsupported query_mode: %s' % self.query_mode)
        query_prior = torch.zeros(
            1, 3, dtype=features.dtype, device=features.device)
        query_prior[:, 2] = 1.0
        return torch.cat([query, features], dim=0), torch.cat(
            [query_prior, prior], dim=0)


# Keep the short name usable for explicit imports in research scripts.
LgcpV2XRsu = LgcpV2XViTRsu
