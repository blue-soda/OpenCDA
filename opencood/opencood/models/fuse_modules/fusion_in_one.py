# -*- coding: utf-8 -*-
"""Minimal fusion modules required by PointPillarCommMultiscale.

This file vendors the Where2comm path used by the COSDH checkpoint probe from
the sibling OpenCOOD workspace. Only the classes needed by
point_pillar_comm_multiscale are included to avoid pulling unrelated model
dependencies into OpenCDA.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from opencood.models.comm_modules.where2comm import Communication
try:
    from opencood.models.sub_modules.torch_transformation_utils import (
        warp_affine_simple,
    )
except ImportError:
    from opencood.models.sub_modules.torch_transformation_utils import (
        warp_affine as warp_affine_simple,
    )


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    return torch.tensor_split(x, cum_sum_len[:-1].cpu())


def _resize_external_mask(mask, height, width):
    if mask.shape[-2:] == (height, width):
        return mask
    return F.interpolate(mask, size=(height, width), mode='nearest')


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value, attn_bias=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if attn_bias is not None:
            score = score + attn_bias.to(device=score.device,
                                         dtype=score.dtype)
        attn = F.softmax(score, -1)
        return torch.bmm(attn, value)


class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix,
                use_warp_feature=True):
        _, _, height, width = x.shape
        split_x = regroup(x, record_len)
        out = []
        for batch_idx in range(pairwise_t_matrix.shape[0]):
            cav_count = record_len[batch_idx]
            tfm = pairwise_t_matrix[batch_idx][:cav_count, :cav_count, :, :]
            if use_warp_feature:
                neighbor_feature = warp_affine_simple(
                    split_x[batch_idx], tfm[0, :, :, :], (height, width))
            else:
                neighbor_feature = split_x[batch_idx]
            out.append(torch.max(neighbor_feature, dim=0)[0].unsqueeze(0))
        return torch.cat(out, dim=0)


class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix,
                use_warp_feature=True, external_comm_mask=None,
                external_attention_prior=None, attention_prior_scale=1.0):
        _, channels, height, width = xx.shape
        split_x = regroup(xx, record_len)
        split_mask = None
        if external_comm_mask is not None:
            external_comm_mask = _resize_external_mask(
                external_comm_mask.to(device=xx.device, dtype=xx.dtype),
                height,
                width)
            split_mask = regroup(external_comm_mask, record_len)

        split_attention_prior = None
        if external_attention_prior is not None:
            external_attention_prior = _resize_external_mask(
                external_attention_prior.to(device=xx.device, dtype=xx.dtype),
                height,
                width)
            split_attention_prior = regroup(external_attention_prior,
                                            record_len)

        out = []
        for batch_idx in range(normalized_affine_matrix.shape[0]):
            cav_count = record_len[batch_idx]
            tfm = normalized_affine_matrix[
                batch_idx][:cav_count, :cav_count, :, :]
            if use_warp_feature:
                x = warp_affine_simple(split_x[batch_idx], tfm[0, :, :, :],
                                       (height, width))
                if split_mask is not None:
                    neighbor_mask = warp_affine_simple(
                        split_mask[batch_idx], tfm[0, :, :, :],
                        (height, width))
                    x = x * neighbor_mask
                if split_attention_prior is not None:
                    key_prior = warp_affine_simple(
                        split_attention_prior[batch_idx], tfm[0, :, :, :],
                        (height, width))
                else:
                    key_prior = None
            else:
                x = split_x[batch_idx]
                if split_mask is not None:
                    x = x * split_mask[batch_idx]
                key_prior = (
                    split_attention_prior[batch_idx]
                    if split_attention_prior is not None else None)

            cav_num = x.shape[0]
            x = x.view(cav_num, channels, -1).permute(2, 0, 1)
            attn_bias = None
            if key_prior is not None:
                key_bias = key_prior.view(cav_num, -1).permute(1, 0).clone()
                key_bias[:, 0] = 0.0
                key_bias = key_bias * float(attention_prior_scale)
                attn_bias = key_bias.unsqueeze(1).expand(-1, cav_num, -1)
            h = self.att(x, x, x, attn_bias=attn_bias)
            h = h.permute(1, 2, 0).view(cav_num, channels, height, width)[0]
            out.append(h)
        return torch.stack(out)


class Where2comm(nn.Module):
    def __init__(self, args, dim):
        super(Where2comm, self).__init__()
        self.fully = args['fully']
        self.external_ego_full = args.get('external_ego_full', False)
        self.external_rate_exclude_ego = args.get(
            'external_rate_exclude_ego', False)
        self.external_mask_mode = args.get('external_mask_mode', 'replace')
        if args['fusion'] == 'att':
            self.fuse_modules = AttFusion(dim)
        elif args['fusion'] == 'max':
            self.fuse_modules = MaxFusion()
        else:
            raise NotImplementedError(
                'Unsupported Where2comm fusion type: %s' % args['fusion'])
        self.naive_communication = Communication(args['communication'])

    def regroup(self, x, record_len):
        return regroup(x, record_len)

    def _prepare_external_comm_mask(self, external_comm_mask, record_len):
        communication_masks = external_comm_mask
        if self.external_ego_full:
            mask_groups = []
            for mask_group in self.regroup(communication_masks, record_len):
                if mask_group.shape[0] > 0:
                    mask_group = mask_group.clone()
                    mask_group[0] = 1
                mask_groups.append(mask_group)
            communication_masks = torch.cat(mask_groups, dim=0)

        if self.external_rate_exclude_ego or self.external_ego_full:
            rates = []
            for mask_group in self.regroup(communication_masks, record_len):
                if mask_group.shape[0] <= 1:
                    rates.append(communication_masks.new_tensor(0.0))
                else:
                    rates.append(mask_group[1:].sum() / mask_group[1:].numel())
            communication_rates = (
                torch.stack(rates).mean()
                if rates else communication_masks.new_tensor(0.0))
        else:
            communication_rates = (
                communication_masks.sum() / communication_masks.numel())

        return communication_masks, communication_rates

    def forward(self, x, psm_single, record_len, normalized_affine_matrix,
                req_mask=None, external_comm_mask=None,
                external_comm_recon=None):
        if external_comm_recon is not None:
            external_comm_recon = external_comm_recon.to(device=x.device,
                                                         dtype=x.dtype)
            if external_comm_recon.shape[1] != x.shape[1]:
                raise ValueError(
                    'external_comm_recon channels %d != feature channels %d' %
                    (int(external_comm_recon.shape[1]), int(x.shape[1])))
            if external_comm_recon.shape[-2:] != x.shape[-2:]:
                external_comm_recon = F.interpolate(
                    external_comm_recon,
                    size=x.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            x = external_comm_recon

        _, _, height, width = x.shape
        batch_node_features = self.regroup(x, record_len)
        batch_confidence_maps = self.regroup(psm_single, record_len)
        batch_warp_x = []
        batch_warp_confidence_maps = []
        batch_warp_masks = []

        for batch_idx in range(normalized_affine_matrix.shape[0]):
            cav_count = record_len[batch_idx]
            tfm = normalized_affine_matrix[
                batch_idx][:cav_count, :cav_count, :, :]
            confidence_map = batch_confidence_maps[batch_idx]
            warp_mask = torch.ones(
                (confidence_map.shape[0], 1, confidence_map.shape[2],
                 confidence_map.shape[3]),
                device=x.device)
            batch_warp_masks.append(
                warp_affine_simple(warp_mask, tfm[0, :, :, :],
                                   (height, width)))
            batch_warp_confidence_maps.append(
                warp_affine_simple(confidence_map, tfm[0, :, :, :],
                                   (height, width)))
            batch_warp_x.append(
                warp_affine_simple(batch_node_features[batch_idx],
                                   tfm[0, :, :, :], (height, width)))

        warp_x = torch.cat(batch_warp_x, dim=0)
        if external_comm_mask is not None:
            external_comm_mask = _resize_external_mask(
                external_comm_mask.to(device=x.device, dtype=x.dtype),
                height,
                width)
            if self.external_mask_mode == 'intersection':
                internal_masks, _ = self.naive_communication(
                    batch_warp_confidence_maps,
                    normalized_affine_matrix.shape[0],
                    batch_warp_masks,
                    req_mask)
                external_masks, _ = self._prepare_external_comm_mask(
                    external_comm_mask, record_len)
                communication_masks, communication_rates = (
                    self._prepare_external_comm_mask(
                        internal_masks * external_masks, record_len))
            else:
                communication_masks, communication_rates = (
                    self._prepare_external_comm_mask(
                        external_comm_mask, record_len))
        else:
            communication_masks, communication_rates = self.naive_communication(
                batch_warp_confidence_maps,
                normalized_affine_matrix.shape[0],
                batch_warp_masks,
                req_mask)

        if self.fully:
            communication_masks = torch.tensor(1, device=warp_x.device)
        else:
            if warp_x.shape[-1] != communication_masks.shape[-1]:
                communication_masks = F.interpolate(
                    communication_masks,
                    size=(warp_x.shape[-2], warp_x.shape[-1]),
                    mode='bilinear',
                    align_corners=False)
            warp_x = warp_x * communication_masks

        x_out = self.fuse_modules(
            warp_x,
            record_len,
            normalized_affine_matrix,
            use_warp_feature=False)
        return x_out, communication_rates
