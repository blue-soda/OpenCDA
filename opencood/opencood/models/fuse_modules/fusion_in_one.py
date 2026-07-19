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


class Where2comm(nn.Module):
    def __init__(self, args, dim):
        super(Where2comm, self).__init__()
        del dim
        self.fully = args['fully']
        if args['fusion'] != 'max':
            raise NotImplementedError(
                'SGCP COSDH probe only vendors Where2comm max fusion.')
        self.fuse_modules = MaxFusion()
        self.naive_communication = Communication(args['communication'])

    def regroup(self, x, record_len):
        return regroup(x, record_len)

    def forward(self, x, psm_single, record_len, normalized_affine_matrix,
                req_mask=None, external_comm_mask=None,
                external_comm_recon=None):
        if external_comm_mask is not None or external_comm_recon is not None:
            raise NotImplementedError(
                'External communication masks are not used by SGCP COSDH '
                'checkpoint probe.')

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
