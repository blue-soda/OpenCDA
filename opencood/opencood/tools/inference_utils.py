# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_late_fusion(batch_data, model, dataset, return_output=False, return_object_ids=False):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        # print(cav_id, cav_content.keys())
        output_dict[cav_id] = model(cav_content)

    if return_object_ids:
        post_process_result = dataset.post_process(
            batch_data,
            output_dict,
            return_object_ids=return_object_ids)
    else:
        post_process_result = dataset.post_process(batch_data, output_dict)
    ret = list(post_process_result)
    if return_output:
        ret.append(output_dict)

    return tuple(ret)


def inference_early_fusion(batch_data, model, dataset, return_output=False, return_object_ids=False):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    post_process_result = dataset.post_process(batch_data, output_dict, return_object_ids=return_object_ids)
    ret = list(post_process_result)
    if return_output:
        ret.append(output_dict)

    return tuple(ret)


def inference_intermediate_fusion(batch_data, model, dataset, return_output=False):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return inference_early_fusion(batch_data, model, dataset, return_output)


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy_test' % timestamp), gt_np)
