# -*- coding: utf-8 -*-
import os
import torch
import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.visualization import vis_utils
from opencood.tools import train_utils, inference_utils
from opencood.utils import eval_utils
from opencood.utils.eval_utils import calculate_ap
from opencood.data_utils.datasets import build_dataset
from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencda.log.logger_config import logger
import torchvision.ops as ops

DATASET_DICT = {
    'late': LateFusionDataset,
    'early': EarlyFusionDataset,
    'intermediate': IntermediateFusionDataset,
}


class OpenCOODManager(object):
    def __init__(self, coperception_params):
        fusion_method = coperception_params['fusion_method']
        models = coperception_params['models']
        assert fusion_method in models, f'Fusion method should be within one of the models supported, it is provided ' \
                                        f'with {fusion_method} '
        self.counter = 0
        self.fusion_method = fusion_method
        self.opt = argparse.Namespace(model_dir=models[fusion_method])
        hypes = yaml_utils.load_yaml(None, self.opt)
        self.model = train_utils.create_model(hypes)
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        saved_path = models[fusion_method]
        _, model = train_utils.load_saved_model(saved_path, self.model)
        model.eval()

        # if self.fusion_method == 'late':
        #     self.late_fusion_model = self.model
        # else:       
        #     late_fusion_opt = argparse.Namespace(model_dir=models['late'])         
        #     hypes = yaml_utils.load_yaml(None, late_fusion_opt)         
        #     self.late_fusion_model = train_utils.create_model(hypes)         
        #     if torch.cuda.is_available():             
        #         self.late_fusion_model.cuda()         
        #     saved_path = models[fusion_method]        
        #     _, late_fusion_model = train_utils.load_saved_model(saved_path, self.late_fusion_model)         
        #     late_fusion_model.eval()

        self.opencood_dataset = build_dataset(hypes, visualize=True, train=False)

        # Create the dictionary for evaluation
        self.result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                            0.5: {'tp': [], 'fp': [], 'gt': 0},
                            0.7: {'tp': [], 'fp': [], 'gt': 0}}

    def to_device(self, data):
        return train_utils.to_device(data, self.device)

    def submit_results(self, pred_box_tensor, pred_score, gt_box_tensor, with_stats=True):
        if not with_stats:
            return
        
        self.counter += 1

        if self.counter <= 5:
            return
    
        logger.debug('submit_results')
        eval_utils.calculate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat,
                                    0.3)
        eval_utils.calculate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat,
                                    0.5)
        eval_utils.calculate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    self.result_stat,
                                    0.7)
        
    def inference(self, batch_data, with_stats=True, fusion_method='default', return_object_ids=False, target_ids=None):

        if fusion_method == 'default':
            fusion_method = self.fusion_method

        if fusion_method == 'late':
            # pred_box_tensor, pred_score, gt_box_tensor = \
            ret = \
                inference_utils.inference_late_fusion(batch_data,
                                                      self.model,
                                                      self.opencood_dataset,
                                                      return_output=False,
                                                      return_object_ids=return_object_ids,
                                                      )
        elif fusion_method == 'early':
            # pred_box_tensor, pred_score, gt_box_tensor = \
            ret = \
                inference_utils.inference_early_fusion(batch_data,
                                                       self.model,
                                                       self.opencood_dataset,
                                                       return_output=False,
                                                       return_object_ids=return_object_ids,
                                                       )
        elif fusion_method.startswith('intermediate'): # intermediate would be different models
            # pred_box_tensor, pred_score, gt_box_tensor = \
            ret = \
                inference_utils.inference_intermediate_fusion(batch_data,
                                                              self.model,
                                                              self.opencood_dataset,
                                                              return_output=False,
                                                              return_object_ids=return_object_ids,
                                                            )
        else:
            raise NotImplementedError('Only early, late and intermediate'
                                      'fusion is supported.')

        # skip the first 60 ticks for calculating the average precision
        if with_stats: #and self.counter % 2 == 0:
            logger.debug(f"Aggregating the current stats into final results: {self.counter}")
            pred_box_tensor, pred_score, gt_box_tensor = ret[0:3]
            self.submit_results(pred_box_tensor, pred_score, gt_box_tensor, with_stats)
        return ret

    def evaluate_final_average_precision(self):
        print(f'cp counter: {self.counter}')
        print('Evaluate final average precision results:')
        print(f'  - Fusion method: {self.fusion_method}')
        ap_30, mrec_30, mpre_30 = calculate_ap(self.result_stat, 0.30)
        ap_50, mrec_50, mpre_50 = calculate_ap(self.result_stat, 0.50)
        ap_70, mrec_70, mpre_70 = calculate_ap(self.result_stat, 0.70)
        print('  - The Average Precision at IOU 0.3 is %.2f, '
              'The Average Precision at IOU 0.5 is %.2f, '
              'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))
        
        logger.info('  - The Average Precision at IOU 0.3 is %.2f, '
              'The Average Precision at IOU 0.5 is %.2f, '
              'The Average Precision at IOU 0.7 is %.2f' % (ap_30, ap_50, ap_70))

    def show_vis(self, pred_box_tensor, gt_box_tensor, batch_data):
        vis_save_path = os.path.join(self.opt.model_dir, 'vis')
        if not os.path.exists(vis_save_path):
            os.makedirs(vis_save_path)
        vis_save_path = os.path.join(vis_save_path, '%05d.png' % self.counter)

        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'],
                                                    True,
                                                    vis_save_path,
                                                    mode='constant')

    @staticmethod
    def all_boxes(pred_box_tensors, pred_scores, gt_box_tensors):
        # If no predicted boxes are provided, return None for all outputs
        if len(pred_box_tensors) == 0 or len(gt_box_tensors) == 0:
            return None, None, None
        
        # Concatenate all predicted boxes, scores, and ground truth boxes
        all_predict_boxes = torch.cat(pred_box_tensors, dim=0)  # Shape: [N, 8, 3]
        all_predict_scores = torch.cat(pred_scores, dim=0)       # Shape: [N]
        all_gt_boxes = torch.cat(gt_box_tensors, dim=0)         # Shape: [M, 8, 3]
        return all_predict_boxes, all_predict_scores, all_gt_boxes
    
    @staticmethod
    def naive_late_fusion(pred_box_tensors, pred_scores=None, iou_threshold=0.15):
        # If no predicted boxes are provided, return None for all outputs
        if len(pred_box_tensors) == 0:
            return None, None, None

        # Concatenate all predicted boxes, scores, and ground truth boxes
        all_predict_boxes = torch.cat(pred_box_tensors, dim=0)  # Shape: [N, 8, 3]
        if pred_scores is None:
            all_predict_scores = torch.ones(all_predict_boxes.shape[0], device=all_predict_boxes.device)  # Shape: [N]
        else:
            # 检查并转换 pred_scores 中的元素
            processed_scores = []
            for score in pred_scores:
                if not isinstance(score, torch.Tensor):
                    # 将非张量元素转换为张量
                    score = torch.tensor(score, dtype=torch.float32).to(all_predict_boxes.device)
                if score.dim() == 0:  # 如果是零维张量（标量）
                    score = score.unsqueeze(0)  # 添加一个维度变为 [1]
                processed_scores.append(score)
            all_predict_scores = torch.cat(processed_scores, dim=0)       # Shape: [N]

        # Extract the 2D bounding box coordinates for NMS (ignore the z-axis)
        min_coords, _ = torch.min(all_predict_boxes[:, :, :2], dim=1)  # Shape: [N, 2]
        max_coords, _ = torch.max(all_predict_boxes[:, :, :2], dim=1)  # Shape: [N, 2]
        boxes_for_nms = torch.cat([min_coords, max_coords], dim=1)     # Shape: [N, 4]
        
        # 确保 boxes_for_nms 是 [N, 4] 形状
        if boxes_for_nms.shape[1] != 4:
            raise ValueError(f"boxes_for_nms should have shape [N, 4], got {boxes_for_nms.shape}")
        
        # 确保 all_predict_scores 是 [N] 形状
        if all_predict_scores.shape[0] != boxes_for_nms.shape[0]:
            raise ValueError(f"all_predict_scores should have shape [N], got {all_predict_scores.shape}")

        # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
        keep_indices = ops.nms(boxes_for_nms, all_predict_scores, iou_threshold=iou_threshold)
        
        # Filter the predicted boxes and scores based on NMS results
        filtered_predict_boxes = all_predict_boxes[keep_indices]  # Shape: [K, 8, 3]
        filtered_predict_scores = all_predict_scores[keep_indices]  # Shape: [K]

        return filtered_predict_boxes, filtered_predict_scores



