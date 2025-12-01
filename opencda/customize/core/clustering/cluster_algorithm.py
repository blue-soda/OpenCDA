# cluster_algorithm.py
import math
import random
from opencda.core.common.misc import compute_distance

def calculate_cos(direction1, direction2):
    """计算两方向向量的余弦相似度"""
    dot_product = direction1[0]*direction2[0] + direction1[1]*direction2[1] + direction1[2]*direction2[2]
    magnitude1 = math.sqrt(sum(d**2 for d in direction1))
    magnitude2 = math.sqrt(sum(d**2 for d in direction2))
    return dot_product / (magnitude1 * magnitude2 + 1e-4)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + math.exp(-x))

class ClusterAlgorithm:
    @staticmethod
    def compute_spatiotemporal_similarity(ego_data, neighbor_data, params):
        """计算时空相似性"""
        # 空间距离项（含航向角调整）
        distance = compute_distance(ego_data['position'], neighbor_data['position'])
        cos_theta = calculate_cos(ego_data['direction'], neighbor_data['direction'])
        distance_term = math.exp(-distance / params['d0']) * cos_theta
        
        # 运动状态项（速度差异）
        speed_diff = abs(ego_data['speed'] - neighbor_data['speed'])
        speed_term = math.exp(-speed_diff / params['s0'])
        
        # 动态权重
        alpha = 1 / (1 + math.exp(-params['kappa'] * (ego_data['neighbor_count'] - params['N_th'])))
        beta = 1 - alpha
        
        # 模型兼容性惩罚
        if ego_data['perception_model'] != neighbor_data['perception_model']:
            similarity = (alpha * distance_term + beta * speed_term) * 0.01
        else:
            similarity = alpha * distance_term + beta * speed_term
        
        return similarity

    @staticmethod
    def calculate_create_probability(avg_similarity, params):
        """计算创建新簇的概率"""
        return sigmoid((params['eta_create'] - avg_similarity) / params['sigma'])

    @staticmethod
    def compute_priority_score(ego_data, cluster_avg_speed, params):
        """计算簇头优先级得分"""
        speed_consistency = 1.5 - sigmoid(abs(ego_data['speed'] - cluster_avg_speed))
        return (
            params['w1'] * ego_data['communication_quality'] +
            params['w2'] * ego_data['computing_capability'] +
            params['w3'] * speed_consistency
        )