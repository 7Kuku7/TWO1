# models/mi_estimator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIEstimator(nn.Module):
    """
    [修复版] 特征解耦模块
    之前版本使用 MINE 估计互信息，在单次优化中容易导致数值爆炸(Loss -> -inf)。
    这里改为使用 '余弦相似度惩罚' (Cosine Similarity Penalty)。
    
    目标：最小化 Content 和 Distortion 特征之间的相似度，使它们正交(无关)。
    """
    def __init__(self, feature_dim=768):
        super(MIEstimator, self).__init__()
        # 不需要额外的神经网络参数，直接计算几何距离，更稳定
        
    def forward(self, feat_c, feat_d):
        """
        feat_c: [B, D]
        feat_d: [B, D]
        """
        # 计算余弦相似度 [-1, 1]
        cosine = F.cosine_similarity(feat_c, feat_d, dim=1)
        
        # 我们希望相似度为 0 (正交)
        # Loss = mean( |cosine| )
        loss_decouple = torch.mean(torch.abs(cosine))
        
        return loss_decouple
