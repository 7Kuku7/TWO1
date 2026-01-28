# models/mi_estimator.py
import torch
import torch.nn as nn

class MIEstimator(nn.Module):
    """
    互信息估计器 (Mutual Information Estimator)
    用于解耦 Content 和 Distortion 特征。
    我们要最小化 I(Content, Distortion)。
    """
    def __init__(self, feature_dim=768, hidden_dim=256):
        super(MIEstimator, self).__init__()
        # 这是一个基于变分下界估计的简单的判别器网络
        # 输入：[feat_c, feat_d]
        # 输出：标量（估计的互信息值）
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_c, feat_d):
        """
        feat_c: Content 特征 [B, D]
        feat_d: Distortion 特征 [B, D]
        """
        # 1. 联合分布样本 (Joint): (c, d) 也就是配对的数据
        joint = torch.cat([feat_c, feat_d], dim=1)
        
        # 2. 边缘分布样本 (Marginal): (c, d') 也就是乱序的数据
        # 通过在 batch 维度 shuffle feat_d 来破坏配对关系
        idx = torch.randperm(feat_d.size(0)).to(feat_d.device)
        feat_d_shuffle = feat_d[idx]
        marginal = torch.cat([feat_c, feat_d_shuffle], dim=1)
        
        # 3. 估计 MI
        # InfoNCE / DV-based estimator logic:
        # MI 越大，说明 c 和 d 越相关。解耦时我们要 minimize 这个输出。
        t_joint = self.net(joint)
        t_marginal = self.net(marginal)
        
        # MINE (Mutual Information Neural Estimation) loss formula
        # loss = -(mean(T_joint) - log(mean(exp(T_marginal))))
        # 但我们在外部通常直接用输出作为 loss，或者如下计算：
        mi_score = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)) + 1e-6)
        
        return mi_score
