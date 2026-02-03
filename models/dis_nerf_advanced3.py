# models/dis_nerf_advanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_content_encoder, get_distortion_encoder
from .mi_estimator import MIEstimator

class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块
    """
    def __init__(self, feature_dim=768):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat_c, feat_d):
        combined = torch.cat([feat_c, feat_d], dim=1)
        weights = self.attn_fc(combined) # [B, 2]
        
        alpha = weights[:, 0].unsqueeze(1) 
        beta = weights[:, 1].unsqueeze(1)  
        
        feat_c_weighted = feat_c * (1 + alpha) 
        feat_d_weighted = feat_d * (1 + beta)
        
        return torch.cat([feat_c_weighted, feat_d_weighted], dim=1)

class DisNeRFQA_Advanced(nn.Module):
    # [修正] 接受消融实验的开关参数
    def __init__(self, num_subscores=4, use_fusion=True, use_multitask=True, use_decoupling=True):
        super().__init__()
        self.use_fusion = use_fusion
        self.use_multitask = use_multitask
        self.use_decoupling = use_decoupling
        
        # 1. Backbones
        self.content_encoder = get_content_encoder(pretrained=True)
        self.distortion_encoder = get_distortion_encoder(pretrained=True)
        self.feat_dim = 768 
        
        # 2. MI Estimator (开关控制)
        self.mi_estimator = None
        if self.use_decoupling:
            self.mi_estimator = MIEstimator(self.feat_dim)
        
        # 3. Adaptive Fusion (开关控制)
        if self.use_fusion:
            self.fusion = AdaptiveFeatureFusion(self.feat_dim)
        
        # 4. Heads
        # Main Quality Regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        
        # [修正] Sub-score Head 
        self.subscore_head = None
        if self.use_multitask:
            # 关键修正：输入维度改为 2倍 (feat_dim * 2)，因为使用融合特征
            self.subscore_head = nn.Sequential(
                nn.Linear(self.feat_dim * 2, 256), 
                nn.ReLU(),
                nn.Linear(256, num_subscores),
                nn.Sigmoid()
            )
        
        # Contrastive Projectors
        self.proj_c = nn.Sequential(nn.Linear(self.feat_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.proj_d = nn.Sequential(nn.Linear(self.feat_dim, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, x_content, x_distortion):
        b, t, c, h, w = x_content.shape
        
        # --- Feature Extraction ---
        x_c_flat = x_content.view(b * t, c, h, w)
        x_d_flat = x_distortion.view(b * t, c, h, w)
        
        # Content Branch
        feat_c_raw = self.content_encoder.forward_features(x_c_flat)
        if hasattr(self.content_encoder, 'global_pool'):
             feat_c_seq = self.content_encoder.forward_head(feat_c_raw, pre_logits=True)
        else:
             feat_c_seq = feat_c_raw[:, 0]
             
        # Distortion Branch
        feat_d_raw = self.distortion_encoder.forward_features(x_d_flat)
        feat_d_seq = feat_d_raw.mean(dim=[1, 2])
        
        feat_c = feat_c_seq.view(b, t, -1).mean(dim=1)
        feat_d = feat_d_seq.view(b, t, -1).mean(dim=1)
        
        # --- Adaptive Fusion ---
        if self.use_fusion and hasattr(self, 'fusion'):
            feat_fused = self.fusion(feat_c, feat_d)
        else:
            feat_fused = torch.cat([feat_c, feat_d], dim=1)
        
        # --- Predictions ---
        score = self.regressor(feat_fused)
        
        # [修正] 多任务预测使用 Fused Feature
        sub_scores = None
        if self.use_multitask and self.subscore_head is not None:
            sub_scores = self.subscore_head(feat_fused) # 使用融合特征
        else:
            sub_scores = torch.zeros(b, 4).to(score.device)
        
        proj_c = self.proj_c(feat_c)
        proj_d = self.proj_d(feat_d)
        
        return score, sub_scores, proj_c, proj_d, feat_c, feat_d
