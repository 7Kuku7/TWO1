# 消融实验修改 布尔开关
import torch
import torch.nn as nn
from .backbone import get_content_encoder, get_distortion_encoder
from .mi_estimator import MIEstimator

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2), 
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat_c, feat_d):
        weights = self.attn_fc(torch.cat([feat_c, feat_d], dim=1))
        alpha = weights[:, 0].unsqueeze(1)
        beta = weights[:, 1].unsqueeze(1)
        return torch.cat([feat_c * (1 + alpha), feat_d * (1 + beta)], dim=1)

class DisNeRFQA_Advanced(nn.Module):
    def __init__(self, num_subscores=4, use_fusion=True, use_multitask=True, use_decoupling=True):
        super().__init__()
        self.use_fusion = use_fusion
        self.use_multitask = use_multitask
        self.use_decoupling = use_decoupling
        
        # 1. Backbones
        self.content_encoder = get_content_encoder(pretrained=True)
        self.distortion_encoder = get_distortion_encoder(pretrained=True)
        self.feat_dim = 768 
        
        # 2. MI Estimator & Projectors (Structural Ablation: Decoupling)
        if self.use_decoupling:
            self.mi_estimator = MIEstimator(self.feat_dim)
            self.proj_c = nn.Sequential(
                nn.Linear(self.feat_dim, 256), nn.ReLU(), nn.Linear(256, 128)
            )
            self.proj_d = nn.Sequential(
                nn.Linear(self.feat_dim, 256), nn.ReLU(), nn.Linear(256, 128)
            )
        else:
            self.mi_estimator = None
            self.proj_c = None
            self.proj_d = None
        
        # 3. Adaptive Fusion (Structural Ablation: Fusion)
        if self.use_fusion:
            self.fusion = AdaptiveFeatureFusion(self.feat_dim)
        else:
            self.fusion = None
        
        # 4. Heads
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid() 
        )
        
        # 5. Sub-score Head (Structural Ablation: Multi-task)
        if self.use_multitask:
            self.subscore_head = nn.Sequential(
                nn.Linear(self.feat_dim, 256), nn.ReLU(),
                nn.Linear(256, num_subscores), nn.Sigmoid()
            )
        else:
            self.subscore_head = None

    def forward(self, x_content, x_distortion):
        b, t, c, h, w = x_content.shape
        
        # --- Feature Extraction ---
        x_c_flat = x_content.view(b * t, c, h, w)
        x_d_flat = x_distortion.view(b * t, c, h, w)
        
        feat_c_raw = self.content_encoder.forward_features(x_c_flat)
        if hasattr(self.content_encoder, 'global_pool'):
             feat_c_seq = self.content_encoder.forward_head(feat_c_raw, pre_logits=True)
        else:
             feat_c_seq = feat_c_raw[:, 0]
             
        feat_d_raw = self.distortion_encoder.forward_features(x_d_flat)
        feat_d_seq = feat_d_raw.mean(dim=[1, 2])
        
        feat_c = feat_c_seq.view(b, t, -1).mean(dim=1)
        feat_d = feat_d_seq.view(b, t, -1).mean(dim=1)
        
        # --- Adaptive Fusion (Structural Switch) ---
        if self.use_fusion and self.fusion is not None:
            feat_fused = self.fusion(feat_c, feat_d)
        else:
            feat_fused = torch.cat([feat_c, feat_d], dim=1)
        
        # --- Predictions ---
        score = self.regressor(feat_fused)
        
        # --- Sub-scores (Structural Switch) ---
        if self.use_multitask and self.subscore_head is not None:
            sub_scores = self.subscore_head(feat_d)
        else:
            sub_scores = None # 返回 None，Solver 会识别并跳过 Loss 计算
        
        # --- Projections (Structural Switch) ---
        if self.use_decoupling and self.proj_c is not None:
            proj_c = self.proj_c(feat_c)
            proj_d = self.proj_d(feat_d)
        else:
            proj_c = None
            proj_d = None
        
        return score, sub_scores, proj_c, proj_d, feat_c, feat_d
