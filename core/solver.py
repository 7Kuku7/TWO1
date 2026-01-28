# core/solver.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import os
from utils import calculate_srcc, calculate_plcc, calculate_krcc

# 假设 RankLoss 在这里定义或从 utils 导入
class RankLoss(nn.Module):
    def forward(self, preds_high, preds_low):
        return torch.mean(torch.relu(preds_low - preds_high + 0.1))

class Solver:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(f"cuda:{config.GPU_ID}" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        
        # 定义损失函数
        self.mse_crit = nn.MSELoss()
        self.rank_crit = RankLoss()
        self.ssl_rank_crit = RankLoss() # 用于 SSL 的 Rank Loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss_avg = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}/{self.cfg.EPOCHS}", leave=False)
        
        for batch in pbar:
            # 数据解包 (根据你的 Dataset 返回格式)
            # 假设 loader 返回: content, distortion, score, sub_scores, key, content_aug, distortion_aug
            x_c, x_d, score, sub_gt, _, x_c_aug, x_d_aug = batch
            
            x_c, x_d = x_c.to(self.device), x_d.to(self.device)
            score, sub_gt = score.to(self.device), sub_gt.to(self.device)
            x_c_aug, x_d_aug = x_c_aug.to(self.device), x_d_aug.to(self.device)

            # --- Forward Pass ---
            pred_score, pred_subs, proj_c, proj_d, feat_c, feat_d = self.model(x_c, x_d)

            # --- Calculate Losses ---
            # 1. Main MSE
            loss_mse = self.mse_crit(pred_score.view(-1), score)
            
            # 2. Rank Loss
            loss_rank = self.rank_crit(pred_score.view(-1), score)
            
            # 3. MI Loss (Decoupling)
            loss_mi = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_MI > 0:
                loss_mi = self.model.mi_estimator(feat_c, feat_d)
                
            # 4. Sub-score Loss (Multi-task)
            loss_sub = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SUB > 0:
                loss_sub = self.mse_crit(pred_subs, sub_gt)

            # 5. SSL Loss (Data Augmentation Consistency)
            loss_ssl = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SSL > 0:
                # 对增强后的数据做一次前向
                pred_score_aug, _, _, _, _, _ = self.model(x_c_aug, x_d_aug)
                loss_ssl = self.ssl_rank_crit(pred_score.view(-1), pred_score_aug.view(-1))

            # --- Total Loss ---
            total_loss = (self.cfg.LAMBDA_MSE * loss_mse +
                          self.cfg.LAMBDA_RANK * loss_rank +
                          self.cfg.LAMBDA_MI * loss_mi +
                          self.cfg.LAMBDA_SUB * loss_sub +
                          self.cfg.LAMBDA_SSL * loss_ssl)

            # --- Backward ---
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_loss_avg += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        return total_loss_avg / len(self.train_loader)

    def evaluate(self):
        """验证模型并返回所有指标"""
        self.model.eval()
        preds, targets, keys = [], [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Val loader 不需要 aug 数据
                x_c, x_d, score, _, key, _, _ = batch
                x_c, x_d = x_c.to(self.device), x_d.to(self.device)
                
                pred_score, _, _, _, _, _ = self.model(x_c, x_d)
                
                preds.extend(pred_score.cpu().numpy().flatten())
                targets.extend(score.numpy().flatten())
                keys.extend(key)

        # 计算指标
        preds = np.array(preds)
        targets = np.array(targets)
        
        metrics = {
            "srcc": calculate_srcc(preds, targets),
            "plcc": calculate_plcc(preds, targets),
            "krcc": calculate_krcc(preds, targets),
            "rmse": np.sqrt(np.mean((preds*100 - targets*100)**2))
        }
        return metrics, preds, targets, keys

    def save_checkpoint(self, epoch, metrics, is_best=False):
        save_path = self.cfg.get_output_path()
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.cfg.__dict__
        }
        # 只保存最佳模型以节省空间
        if is_best:
            torch.save(state, os.path.join(save_path, "best_model.pth"))
            print(f" -> Saved Best Model at Epoch {epoch} (SRCC: {metrics['srcc']:.4f})")
