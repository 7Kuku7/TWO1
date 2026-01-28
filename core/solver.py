# core/solver.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import os
from utils import calculate_srcc, calculate_plcc, calculate_krcc

# 真正的 Pairwise Rank Loss (用于有监督)
class RankLoss(nn.Module):
    def forward(self, preds, targets):
        preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        targets_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        S = torch.sign(targets_diff)
        mask = (S != 0) & (S.abs() > 0)
        if mask.sum() == 0: return torch.tensor(0.0).to(preds.device)
        loss = torch.relu(-S * preds_diff + 0.1)
        return (loss * mask).sum() / (mask.sum() + 1e-6)

class Solver:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(f"cuda:{config.GPU_ID}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        
        self.mse_crit = nn.MSELoss()
        self.rank_crit = RankLoss()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss_avg = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}/{self.cfg.EPOCHS}", leave=False)
        
        for batch in pbar:
            x_c, x_d, score, sub_gt, _, x_c_aug, x_d_aug = batch
            x_c, x_d = x_c.to(self.device), x_d.to(self.device)
            score, sub_gt = score.to(self.device), sub_gt.to(self.device)
            x_c_aug, x_d_aug = x_c_aug.to(self.device), x_d_aug.to(self.device)

            # --- Forward Pass ---
            pred_score, pred_subs, proj_c, proj_d, feat_c, feat_d = self.model(x_c, x_d)
            pred_score = pred_score.view(-1)
            
            # --- Calculate Losses ---
            
            # 1. Main MSE (MOS Regression) - 如果是无监督实验，这个权重是 0
            loss_mse = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_MSE > 0:
                loss_mse = self.mse_crit(pred_score, score)
            
            # 2. Rank Loss (MOS Ranking) - 如果是无监督实验，这个权重是 0
            loss_rank = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_RANK > 0:
                loss_rank = self.rank_crit(pred_score, score)
            
            # 3. MI Loss (Feature Decoupling)
            loss_mi = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_MI > 0:
                loss_mi = self.model.mi_estimator(feat_c, feat_d)
                
            # 4. Sub-score Loss
            loss_sub = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SUB > 0:
                loss_sub = self.mse_crit(pred_subs, sub_gt)

            # 5. SSL / Proxy Loss (重点修改！)
            loss_ssl = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SSL > 0:
                # 增强图的前向
                pred_score_aug, _, _, _, _, _ = self.model(x_c_aug, x_d_aug)
                pred_score_aug = pred_score_aug.view(-1)
                
                # [核心逻辑]
                # 如果我们有 MOS (LAMBDA_MSE > 0)，我们可能只需要一致性。
                # 但如果我们没有 MOS (LAMBDA_MSE == 0)，我们必须强制 Rank：
                #    "原图分数 > 增强图分数"
                
                if self.cfg.LAMBDA_MSE == 0:
                    # == 无监督模式 (Unsupervised Mode) ==
                    # Loss = ReLU( 增强图分 - 原图分 + margin )
                    # 这迫使 原图分 > 增强图分 + 0.1
                    loss_ssl = torch.mean(torch.relu(pred_score_aug - pred_score + 0.1))
                else:
                    # == 有监督辅助模式 (Auxiliary Mode) ==
                    # 也可以用同样的逻辑，或者只用 MSE 一致性
                    loss_ssl = torch.mean(torch.relu(pred_score_aug - pred_score + 0.05))

            # --- Total Loss ---
            total_loss = (self.cfg.LAMBDA_MSE * loss_mse +
                          self.cfg.LAMBDA_RANK * loss_rank +
                          self.cfg.LAMBDA_MI * loss_mi +
                          self.cfg.LAMBDA_SUB * loss_sub +
                          self.cfg.LAMBDA_SSL * loss_ssl)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss_avg += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'mse': f"{loss_mse.item():.4f}"})
            
        return total_loss_avg / len(self.train_loader)
    
    # 记得加上 save_model 方法，main.py 用到了
    def save_model(self, path, epoch, metrics):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.cfg.__dict__
        }
        torch.save(state, os.path.join(path, "best_model.pth"))

    def evaluate(self):
        self.model.eval()
        preds, targets, keys = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                x_c, x_d, score, _, key, _, _ = batch
                x_c, x_d = x_c.to(self.device), x_d.to(self.device)
                pred_score, _, _, _, _, _ = self.model(x_c, x_d)
                preds.extend(pred_score.cpu().numpy().flatten())
                targets.extend(score.numpy().flatten())
                keys.extend(key)
        preds = np.array(preds)
        targets = np.array(targets)
        metrics = {
            "srcc": calculate_srcc(preds, targets),
            "plcc": calculate_plcc(preds, targets),
            "krcc": calculate_krcc(preds, targets),
            "rmse": np.sqrt(np.mean((preds*100 - targets*100)**2))
        }
        return metrics, preds, targets, keys
