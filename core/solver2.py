# 消融实验 布尔开关

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import json
from utils import calculate_srcc, calculate_plcc, calculate_krcc

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
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}", leave=False)
        
        for batch in pbar:
            x_c, x_d, score, sub_gt, _, x_c_aug, x_d_aug = batch
            x_c, x_d = x_c.to(self.device), x_d.to(self.device)
            score, sub_gt = score.to(self.device), sub_gt.to(self.device)
            if x_c_aug is not None:
                x_c_aug, x_d_aug = x_c_aug.to(self.device), x_d_aug.to(self.device)

            # --- Forward Pass ---
            pred_score, pred_subs, proj_c, proj_d, feat_c, feat_d = self.model(x_c, x_d)
            pred_score = pred_score.view(-1)
            
            # --- Calculate Losses ---
            loss_mse = self.mse_crit(pred_score, score) if self.cfg.LAMBDA_MSE > 0 else torch.tensor(0.0).to(self.device)
            loss_rank = self.rank_crit(pred_score, score) if self.cfg.LAMBDA_RANK > 0 else torch.tensor(0.0).to(self.device)
            
            # [Smart Loss] 只有当模型真正输出了 sub_scores (即模块存在) 时才计算 Loss
            loss_sub = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SUB > 0 and pred_subs is not None:
                loss_sub = self.mse_crit(pred_subs, sub_gt)
            
            # [Smart Loss] 只有当模型有 mi_estimator 时才计算 MI Loss
            loss_mi = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_MI > 0 and self.model.mi_estimator is not None:
                loss_mi = self.model.mi_estimator(feat_c, feat_d)
                
            loss_ssl = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SSL > 0 and x_c_aug is not None:
                pred_score_aug, _, _, _, _, _ = self.model(x_c_aug, x_d_aug)
                pred_score_aug = pred_score_aug.view(-1)
                loss_ssl = torch.mean(torch.relu(pred_score_aug - pred_score + 0.05))

            total_loss = (self.cfg.LAMBDA_MSE * loss_mse +
                          self.cfg.LAMBDA_RANK * loss_rank +
                          self.cfg.LAMBDA_MI * loss_mi +
                          self.cfg.LAMBDA_SUB * loss_sub +
                          self.cfg.LAMBDA_SSL * loss_ssl)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss_avg += total_loss.item()
            
        return total_loss_avg / len(self.train_loader)

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
        
        preds, targets = np.array(preds), np.array(targets)
        metrics = {"srcc": calculate_srcc(preds, targets), "plcc": calculate_plcc(preds, targets)}
        return metrics, preds, targets, keys

    def save_model(self, path, epoch, metrics):
        torch.save({'state_dict': self.model.state_dict(), 'metrics': metrics}, os.path.join(path, "best_model.pth"))
