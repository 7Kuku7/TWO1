# scripts/extract_features.py
import torch
import numpy as np
import os
import sys
# 把根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from datasets.nerf_loader import NerfDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

def extract():
    cfg = Config()
    device = torch.device(f"cuda:{cfg.GPU_ID}")
    
    # 1. 准备模型
    # 注意：这里应该加载你训练好的 checkpoint，而不是随机初始化的模型
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=True)
    
    ckpt_path = "experiments_results/Exp_v2_Standard_Split/seed_42/best_model.pth" # 改成你的路径
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Warning: No checkpoint found! Extracting features with random weights.")
    
    model.to(device)
    model.eval()
    
    # 2. 准备数据
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 我们只提取测试集的特征
    dataset = NerfDataset(cfg.ROOT_DIR, cfg.MOS_FILE, mode='test', basic_transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_content_feats = []
    all_dist_feats = []
    all_scores = []
    all_keys = []

    # 3. 提取循环
    with torch.no_grad():
        for batch in tqdm(loader):
            x_c, x_d, score, _, key, _, _ = batch
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # 这里的 forward 需要 model 返回特征，确保 model 代码里返回了 feat_c, feat_d
            _, _, _, _, feat_c, feat_d = model(x_c, x_d)
            
            all_content_feats.append(feat_c.cpu().numpy())
            all_dist_feats.append(feat_d.cpu().numpy())
            all_scores.append(score.numpy())
            all_keys.extend(key)

    # 4. 保存
    save_dir = "features_output"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "feats_content.npy"), np.vstack(all_content_feats))
    np.save(os.path.join(save_dir, "feats_distortion.npy"), np.vstack(all_dist_feats))
    np.save(os.path.join(save_dir, "scores.npy"), np.vstack(all_scores))
    print(f"Features saved to {save_dir}")

if __name__ == "__main__":
    extract()
