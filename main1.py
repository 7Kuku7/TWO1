#消融实验 接受结构化参数
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import argparse
import torchvision.transforms as T
from config import Config
from core.solver import Solver
from datasets.nerf_loader import NerfDataset
from datasets.ssl_transforms import SelfSupervisedAugmentor 
from models.dis_nerf_advanced import DisNeRFQA_Advanced

class MultiScaleCrop:
    def __init__(self, size=224): self.size = size
    def __call__(self, img):
        scale = int(np.random.choice([224, 256, 288]))
        img = T.Resize(scale)(img)
        img = T.RandomCrop(self.size)(img)
        return img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="Default")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=50)
    
    # Loss 权重 (Parameter Ablation)
    parser.add_argument("--lambda_ssl", type=float, default=-1)
    parser.add_argument("--lambda_mi", type=float, default=-1)
    parser.add_argument("--lambda_sub", type=float, default=-1)
    parser.add_argument("--lambda_rank", type=float, default=-1)
    
    # 结构开关 (Structural Ablation)
    # 加上这些 flag 代表 DISABLE 该模块
    parser.add_argument("--no_fusion", action='store_true', help="Disable Adaptive Fusion (Use Concat)")
    parser.add_argument("--no_multitask", action='store_true', help="Remove Sub-score Head")
    parser.add_argument("--no_decoupling", action='store_true', help="Remove MI Estimator")
    
    return parser.parse_args()

def main():
    cfg = Config()
    args = parse_args()
    
    # 应用参数
    cfg.EXP_NAME = args.exp_name
    cfg.GPU_ID = args.gpu
    cfg.EPOCHS = args.epochs
    
    # Loss 覆盖
    if args.lambda_ssl >= 0: cfg.LAMBDA_SSL = args.lambda_ssl
    if args.lambda_mi >= 0: cfg.LAMBDA_MI = args.lambda_mi
    if args.lambda_sub >= 0: cfg.LAMBDA_SUB = args.lambda_sub
    if args.lambda_rank >= 0: cfg.LAMBDA_RANK = args.lambda_rank
    
    # 结构覆盖
    use_fusion = not args.no_fusion
    use_multitask = not args.no_multitask
    use_decoupling = not args.no_decoupling

    # 如果结构被移除了，对应的 Loss 权重强制归零（防止 Solver 打印困惑的信息）
    if not use_multitask: cfg.LAMBDA_SUB = 0.0
    if not use_decoupling: cfg.LAMBDA_MI = 0.0
    
    # 路径处理
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    # Transforms
    basic_transform = T.Compose([
        MultiScaleCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # SSL 模块 (仅当 LAMBDA_SSL > 0 时才加载)
    ssl_augmentor = SelfSupervisedAugmentor() if cfg.LAMBDA_SSL > 0 else None

    # Dataset
    train_set = NerfDataset(cfg.ROOT_DIR, cfg.MOS_FILE, 'train', basic_transform, ssl_augmentor, True, cfg.USE_SUBSCORES)
    val_set = NerfDataset(cfg.ROOT_DIR, cfg.MOS_FILE, 'val', basic_transform, None, False, cfg.USE_SUBSCORES)
    
    train_loader = DataLoader(train_set, cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # Model (传入结构开关)
    print(f"Build Model | Fusion: {use_fusion} | MultiTask: {use_multitask} | Decouple: {use_decoupling}")
    model = DisNeRFQA_Advanced(
        num_subscores=4, 
        use_fusion=use_fusion, 
        use_multitask=use_multitask, 
        use_decoupling=use_decoupling
    )

    solver = Solver(model, cfg, train_loader, val_loader)
    
    best_srcc = -1
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = solver.train_epoch(epoch)
        metrics, _, _, _ = solver.evaluate()
        print(f"Ep {epoch} | Loss: {loss:.4f} | Val SRCC: {metrics['srcc']:.4f}")
        
        if metrics['srcc'] > best_srcc:
            best_srcc = metrics['srcc']
            solver.save_model(output_dir, epoch, metrics)
            with open(os.path.join(output_dir, "best_results.json"), "w") as f:
                json.dump(metrics, f)

if __name__ == "__main__":
    main()
