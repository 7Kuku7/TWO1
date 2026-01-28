# main.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import torchvision.transforms as T

# ================= 导入我们拆分好的新模块 =================
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

def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # 修正: torch.cuda.manual_seed 是单卡，all 是多卡
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set Global Seed: {seed}")

def main():
    # 1. 初始化配置与环境
    cfg = Config()
    set_seed(cfg.SEED)
    print("="*50)
    print(f"Start Experiment: {cfg.EXP_NAME}")
    print(f"Description: {cfg.DESCRIPTION}")
    print(f"Output Dir: {cfg.get_output_path()}")
    print("="*50)

    # 2. 准备数据 Transforms (基础变换)
    basic_transform = T.Compose([
        MultiScaleCrop(224), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 实例化 SSL 增强器 (仅当权重 > 0 时)
    ssl_augmentor = None
    if cfg.LAMBDA_SSL > 0:
        print(" -> SSL Augmentation Module: ENABLED")
        ssl_augmentor = SelfSupervisedAugmentor()
    else:
        print(" -> SSL Augmentation Module: DISABLED (Ablation)")

    # 3. 实例化数据集
    train_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='train',
        basic_transform=basic_transform,
        ssl_transform=ssl_augmentor,
        distortion_sampling=True,
        use_subscores=cfg.USE_SUBSCORES
    )
    
    val_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='val',
        basic_transform=basic_transform,
        ssl_transform=None,
        distortion_sampling=False,
        use_subscores=cfg.USE_SUBSCORES
    )

    print(f"Dataset Loaded. Train: {len(train_set)}, Val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    # 4. 实例化模型
    model = DisNeRFQA_Advanced(
        num_subscores=4, 
        use_fusion=cfg.USE_FUSION
    )

    # 5. 初始化 Solver (训练器)
    solver = Solver(model, cfg, train_loader, val_loader)

    # 6. 开始训练循环
    best_srcc = -1.0
    
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = solver.train_epoch(epoch)
        # 验证
        metrics, preds, targets, keys = solver.evaluate()
        
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Val SRCC: {metrics['srcc']:.4f} | PLCC: {metrics['plcc']:.4f} | KRCC: {metrics['krcc']:.4f}")
        
        # 保存最佳结果
        if metrics['srcc'] > best_srcc:
            best_srcc = metrics['srcc']
            print(f"  >>> New Best SRCC: {best_srcc:.4f} (Saving Checkpoint)")
            solver.save_checkpoint(epoch, metrics, is_best=True)
            
            # 保存详细 JSON
            if cfg.SAVE_PER_VIDEO_RESULT:
                res_path = os.path.join(cfg.get_output_path(), "best_results.json")
                
                # [关键修复] 将 numpy 类型转换为 python原生 float，否则 json.dump 会报错
                safe_metrics = {k: float(v) for k, v in metrics.items()}
                
                with open(res_path, 'w') as f:
                    json.dump({
                        "run_info": {"epoch": epoch, "seed": cfg.SEED},
                        "metrics": safe_metrics, # 使用处理过的 metrics
                        "preds": preds.tolist(), # numpy array 转 list
                        "targets": targets.tolist(),
                        "keys": keys
                    }, f, indent=4)

    print("="*50)
    print(f"Experiment Finished. Best SRCC: {best_srcc:.4f}")
    print(f"Results saved to: {cfg.get_output_path()}")
    print("="*50)

if __name__ == "__main__":
    main()
