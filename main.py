# main.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import datetime  # [新增]
import torchvision.transforms as T

# ================= 导入模块 =================
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set Global Seed: {seed}")

def main():
    # 1. 初始化
    cfg = Config()
    set_seed(cfg.SEED)
    
    # [新增] 生成带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.get_output_path(), f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("="*50)
    print(f"Start Experiment: {cfg.EXP_NAME}")
    print(f"Time: {timestamp}")
    print(f"Output Dir: {output_dir}")
    print("="*50)

    # 2. Transforms
    basic_transform = T.Compose([
        MultiScaleCrop(224), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ssl_augmentor = None
    if cfg.LAMBDA_SSL > 0:
        print(" -> SSL Augmentation Module: ENABLED")
        ssl_augmentor = SelfSupervisedAugmentor()
    else:
        print(" -> SSL Augmentation Module: DISABLED (Ablation)")

    # 3. Dataset
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

    # 4. Model
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=cfg.USE_FUSION)

    # 5. Solver
    solver = Solver(model, cfg, train_loader, val_loader)

    # 6. Train Loop
    # best_srcc = -1.0
    
    # for epoch in range(1, cfg.EPOCHS + 1):
    #     loss = solver.train_epoch(epoch)
    #     metrics, preds, targets, keys = solver.evaluate()
        
    #     print(f"Epoch {epoch} | Loss: {loss:.4f} | Val SRCC: {metrics['srcc']:.4f} | PLCC: {metrics['plcc']:.4f}")
        
    #     if metrics['srcc'] > best_srcc:
    #         best_srcc = metrics['srcc']
    #         print(f"  >>> New Best SRCC: {best_srcc:.4f} (Saving...)")
            
    #         # 保存最佳模型到带时间戳的目录
    #         solver.save_model(output_dir, epoch, metrics)
    # 6. Train Loop
    best_score = -1.0  # [修改] 改为记录综合分数
    
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = solver.train_epoch(epoch)
        metrics, preds, targets, keys = solver.evaluate()
        
        # [修改] 计算综合得分：SRCC + PLCC
        current_score = metrics['srcc'] + metrics['plcc']
        
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Val SRCC: {metrics['srcc']:.4f} | PLCC: {metrics['plcc']:.4f} | Sum: {current_score:.4f}")
        
        # [修改] 判断综合分数是否提升
        if current_score > best_score:
            best_score = current_score
            print(f"  >>> New Best Score (SRCC+PLCC): {best_score:.4f} (Saving...)")
            
            # 保存模型
            solver.save_model(output_dir, epoch, metrics)
            
            # 保存详细 JSON
            if cfg.SAVE_PER_VIDEO_RESULT:
                res_path = os.path.join(output_dir, "best_results.json")
                safe_metrics = {k: float(v) for k, v in metrics.items()}
                with open(res_path, 'w') as f:
                    json.dump({
                        "run_info": {"epoch": epoch, "seed": cfg.SEED, "time": timestamp},
                        "metrics": safe_metrics,
                        "preds": preds.tolist(),
                        "targets": targets.tolist(),
                        "keys": keys
                    }, f, indent=4)

    print("="*50)
    print(f"Finished. Best SRCC: {best_srcc:.4f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
