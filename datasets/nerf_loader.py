# main.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os

# 导入我们拆分的模块
from config import Config
from core.solver import Solver
from datasets.of_nerf import AdvancedOFNeRFDataset # 假设你保留了这个类名
from models.dis_nerf_advanced import DisNeRFQA_Advanced
import torchvision.transforms as T
from datasets.of_nerf import MultiScaleCrop # 确保这个 transform 被正确导入

def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set Global Seed: {seed}")

def main():
    # 1. 初始化配置与环境
    cfg = Config()
    set_seed(cfg.SEED)
    print(f"Start Experiment: {cfg.EXP_NAME}")
    print(f"Description: {cfg.DESCRIPTION}")

    # 2. 准备数据 Transforms
    # 可以根据 Config 里的开关决定是否用 Multiscale
    transform = T.Compose([
        MultiScaleCrop(224), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 实例化数据集
    # 注意：这里直接用 Config 里的参数，不再需要 argparse 传参
    train_set = AdvancedOFNeRFDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='train',
        transform=transform,
        distortion_sampling=True,
        use_subscores=cfg.USE_SUBSCORES,
        enable_ssl=(cfg.LAMBDA_SSL > 0) # 如果权重>0，则开启SSL数据增强
    )
    
    val_set = AdvancedOFNeRFDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='val',
        transform=transform,
        distortion_sampling=False, # 验证集不做 grid sampling 增强
        use_subscores=cfg.USE_SUBSCORES,
        enable_ssl=False
    )

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
        metrics, preds, targets, keys = solver.evaluate()
        
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Val SRCC: {metrics['srcc']:.4f} | PLCC: {metrics['plcc']:.4f}")
        
        # 保存最佳结果
        if metrics['srcc'] > best_srcc:
            best_srcc = metrics['srcc']
            solver.save_checkpoint(epoch, metrics, is_best=True)
            
            # 保存详细 JSON
            if cfg.SAVE_PER_VIDEO_RESULT:
                res_path = os.path.join(cfg.get_output_path(), "best_results.json")
                with open(res_path, 'w') as f:
                    json.dump({
                        "epoch": epoch,
                        "metrics": metrics,
                        "preds": preds.tolist(),
                        "targets": targets.tolist(),
                        "keys": keys
                    }, f, indent=4)

    print(f"Done. Best SRCC: {best_srcc:.4f}. Results saved to {cfg.get_output_path()}")

if __name__ == "__main__":
    main()
