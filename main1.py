# run_ablation.py
# python run_ablation.py --exp_name Proposed_Full --lambda_mi 0.0
# python run_ablation.py --exp_name Ablation_wo_MultiTask --no_multitask --lambda_mi 0.0
# python run_ablation.py --exp_name Ablation_wo_SSL --lambda_ssl 0.0 --lambda_mi 0.0


import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import argparse
import torchvision.transforms as T

# ================= 导入模块 =================
from config import Config
from core.solver import Solver
from datasets.nerf_loader import NerfDataset
from datasets.ssl_transforms import SelfSupervisedAugmentor 
from models.dis_nerf_advanced import DisNeRFQA_Advanced

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    
    # 实验名称
    parser.add_argument("--exp_name", type=str, default="Default", help="Experiment name override")
    
    # 损失权重覆盖 (传入 -1 表示使用 config.py 默认值)
    parser.add_argument("--lambda_ssl", type=float, default=-1.0)
    parser.add_argument("--lambda_mi", type=float, default=-1.0)
    parser.add_argument("--lambda_sub", type=float, default=-1.0)
    parser.add_argument("--lambda_rank", type=float, default=-1.0)
    
    # 结构开关 (加上这些 flag 代表 DISABLE 该模块)
    parser.add_argument("--no_fusion", action='store_true', help="Disable Adaptive Fusion")
    parser.add_argument("--no_multitask", action='store_true', help="Remove Sub-score Head")
    parser.add_argument("--no_decoupling", action='store_true', help="Remove MI Estimator")
    
    # 训练设置
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpu", type=str, default="0")
    
    return parser.parse_args()

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
    # 1. 初始化 Config
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
    
    # 结构开关 (args.no_XXX 为 True 时表示禁用)
    use_fusion = not args.no_fusion
    use_multitask = not args.no_multitask
    use_decoupling = not args.no_decoupling

    # 联动逻辑：如果结构被移除，Loss 必须归零
    if not use_multitask: cfg.LAMBDA_SUB = 0.0
    if not use_decoupling: cfg.LAMBDA_MI = 0.0
    
    set_seed(cfg.SEED)
    
    # 路径处理
    output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print(f"Start Experiment: {cfg.EXP_NAME}")
    print(f"  > Structure: Fusion={use_fusion}, MultiTask={use_multitask}, Decouple={use_decoupling}")
    print(f"  > Weights: MSE={cfg.LAMBDA_MSE}, Rank={cfg.LAMBDA_RANK}, SSL={cfg.LAMBDA_SSL}, MI={cfg.LAMBDA_MI}, Sub={cfg.LAMBDA_SUB}")
    print(f"  > Output: {output_dir}")
    print("="*60)
    
    # 保存配置
    with open(os.path.join(output_dir, "config_snapshot.json"), "w") as f:
        cfg_dict = {k: v for k, v in Config.__dict__.items() if k.isupper()}
        json.dump(cfg_dict, f, indent=4)

    # 2. Transforms
    basic_transform = T.Compose([
        MultiScaleCrop(224), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ssl_augmentor = SelfSupervisedAugmentor() if cfg.LAMBDA_SSL > 0 else None

    # 3. Dataset
    train_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='train',
        basic_transform=basic_transform,
        ssl_transform=ssl_augmentor,
        distortion_sampling=True, # 这里实际上会调用 _get_distortion_input
        num_frames=8,
        use_subscores=cfg.USE_SUBSCORES
    )
    
    val_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='val',
        basic_transform=basic_transform,
        ssl_transform=None,
        distortion_sampling=False, # 验证集通常不做 distortion 分支的特殊增强，或者也可以保持一致
        num_frames=8,
        use_subscores=cfg.USE_SUBSCORES
    )

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Model (传入结构开关)
    model = DisNeRFQA_Advanced(
        num_subscores=4, 
        use_fusion=use_fusion, 
        use_multitask=use_multitask, 
        use_decoupling=use_decoupling
    )

    # 5. Solver
    solver = Solver(model, cfg, train_loader, val_loader)

    # 6. Train Loop
    best_score = -1.0 # 综合分数 (SRCC + PLCC)
    
    for epoch in range(1, cfg.EPOCHS + 1):
        loss = solver.train_epoch(epoch)
        metrics, preds, targets, keys = solver.evaluate()
        
        # [修正] 使用 SRCC + PLCC 作为保存标准
        current_score = metrics['srcc'] + metrics['plcc']
        
        print(f"Ep {epoch} | Loss: {loss:.4f} | Val SRCC: {metrics['srcc']:.4f} | PLCC: {metrics['plcc']:.4f} | Sum: {current_score:.4f}")
        
        if current_score > best_score:
            best_score = current_score
            print(f"  >>> New Best Score: {best_score:.4f} (Saving...)")
            
            solver.save_model(output_dir, epoch, metrics)
            
            # 保存详细结果
            with open(os.path.join(output_dir, "best_results.json"), "w") as f:
                json.dump({
                    "run_info": {"epoch": epoch, "best_score": best_score},
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "preds": preds.tolist(),
                    "targets": targets.tolist(),
                    "keys": keys
                }, f)

if __name__ == "__main__":
    main()
