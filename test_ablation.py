# test_ablation.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import random
import os
import json
import argparse

# 引入项目模块
from config import Config
from datasets.nerf_loader import NerfDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc

def parse_args():
    parser = argparse.ArgumentParser()
    # 必填：运行目录
    parser.add_argument("--run_dir", type=str, required=True, help="Path to result folder")
    
    # 结构开关：必须与训练时保持一致！
    # 如果训练时用了 --no_fusion，测试时也要加 --no_fusion
    parser.add_argument("--no_fusion", action='store_true', help="Disable Fusion")
    parser.add_argument("--no_multitask", action='store_true', help="Disable Multi-task Head")
    parser.add_argument("--no_decoupling", action='store_true', help="Disable Decoupling")
    
    return parser.parse_args()

def test():
    args = parse_args()
    cfg = Config()
    
    # 1. 结构控制 (与 main.py 逻辑一致)
    use_fusion = not args.no_fusion
    use_multitask = not args.no_multitask
    use_decoupling = not args.no_decoupling
    
    print("="*40)
    print(f"[Test] Run Dir: {args.run_dir}")
    print(f"[Test] Structure: Fusion={use_fusion}, MultiTask={use_multitask}, Decouple={use_decoupling}")
    print("="*40)

    # 2. 数据准备
    # 保持分布一致的 Transform
    test_transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='test', 
        basic_transform=test_transform,
        ssl_transform=None,
        distortion_sampling=False,
        use_subscores=cfg.USE_SUBSCORES 
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    # 3. 初始化模型 (传入结构参数)
    device = torch.device(f"cuda:{cfg.GPU_ID}")
    model = DisNeRFQA_Advanced(
        num_subscores=4, 
        use_fusion=use_fusion,          # <--- 关键
        use_multitask=use_multitask,    # <--- 关键
        use_decoupling=use_decoupling   # <--- 关键
    ).to(device)
    
    # 4. 加载权重
    ckpt_path = os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("[Success] Weights loaded successfully.")
    except RuntimeError as e:
        print("\n[Error] Weight mismatch! Did you forget a flag (e.g., --no_fusion)?")
        print(e)
        return

    model.eval()

    # 5. 推理
    preds, targets, keys = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_c, x_d, score, _, key, _, _ = batch
            x_c, x_d = x_c.to(device), x_d.to(device)
            pred_score, _, _, _, _, _ = model(x_c, x_d)
            preds.extend(pred_score.cpu().numpy().flatten())
            targets.extend(score.numpy().flatten())
            keys.extend(key)

    # 6. 计算指标
    preds, targets = np.array(preds), np.array(targets)
    srcc = calculate_srcc(preds, targets)
    plcc = calculate_plcc(preds, targets)
    rmse = np.sqrt(np.mean((preds*100 - targets*100)**2))

    print(f"\n>>> Test Results: SRCC={srcc:.4f}, PLCC={plcc:.4f}, RMSE={rmse:.4f}")

    # 保存结果
    save_path = os.path.join(args.run_dir, "test_results_manual.json")
    with open(save_path, "w") as f:
        json.dump({"srcc": srcc, "plcc": plcc, "preds": preds.tolist()}, f, indent=4)

if __name__ == "__main__":
    test()
