# test.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import random
import os
import json
import argparse

from config import Config
from datasets.nerf_loader import NerfDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc

# 这里的 Transform 和 main.py 保持一致
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[Test] Set Global Seed: {seed}")

def test():
    parser = argparse.ArgumentParser()
    # 允许你指定某个特定的运行目录，例如 "experiments_results/Exp_v2/seed_42/run_20260128_..."
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory containing best_model.pth")
    args = parser.parse_args()

    cfg = Config()
    
    # 1. 关键：设置种子，保证数据集划分一致！
    set_seed(cfg.SEED)

    # 2. 准备测试数据 (mode='test')
    # 测试时一般不做 RandomCrop，而是做 CenterCrop 或直接 Resize，保持确定性
    # 但为了和训练保持分布一致，也可以用同样的 transform
    test_transform = T.Compose([
        T.Resize((224, 224)), # 测试时简单一点，直接缩放
        T.ToTensor(), 
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_set = NerfDataset(
        root_dir=cfg.ROOT_DIR,
        mos_file=cfg.MOS_FILE,
        mode='test', # 这里的 mode='test' 会去 consts_simple_split 里拿测试集列表
        basic_transform=test_transform,
        ssl_transform=None,
        distortion_sampling=False, # 测试一般不用 Grid Sampling，只看全图
        use_subscores=cfg.USE_SUBSCORES
    )
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    print(f"[Test] Test Set Size: {len(test_set)}")

    # 3. 加载模型
    device = torch.device(f"cuda:{cfg.GPU_ID}")
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=cfg.USE_FUSION).to(device)
    
    # 加载权重
    ckpt_path = os.path.join(args.run_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"[Test] Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 4. 推理
    preds, targets, keys = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_c, x_d, score, _, key, _, _ = batch
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # 前向传播
            pred_score, _, _, _, _, _ = model(x_c, x_d)
            
            preds.extend(pred_score.cpu().numpy().flatten())
            targets.extend(score.numpy().flatten())
            keys.extend(key)

    # 5. 计算指标
    preds = np.array(preds)
    targets = np.array(targets)
    
    srcc = calculate_srcc(preds, targets)
    plcc = calculate_plcc(preds, targets)
    krcc = calculate_krcc(preds, targets)
    rmse = np.sqrt(np.mean((preds*100 - targets*100)**2))

    print("\n" + "="*30)
    print("       TEST RESULTS       ")
    print("="*30)
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"KRCC: {krcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*30)

    # 保存测试结果
    save_path = os.path.join(args.run_dir, "test_results.json")
    with open(save_path, "w") as f:
        json.dump({
            "metrics": {"srcc": float(srcc), "plcc": float(plcc), "krcc": float(krcc), "rmse": float(rmse)},
            "preds": preds.tolist(),
            "targets": targets.tolist(),
            "keys": keys
        }, f, indent=4)
    print(f"[Test] Results saved to {save_path}")

if __name__ == "__main__":
    test()
