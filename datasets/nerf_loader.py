# datasets/nerf_loader.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import torchvision.transforms as T
import sys
import os

# 引用你原来的 split 工具 (如果路径报错，请根据实际情况调整 sys.path)
# 假设 consts_simple_split 在项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from consts_simple_split import build_simple_split
except ImportError:
    print("Warning: consts_simple_split not found.")

class NerfDataset(Dataset):
    def __init__(self, root_dir, mos_file, mode='train', 
                 basic_transform=None, 
                 ssl_transform=None,  # 关键：从外部传入 SSL 增强器
                 distortion_sampling=False, 
                 num_frames=8,
                 use_subscores=True):
        
        self.root_dir = Path(root_dir)
        self.basic_transform = basic_transform
        self.ssl_transform = ssl_transform # 如果为 None，就不产生 SSL 数据
        self.distortion_sampling = distortion_sampling
        self.num_frames = num_frames
        self.use_subscores = use_subscores
        
        # 加载数据集划分
        train, val, test = build_simple_split(self.root_dir)
        self.samples = {'train': train, 'val': val, 'test': test}[mode]
        
        # 加载 MOS 标签
        with open(mos_file, 'r') as f:
            self.mos_labels = json.load(f)
            
        # 过滤有效样本
        self.valid_samples = []
        for s in self.samples:
            key = self._get_key_from_path(s)
            if key in self.mos_labels:
                self.valid_samples.append(s)

    def _get_key_from_path(self, path):
        parts = path.name.split("__")
        if len(parts) == 4:
            return "+".join(parts)
        return path.name

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        # ... (保留你原来的寻找帧的逻辑) ...
        if not all_frames: 
            raise ValueError(f"No frames in {folder_path}")
            
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        return [Image.open(all_frames[i]).convert('RGB') for i in indices]

    def _grid_mini_patch_sampling(self, tensor_img):
        # ... (保留你原来的 grid sampling 逻辑) ...
        # 这里为了节省篇幅省略具体代码，把原文件里的复制过来即可
        return tensor_img 

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
        # 1. 获取标签
        entry = self.mos_labels[key]
        if isinstance(entry, dict):
            score = entry['mos'] / 100.0
            sub_data = entry.get('sub_scores', {})
        else:
            score = entry / 100.0
            sub_data = {}
            
        score_tensor = torch.tensor(score, dtype=torch.float32)
        
        sub_scores_tensor = torch.zeros(4, dtype=torch.float32)
        if self.use_subscores:
             sub_scores_tensor = torch.tensor([
                sub_data.get("discomfort", 0), sub_data.get("blur", 0),
                sub_data.get("lighting", 0), sub_data.get("artifacts", 0)
            ], dtype=torch.float32) / 5.0
            
        # 2. 加载图片
        frames_pil = self._load_frames_pil(folder_path)
        
        # 3. 基础处理 (Main Branch)
        # Apply transform to list of PIL
        t_imgs = [self.basic_transform(img) for img in frames_pil]
        content_input = torch.stack(t_imgs) # [T, C, H, W]
        
        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(content_input)
        else:
            distortion_input = content_input.clone()
            
        # 4. SSL 处理 (Auxiliary Branch) - 只有传入了 ssl_transform 才会跑
        content_input_aug = torch.tensor(0.0) # Placeholder
        distortion_input_aug = torch.tensor(0.0)
        
        if self.ssl_transform is not None:
            frames_aug_pil = self.ssl_transform(frames_pil) # 先做增强
            t_imgs_aug = [self.basic_transform(img) for img in frames_aug_pil] # 再转 Tensor
            content_input_aug = torch.stack(t_imgs_aug)
            
            if self.distortion_sampling:
                distortion_input_aug = self._grid_mini_patch_sampling(content_input_aug)
            else:
                distortion_input_aug = content_input_aug.clone()
                
        # 返回所有需要的东西
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key, content_input_aug, distortion_input_aug
