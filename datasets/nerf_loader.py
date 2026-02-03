# datasets/nerf_loader.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import torchvision.transforms as T
import sys
import os

# 引用你原来的 split 工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from consts_simple_split import build_simple_split
except ImportError:
    print("Warning: consts_simple_split not found.")

class NerfDataset(Dataset):
    def __init__(self, root_dir, mos_file, mode='train', 
                 basic_transform=None, 
                 ssl_transform=None,  
                 distortion_sampling=False, 
                 num_frames=8,
                 use_subscores=True):
        
        self.root_dir = Path(root_dir)
        self.basic_transform = basic_transform
        self.ssl_transform = ssl_transform 
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
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: all_frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        if not all_frames: 
            # 如果真的没找到帧，为了防止崩溃，可以打印警告并返回空，或者直接报错
            # 这里为了稳健性，报错
            raise ValueError(f"No frames found in {folder_path}")
            
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        return [Image.open(all_frames[i]).convert('RGB') for i in indices]

    def _grid_mini_patch_sampling(self, tensor_img):
        # 将之前的 grid sampling 逻辑复制过来
        # 假设输入是 [T, C, H, W]
        T, C, H, W = tensor_img.shape
        grid_h, grid_w = 4, 4
        patch_h, patch_w = H // grid_h, W // grid_w
        
        patches = tensor_img.view(T, C, grid_h, patch_h, grid_w, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(T, grid_h * grid_w, C, patch_h, patch_w)
        
        shuffled_patches = []
        for t in range(T):
            indices = torch.randperm(grid_h * grid_w)
            p = patches[t][indices]
            shuffled_patches.append(p)
            
        shuffled_patches = torch.stack(shuffled_patches)
        shuffled_patches = shuffled_patches.view(T, grid_h, grid_w, C, patch_h, patch_w)
        shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return shuffled_patches.view(T, C, H, W)

    # [修改 1] 将原来的 _grid_mini_patch_sampling 替换为基于滤波的特征提取
    def _get_distortion_input(self, tensor_img):
        """
        不再打乱 Patch，而是使用拉普拉斯算子提取高频残差图。
        这能保留几何结构，同时突出 NeRF 的边缘伪影和噪声。
        Input: [T, C, H, W]
        """
        # 定义拉普拉斯卷积核 (提取边缘和高频噪声)
        # 形状: [C, 1, 3, 3] -> 针对每个通道独立卷积
        kernel = torch.tensor([[-1, -1, -1], 
                               [-1,  8, -1], 
                               [-1, -1, -1]], dtype=torch.float32)
        
        # 扩展到 3 个通道 (RGB)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # 放到与输入相同的设备上 (CPU)
        weights = kernel
        
        T, C, H, W = tensor_img.shape
        
        # 为了进行卷积，先 reshape 成 [T, C, H, W] -> 也就是 batch size = T
        # 注意：这里我们在 CPU 上做，数据量不大，速度通常可以接受
        with torch.no_grad():
            # 使用 groups=3 保证 RGB 通道独立处理，padding=1 保持尺寸不变
            out = torch.nn.functional.conv2d(tensor_img, weights, groups=3, padding=1)
        
        return out

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
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
            
        frames_pil = self._load_frames_pil(folder_path)
        
        # Main Branch
        t_imgs = [self.basic_transform(img) for img in frames_pil]
        content_input = torch.stack(t_imgs)
        
        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(content_input)
        else:
            distortion_input = content_input.clone()
            
        # SSL Branch
        content_input_aug = torch.tensor(0.0) 
        distortion_input_aug = torch.tensor(0.0)
        
        if self.ssl_transform is not None:
            frames_aug_pil = self.ssl_transform(frames_pil)
            t_imgs_aug = [self.basic_transform(img) for img in frames_aug_pil]
            content_input_aug = torch.stack(t_imgs_aug)
            
            if self.distortion_sampling:
                distortion_input_aug = self._grid_mini_patch_sampling(content_input_aug)
            else:
                distortion_input_aug = content_input_aug.clone()
                
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key, content_input_aug, distortion_input_aug

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # [修复点] 必须加上这个方法！
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    def __len__(self):
        return len(self.valid_samples)
