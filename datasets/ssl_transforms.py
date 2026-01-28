# datasets/ssl_transforms.py
import random
from PIL import Image, ImageFilter
import torchvision.transforms as T

class SelfSupervisedAugmentor:
    """
    专门用于 SSL 分支的数据增强模块。
    包含几何变换（模糊、像素化）和光度变换（颜色抖动）。
    """
    def __init__(self):
        self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        
    def add_geometric_noise(self, img):
        choice = random.choice(['blur', 'pixelate', 'none'])
        if choice == 'blur':
            radius = random.uniform(1, 3)
            return img.filter(ImageFilter.GaussianBlur(radius))
        elif choice == 'pixelate':
            w, h = img.size
            ratio = random.uniform(0.2, 0.5)
            img_small = img.resize((int(w*ratio), int(h*ratio)), resample=Image.NEAREST)
            return img_small.resize((w, h), resample=Image.NEAREST)
        return img

    def __call__(self, frames):
        # frames: List of PIL Images
        augmented_frames = []
        # 随机决定是否应用某类增强
        apply_photo = random.random() > 0.3
        apply_geo = random.random() > 0.3
        
        # 保证至少有一种增强，否则 SSL 没有意义
        if not apply_photo and not apply_geo: 
            apply_photo = True 
            
        for img in frames:
            res = img
            if apply_geo: res = self.add_geometric_noise(res)
            if apply_photo: res = self.photo_jitter(res)
            augmented_frames.append(res)
        return augmented_frames
