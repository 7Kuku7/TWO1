import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualBlock(nn.Module):
    """
    [移植] 来自 MMIF-CDDFuse 的反向残差块
    """
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim), # 原版注释掉了BN，这里保持一致或根据需要开启
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    """
    [移植] 来自 MMIF-CDDFuse 的 DetailNode (基于 INN 耦合层)
    修改：增加了 dim 参数，使其不局限于 64 通道
    """
    def __init__(self, dim=64):
        super(DetailNode, self).__init__()
        # 输入 dim 分为两半处理
        half_dim = dim // 2
        
        # Scale is Ax + b, i.e. affine transformation
        # 这里使用了 InvertedResidualBlock 来预测仿射变换参数
        self.theta_phi = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        
        self.shffleconv = nn.Conv2d(dim, dim, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        # 先混合再拆分，促进信息交互
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
            
        # 仿射耦合变换 (Affine Coupling)
        # 1. z2 update
        z2 = z2 + self.theta_phi(z1)
        # 2. z1 update
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        
        return z1, z2

class DistortionCNN(nn.Module):
    """
    [封装] 专门用于提取失真特征的 CNN
    结构: Stem -> Stack of DetailNodes -> Projection -> GAP
    """
    def __init__(self, in_chans=3, feature_dim=768, base_dim=64, num_layers=3):
        super(DistortionCNN, self).__init__()
        
        # 1. Stem: 将图像投影到特征空间 (e.g. 3 -> 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. Detail Feature Extraction: 堆叠 DetailNode
        # 每一层都需要输入 z1, z2，所以我们在 forward 里拆分
        self.detail_layers = nn.ModuleList([DetailNode(dim=base_dim) for _ in range(num_layers)])
        
        # 3. Projection: 将特征维度映射到目标维度 (e.g. 64 -> 768)
        self.proj = nn.Conv2d(base_dim, feature_dim, kernel_size=1, bias=True)
        
    def forward(self, x):
        # x: [B, 3, H, W]
        
        # Stem
        x = self.stem(x) # [B, 64, H, W]
        
        # Detail Extraction
        # 将特征拆分为两半作为 DetailNode 的初始输入
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        
        for layer in self.detail_layers:
            z1, z2 = layer(z1, z2)
            
        # 合并回 [B, 64, H, W]
        x_detail = torch.cat((z1, z2), dim=1)
        
        # 投影到目标维度 [B, 768, H, W]
        x_out = self.proj(x_detail)
        
        # 全局平均池化 (Global Average Pooling) -> [B, 768]
        # 将空间维度 (H, W) 压缩为 1
        x_out = x_out.mean(dim=[2, 3])
        
        return x_out
