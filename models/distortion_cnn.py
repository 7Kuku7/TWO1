import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidualBlock(nn.Module):
    """
    来自 MMIF-CDDFuse 的反向残差块，用于捕获局部特征
    """
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw (Pointwise Conv)
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            # dw (Depthwise Conv)
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    """
    来自 MMIF-CDDFuse 的细节提取节点 (基于 INN 耦合层思想)
    修改：支持动态通道维度 (dim)
    """
    def __init__(self, dim=64):
        super(DetailNode, self).__init__()
        half_dim = dim // 2
        
        # Scale is Ax + b, i.e. affine transformation
        # 输入输出通道改为动态计算的 half_dim
        self.theta_phi = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=half_dim, oup=half_dim, expand_ratio=2)
        
        self.shffleconv = nn.Conv2d(dim, dim, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        # 将特征在通道维度一分为二
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        # 先拼接混合，再切分，促进通道间信息交互
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        
        # 仿射耦合变换 (Affine Coupling)
        # 利用 z1 预测参数来调整 z2
        z2 = z2 + self.theta_phi(z1)
        # 利用新的 z2 预测参数来调整 z1
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        
        return z1, z2

class DistortionCNN(nn.Module):
    """
    封装后的 Distortion Encoder
    结构：Stem -> DetailFeatureExtraction (INN Stack) -> Projection -> Global Pooling
    """
    def __init__(self, in_chans=3, feature_dim=768, base_dim=64, num_layers=3):
        super(DistortionCNN, self).__init__()
        
        # 1. Stem Layer: 将输入图像 (RGB 3通道) 映射到特征空间 (如 64通道)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. Detail Feature Extraction (堆叠 DetailNode)
        self.detail_layers = nn.ModuleList([DetailNode(dim=base_dim) for _ in range(num_layers)])
        
        # 3. Output Projection: 将特征维度映射到主干网络的维度 (如 64 -> 768)
        self.proj = nn.Conv2d(base_dim, feature_dim, kernel_size=1, bias=True)
        
    def forward(self, x):
        # x shape: [B, C, H, W]
        
        # Stem
        x = self.stem(x) # -> [B, 64, H, W]
        
        # Detail Extraction
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.detail_layers:
            z1, z2 = layer(z1, z2)
        x_detail = torch.cat((z1, z2), dim=1) # -> [B, 64, H, W]
        
        # Projection to Target Dimension
        x_proj = self.proj(x_detail) # -> [B, 768, H, W]
        
        # Global Average Pooling (Spatial)
        # 输出形状 [B, 768]
        feat = x_proj.mean(dim=[2, 3])
        
        return feat
