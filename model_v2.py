"""
GAN_Task2 V2 - 修复模式坍塌 + 低对比度
=========================================
设计原则:
  1. 判别器保持简洁——标准 DCGAN + Spectral Normalization，不加多支结构
     (三支判别器过强是模式坍塌的根因)
  2. 增强模块全放在生成器侧——CBAM + Self-Attention 帮 G 生成更好图像
  3. 判别器输出标量（无Sigmoid），配合 Hinge Loss 训练
  4. 通道数加宽，提升模型容量

架构:
  GeneratorV2: 100→512×4×4→256×8×8→128×16×16(SelfAttn)→64×32×32→32×64×64→3
               每级后加 CBAM 注意力
  DiscriminatorV2: 3×64×64→64→128→256→512→1024→1 (SpectralNorm 每一层)
"""

import torch
import torch.nn as nn
from modules import CBAM, SelfAttention2d, add_sn_to_module


class GeneratorV2(nn.Module):
    """
    增强版生成器 V2
    - 加宽通道: 256→128→64→32 (每层都是原版的2倍宽度)
    - CBAM 注意力在每个上采样块后
    - SelfAttention 在 16×16 分辨率捕获全局结构
    - 输出: 64×64×3 Tanh

    尺寸变化: 1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64
    """
    def __init__(self, noise_dim=100):
        super().__init__()

        # Block 1: 100×1×1 → 512×4×4
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, kernel_size=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM(512)

        # Block 2: 512×4×4 → 256×8×8
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM(256)

        # Block 3: 256×8×8 → 128×16×16
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM(128)
        # Self-Attention at 16×16 (关键分辨率，捕获全局结构)
        self.attn = SelfAttention2d(128)

        # Block 4: 128×16×16 → 64×32×32
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.cbam4 = CBAM(64)

        # 输出层: 64×32×32 → 3×64×64
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cbam1(self.block1(x))           # [B, 512, 4, 4]
        x = self.cbam2(self.block2(x))           # [B, 256, 8, 8]
        x = self.cbam3(self.block3(x))           # [B, 128, 16, 16]
        x = self.attn(x)                          # Self-Attention
        x = self.cbam4(self.block4(x))           # [B, 64, 32, 32]
        x = self.final(x)                         # [B, 3, 64, 64]
        return x


class DiscriminatorV2(nn.Module):
    """
    简化版判别器 V2
    - 标准 DCGAN 结构，不加多支融合
    - 每层 Spectral Normalization 稳定训练
    - 输出标量（无 Sigmoid），配合 Hinge Loss
    - 通道数: 3→64→128→256→512
    - 尺寸: 64→32→16→8→4, FC: 4×4×512=8192→1
    """
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 4 * 512, 1)

        # 对 Conv 和 Linear 层施加 Spectral Normalization
        add_sn_to_module(self)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))    # [B, 64, 32, 32]
        x = self.lrelu(self.conv2(x))    # [B, 128, 16, 16]
        x = self.lrelu(self.conv3(x))    # [B, 256, 8, 8]
        x = self.lrelu(self.conv4(x))    # [B, 512, 4, 4]
        x = self.flatten(x)              # [B, 8192]
        x = self.fc(x)                   # [B, 1]
        return x.view(-1)
