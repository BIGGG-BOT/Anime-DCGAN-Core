"""
GAN_Task2 V3 - WGAN-GP for Mode Collapse Prevention
====================================================
Core fix: Replace Hinge Loss + SpectralNorm with WGAN-GP.
D becomes a strong "critic" with:
  - No Sigmoid output (scalar critic score, unbounded)
  - No SpectralNorm (gradient penalty replaces it)
  - No BatchNorm/LayerNorm (clean path for GP computation)

Architecture:
  GeneratorV3:  100 → 512×4×4 → 256×8×8 → 128×16×16(SelfAttn) → 64×32×32 → 3×64×64
                CBAM at each level, SelfAttention at 16×16 (kept from V2)
  DiscriminatorV3:  3×64×64 → 128×32×32 → 256×16×16 → 512×8×8 → 1024×4×4 → 1
                    Pure Conv+LeakyReLU, no normalization layers
"""

import torch
import torch.nn as nn
from modules import CBAM, SelfAttention2d


class GeneratorV3(nn.Module):
    """
    生成器 V3 — 保留 V2 架构不变

    尺寸变化: 1×1 → 4×4 → 8×8 → 16×16 → 32×32 → 64×64
    通道:    100 → 512 → 256 → 128 → 64 → 3
    CBAM 每个上采样块后增强特征
    SelfAttention 在 16×16 分辨率捕获全局结构
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
        self.attn = SelfAttention2d(128)  # 16×16 捕获全局结构

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
        x = self.cbam1(self.block1(x))       # [B, 512, 4, 4]
        x = self.cbam2(self.block2(x))       # [B, 256, 8, 8]
        x = self.cbam3(self.block3(x))       # [B, 128, 16, 16]
        x = self.attn(x)                      # Self-Attention
        x = self.cbam4(self.block4(x))       # [B, 64, 32, 32]
        x = self.final(x)                     # [B, 3, 64, 64]
        return x


class DiscriminatorV3(nn.Module):
    """
    WGAN-GP 判别器 (Critic)

    设计原则:
      - 无 BatchNorm (破坏 per-sample Lipschitz 约束)
      - 无 SpectralNorm (梯度惩罚替代)
      - 无 Sigmoid (WGAN 需要无界输出)
      - 纯 Conv + LeakyReLU，梯度惩罚提供稳定性

    尺寸: 64 → 32 → 16 → 8 → 4
    通道: 3 → 128 → 256 → 512 → 1024 → 1
    参数量: ~11M (接近 G 的 14.6M)
    """
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        # 4×4×1024 = 16384
        self.fc = nn.Linear(16384, 1)

        # 无 BN, 无 SN, 无 Sigmoid — 纯 WGAN-GP critic

    def forward(self, x):
        """返回标量 critic score，形状 [B]"""
        x = self.lrelu(self.conv1(x))   # [B, 128, 32, 32]
        x = self.lrelu(self.conv2(x))   # [B, 256, 16, 16]
        x = self.lrelu(self.conv3(x))   # [B, 512, 8, 8]
        x = self.lrelu(self.conv4(x))   # [B, 1024, 4, 4]
        x = self.flatten(x)              # [B, 16384]
        x = self.fc(x)                   # [B, 1]
        return x.view(-1)
