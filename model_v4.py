"""
GAN_Task2 V4 - LSGAN + SpectralNorm (Balanced)
===============================================
吸取 V2/V3 教训:
  V2: D 太弱 (6.3M, Hinge+SN) → D 崩溃
  V3: D 太强 (11M, WGAN-GP, 5x更新) → G 被碾压

V4 策略 — 平衡设计:
  1. 强 D 架构 (128→256→512→512, ~6.5M) + SpectralNorm 稳定
  2. G 架构不变 (512→256→128→64, ~3.6M, CBAM+SelfAttn)
  3. LSGAN loss — 比 BCE 稳定，比 WGAN-GP 简单
  4. D/G 更新 1:1 — 完全平衡
  5. 轻度标签平滑 + 实例噪声

LSGAN:
  D_loss = 0.5 * MSE(D(real), 1) + 0.5 * MSE(D(fake), 0)
  G_loss = 0.5 * MSE(D(fake), 1)
  优势: 平滑梯度，不饱和，天然防模式崩溃
"""

import torch
import torch.nn as nn
from modules import CBAM, SelfAttention2d, add_sn_to_module


class GeneratorV4(nn.Module):
    """生成器 — 与 V2/V3 相同架构"""
    def __init__(self, noise_dim=100):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 512, kernel_size=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAM(512)

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.cbam2 = CBAM(256)

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cbam3 = CBAM(128)
        self.attn = SelfAttention2d(128)

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.cbam4 = CBAM(64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cbam1(self.block1(x))
        x = self.cbam2(self.block2(x))
        x = self.cbam3(self.block3(x))
        x = self.attn(x)
        x = self.cbam4(self.block4(x))
        x = self.final(x)
        return x


class DiscriminatorV4(nn.Module):
    """
    判别器 — 适度强化 + SpectralNorm

    通道: 3→128→256→512→512 (比 V2 宽)
    每层 SpectralNorm 稳定训练
    输出 Sigmoid (配合 LSGAN)
    参数量: ~6.5M (D/G ≈ 1.8, 适度优势)
    """
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        # 4×4×512 = 8192
        self.fc = nn.Linear(8192, 1)

        # SpectralNorm 稳定训练
        add_sn_to_module(self)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))   # [B, 128, 32, 32]
        x = self.lrelu(self.conv2(x))   # [B, 256, 16, 16]
        x = self.lrelu(self.conv3(x))   # [B, 512, 8, 8]
        x = self.lrelu(self.conv4(x))   # [B, 512, 4, 4]
        x = self.flatten(x)
        x = self.fc(x)
        return torch.sigmoid(x).view(-1)
