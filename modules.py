"""
GAN_Task2 深度调优 - 高级特征增强模块
==============================================
包含:
  1. CBAM 注意力机制 (通道注意力 + 空间注意力)
  2. 可学习边缘检测模块 (Sobel/Laplacian算子)
  3. 拉普拉斯金字塔 (多尺度边缘特征)
  4. 频域处理模块 (FFT幅度/相位处理)
  5. 辅助损失函数 (边缘损失 + 频域损失)

技术选型理由:
  - CBAM: 同时关注"哪些通道重要"和"哪些位置重要"，相比SENet更全面
  - Sobel边缘检测: Canny不可微分，Sobel卷积核可学习且能初始化，适合端到端训练
  - 拉普拉斯金字塔: 多尺度边缘信息，弥补单尺度Sobel的不足
  - FFT频域处理: PyTorch原生支持、完全可微分，比DCT更高效
  - 辅助损失: 通过加权约束生成器，避免模糊输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 1. CBAM 注意力机制 ====================

class ChannelAttention(nn.Module):
    """
    通道注意力模块 (源于SENet思想)
    通过学习每个通道的重要性权重，让模型自动关注关键特征通道
    技术细节: 同时使用平均池化和最大池化，互补地描述通道特征
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # 双池化 → 共享MLP → 融合
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 使用1x1卷积代替全连接层，更高效
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    通过学习每个空间位置的重要性，让模型聚焦于关键区域(如货物主体、障碍物核心)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        # 在通道维度拼接平均池化和最大池化结果
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算统计量
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(attention)


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    串联通道注意力和空间注意力，全面提升特征表达能力
    技术选型理由: 相比SENet(仅通道注意力)，CBAM增加了空间注意力，
    对GAN任务尤其重要，因为生成图像的空间结构需要精细控制
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先通道注意力，再空间注意力 (顺序经论文验证最优)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ==================== 2. 边缘检测模块 ====================

class LearnableEdgeDetector(nn.Module):
    """
    可学习的边缘检测器
    用Sobel和Laplacian算子初始化卷积核，训练过程中自动优化
    技术选型理由: Canny算子涉及非极大值抑制和双阈值，不可微分；
    Sobel/Laplacian是纯卷积操作，可端到端训练，且能自适应学习边缘模式
    调试难点: groups=in_channels确保每个通道独立检测边缘，避免通道间干扰
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # Sobel 水平梯度算子 (检测垂直边缘)
        sobel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]])
        # Sobel 垂直梯度算子 (检测水平边缘)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                 [0.,  0.,  0.],
                                 [1.,  2.,  1.]])
        # Laplacian 算子 (检测各方向边缘，对噪声敏感但全面)
        laplacian = torch.tensor([[0.,  1.,  0.],
                                   [1., -4.,  1.],
                                   [0.,  1.,  0.]])

        # 分组卷积: 每个输入通道独立处理
        self.conv_x = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                groups=in_channels, bias=False)
        self.conv_y = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                groups=in_channels, bias=False)
        self.conv_lap = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                  groups=in_channels, bias=False)

        # 用预设算子初始化 (这些权重随后会在训练中更新)
        with torch.no_grad():
            for i in range(in_channels):
                self.conv_x.weight.data[i, 0] = sobel_x
                self.conv_y.weight.data[i, 0] = sobel_y
                self.conv_lap.weight.data[i, 0] = laplacian

        # 融合三个检测器的输出
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        edge_lap = self.conv_lap(x)
        # 计算边缘强度 (欧几里得范数)
        edge_magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + edge_lap.pow(2) + 1e-8)
        edges = self.fusion(torch.cat([edge_x, edge_y, edge_lap], dim=1))
        return edges


class LaplacianPyramid(nn.Module):
    """
    拉普拉斯金字塔 - 多尺度边缘特征提取
    通过逐级下采样和差分，捕获不同尺度的边缘信息
    技术选型理由: 单一尺度边缘检测会丢失细节，金字塔结构能同时捕获
    粗粒度轮廓和细粒度纹理，对GAN判别器尤为重要
    调试难点: 高斯核的标准差σ需与下采样因子匹配，σ≈1.0对应2倍下采样
    """
    def __init__(self, in_channels=3, levels=2):
        super().__init__()
        self.levels = levels
        # 5x5高斯核，σ≈1.0 (与2倍下采样匹配)
        gaussian_kernel = self._create_gaussian_kernel(5, 1.0)
        self.register_buffer('gaussian_kernel', gaussian_kernel)
        # 每个level后有一个轻量特征提取器
        self.level_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(levels)
        ])

    def _create_gaussian_kernel(self, size, sigma):
        """创建2D高斯卷积核"""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)
        kernel_4d = kernel_2d.expand(3, 1, size, size)
        return kernel_4d

    def forward(self, x):
        pyramid_features = []
        current = x
        for level in range(self.levels):
            # 高斯模糊 + 下采样
            blurred = F.conv2d(current, self.gaussian_kernel,
                               padding=2, groups=current.shape[1])
            down = F.interpolate(blurred, scale_factor=0.5,
                                 mode='bilinear', align_corners=False)
            # 上采样重建
            up = F.interpolate(down, size=current.shape[2:],
                               mode='bilinear', align_corners=False)
            # 拉普拉斯残差 (高频细节 = 原图 - 低频重建)
            laplacian = current - up
            pyramid_features.append(self.level_convs[level](laplacian))
            current = down

        return pyramid_features


# ==================== 3. 频域处理模块 ====================

class FrequencyProcessor(nn.Module):
    """
    频域特征处理模块
    通过FFT将空间域转换至频域，分别处理幅度谱和相位谱后重建
    技术选型理由:
      - FFT是PyTorch原生支持的完全可微操作
      - 幅度谱反映纹理强度/噪声水平，相位谱反映结构/位置信息
      - 分别处理两者可针对性增强纹理细节、抑制频域噪声
    调试难点: FFT产生复数，需正确分离幅度和相位；IFFT结果取实部
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # 幅度谱处理 (纹理/噪声特征)
        self.mag_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )
        # 相位谱处理 (结构/位置特征)
        self.phase_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, in_channels, 3, padding=1),
        )
        # 空间域 + 频域特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 2D FFT
        x_fft = torch.fft.fft2(x)
        mag = torch.abs(x_fft)          # 幅度谱
        phase = torch.angle(x_fft)       # 相位谱

        # 分别处理
        mag_processed = self.mag_conv(mag)
        phase_processed = self.phase_conv(phase)

        # 用处理后的幅度和原始/处理后的相位重建
        x_fft_processed = mag_processed * torch.exp(1j * phase_processed)
        x_recon = torch.fft.ifft2(x_fft_processed).real

        # 融合原始空间域特征和频域处理后的特征
        output = self.fusion(torch.cat([x, x_recon], dim=1))
        return output


# ==================== 4. 辅助损失函数 ====================

class EdgeLoss(nn.Module):
    """
    边缘感知损失
    约束生成图像的边缘分布与真实图像一致，惩罚模糊生成
    技术选型理由: GAN生成图像常见问题是边缘模糊，边缘损失直接
    在特征层面(而非像素层面)施加约束，保留GAN的创造性同时提升清晰度
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.edge_detector = LearnableEdgeDetector(in_channels)

    def forward(self, fake_img, real_img):
        fake_edges = self.edge_detector(fake_img)
        real_edges = self.edge_detector(real_img)
        return F.l1_loss(fake_edges, real_edges)


class FrequencyLoss(nn.Module):
    """
    频域一致性损失
    约束生成图像的频域特征与真实图像一致
    技术选型理由: L1像素损失和对抗损失主要作用在空间域，
    频域损失能补充纹理/周期性特征的约束，减少高频伪影
    调试难点: 仅约束幅度谱(不约束相位)，因为GAN应该有一定创作自由度
    """
    def __init__(self):
        super().__init__()

    def forward(self, fake_img, real_img):
        fake_fft = torch.fft.fft2(fake_img)
        real_fft = torch.fft.fft2(real_img)
        # 仅比较幅度谱，保留相位自由度
        fake_mag = torch.abs(fake_fft)
        real_mag = torch.abs(real_fft)
        # 归一化：除以像素数，使 loss 量级与空间域 L1 可比
        norm = fake_img.shape[2] * fake_img.shape[3]
        return F.l1_loss(fake_mag, real_mag) / norm


# ==================== 5. Self-Attention (SAGAN 风格) ====================

class SelfAttention2d(nn.Module):
    """
    SAGAN 风格的自注意力模块
    让生成器/判别器在任意两个空间位置之间建立长程依赖
    技术原理: 
      - Q(query), K(key), V(value) 三路1×1卷积
      - attention map = softmax(Q^T · K)，尺寸为 N×N (N=H*W)
      - 输出 = attention · V + 原输入（残差连接）
    用于生成器: 帮助捕获全局结构和长程一致性
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        # 1×1卷积生成 Q, K, V
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # 可学习的缩放参数 (初始化为0，让网络先学局部特征再逐渐引入全局注意力)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, H, W = x.shape
        
        # 生成 Q, K, V
        # Q: [B, C//8, H, W] → [B, C//8, N]
        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # [B, N, C//8]
        # K: [B, C//8, H, W] → [B, C//8, N]
        key = self.key(x).view(batch, -1, H * W)  # [B, C//8, N]
        
        # 注意力图: [B, N, N]
        attention = self.softmax(torch.bmm(query, key))  # Q·K
        
        # V: [B, C, H, W] → [B, C, N]
        value = self.value(x).view(batch, -1, H * W)  # [B, C, N]
        
        # 加权求和: [B, C, N] → [B, C, H, W]
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        
        # 残差连接 + 可学习缩放
        return self.gamma * out + x


# ==================== 6. Spectral Normalization 工具 ====================

def add_sn_to_module(module):
    """
    递归给 Module 中所有 Conv2d 和 Linear 层添加 Spectral Normalization
    Spectral Normalization 将每层的谱范数约束为1，防止判别器梯度爆炸
    是 WGAN/WGAN-GP 和 SNGAN 的核心稳定技术
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            setattr(module, name, nn.utils.spectral_norm(child))
        else:
            add_sn_to_module(child)


# ==================== 7. 工具函数 ====================

def compute_edge_density(img_tensor):
    """
    计算图像的边缘密度 (边缘像素占比)
    用于评估生成图像的清晰度
    """
    detector = LearnableEdgeDetector(img_tensor.shape[1])
    detector = detector.to(img_tensor.device)
    with torch.no_grad():
        edges = detector(img_tensor)
        return edges.mean().item()


def compute_high_freq_energy(img_tensor):
    """
    计算图像的高频能量占比
    用于评估生成图像的纹理丰富度
    """
    with torch.no_grad():
        fft = torch.fft.fft2(img_tensor)
        mag = torch.abs(fft)
        h, w = mag.shape[2], mag.shape[3]
        # 计算高频区域(频谱边缘)的能量占比
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4  # 低频区域半径
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(img_tensor.device), x.to(img_tensor.device)
        dist = torch.sqrt((y - center_h).pow(2) + (x - center_w).pow(2))
        high_freq_mask = (dist > radius).float()
        total_energy = mag.sum(dim=(2, 3))
        high_energy = (mag * high_freq_mask).sum(dim=(2, 3))
        ratio = (high_energy / (total_energy + 1e-8)).mean().item()
        return ratio
