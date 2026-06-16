"""
GAN_Task2 V2 - Hinge Loss 训练脚本
===================================
核心改进 vs 增强版:
  1. Hinge Adversarial Loss (替代 BCELoss) — 根本性解决 mode collapse
  2. D更新2次 / G更新1次 — 纠正之前的G多步更新（G多步会加剧collapse）
  3. Spectral Normalization 判别器 — 稳定训练，防止D梯度爆炸
  4. 学习率 Warmup + Cosine 衰减
  5. 轻量辅助损失 (edge=0.05, freq=0.02) — 仅做微调约束
  6. 标签平滑 + 实例噪声 — 防D过拟合

Hinge Loss 原理:
  D_loss = ReLU(1 - D(real)) + ReLU(1 + D(fake))
  G_loss = -D(fake)
  直观理解: D希望真图得分>1, 假图得分<-1; G希望假图得分越高越好

使用方法:
  python train_v2.py
  python train_v2.py --epochs 800 --lambda_edge 0.03
"""

import sys, os
# === 强制 GPU 检查 ===
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        f"\n❌ CUDA 不可用！\n"
        f"   当前 Python: {sys.executable}\n"
        f"   请改用: {os.path.dirname(__file__)}\\.venv\\Scripts\\python.exe {sys.argv[0]}\n"
    )
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
# =====================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import os
import argparse
import time
import math

from create_dataset import My_dataset, save_img
from model_v2 import GeneratorV2, DiscriminatorV2
from modules import EdgeLoss, FrequencyLoss, compute_edge_density, compute_high_freq_energy


# ==================== Hinge Loss ====================

def hinge_loss_d(real_out, fake_out):
    """
    判别器 Hinge Loss
    D 希望: D(real) ≥ 1, D(fake) ≤ -1
    返回标量 loss
    """
    real_loss = F.relu(1.0 - real_out).mean()
    fake_loss = F.relu(1.0 + fake_out).mean()
    return real_loss + fake_loss


def hinge_loss_g(fake_out):
    """
    生成器 Hinge Loss
    G 希望: D(fake) 尽可能大
    """
    return -fake_out.mean()


# ==================== 学习率调度 ====================

class WarmupCosineScheduler:
    """
    线性 warmup → 余弦退火
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 线性 warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ==================== 实例噪声 ====================

def add_instance_noise(img, std=0.02):
    """向图像添加小高斯噪声，防止判别器像素级记忆"""
    noise = torch.randn_like(img) * std
    return img + noise


# ==================== 训练函数 ====================

def train_v2(generator, discriminator, dataloader, config):
    """
    V2 Hinge Loss 训练
    
    训练策略:
      - 每个 iteration: D更新d_steps次 → G更新1次
      - Hinge loss 天然防mode collapse
      - Spectral Norm 稳定D
      - 辅助损失仅微调 (edge=0.05, freq=0.02)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    epochs = config['epochs']
    batch_size = config['batch_size']
    d_steps = config.get('d_steps', 2)
    g_lr = config.get('g_lr', 2e-4)
    d_lr = config.get('d_lr', 2e-4)
    warmup_epochs = config.get('warmup_epochs', 20)
    lambda_edge = config.get('lambda_edge', 0.05)
    lambda_freq = config.get('lambda_freq', 0.02)
    noise_std = config.get('noise_std', 0.02)
    sample_dir = config['sample_dir']
    weight_dir = config['weight_dir']

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    # 优化器
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.0, 0.9))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))

    # 学习率调度器
    d_scheduler = WarmupCosineScheduler(d_optimizer, warmup_epochs, epochs, d_lr)
    g_scheduler = WarmupCosineScheduler(g_optimizer, warmup_epochs, epochs, g_lr)

    # 辅助损失
    edge_loss_fn = EdgeLoss().to(device)
    freq_loss_fn = FrequencyLoss().to(device)

    # 指标记录
    metrics = {
        'epoch': [], 'd_loss': [], 'g_loss': [], 'g_adv_loss': [],
        'g_edge_loss': [], 'g_freq_loss': [], 'd_real_mean': [], 'd_fake_mean': [],
        'edge_density': [], 'hf_energy': [], 'time_per_epoch': [],
    }

    print(f"\n{'='*60}")
    print(f"V2 Hinge Loss 训练")
    print(f"设备: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"学习率: G={g_lr}, D={d_lr} | D步数: {d_steps} | Warmup: {warmup_epochs}ep")
    print(f"辅助损失: edge={lambda_edge}, freq={lambda_freq} | 噪声σ={noise_std}")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_g_adv = 0.0
        epoch_g_edge = 0.0
        epoch_g_freq = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        num_batches = 0

        for i, img in enumerate(dataloader):
            bs = img.size(0)
            real_img = img.to(device)

            # ===== 训练判别器 (d_steps 次) =====
            for step in range(d_steps):
                noise = torch.randn(bs, 100, 1, 1, device=device)
                with torch.no_grad():
                    fake_img = generator(noise)

                # 实例噪声
                real_noisy = add_instance_noise(real_img, noise_std)
                fake_noisy = add_instance_noise(fake_img, noise_std)

                real_out = discriminator(real_noisy)
                fake_out = discriminator(fake_noisy)

                d_loss = hinge_loss_d(real_out, fake_out)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # ===== 训练生成器 (1 次) =====
            noise = torch.randn(bs, 100, 1, 1, device=device)
            fake_img = generator(noise)
            fake_out = discriminator(fake_img)

            # 对抗损失
            g_adv_loss = hinge_loss_g(fake_out)

            # 辅助损失 (轻量)
            g_edge_loss = edge_loss_fn(fake_img, real_img)
            g_freq_loss = freq_loss_fn(fake_img, real_img)

            g_loss = g_adv_loss + lambda_edge * g_edge_loss + lambda_freq * g_freq_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # 记录
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_g_adv += g_adv_loss.item()
            epoch_g_edge += g_edge_loss.item()
            epoch_g_freq += g_freq_loss.item()
            epoch_d_real += real_out.mean().item()
            epoch_d_fake += fake_out.mean().item()
            num_batches += 1

            if (i + 1) % 5 == 0:
                print(f'[V2] Epoch[{epoch+1}/{epochs}] Batch[{i+1}] '
                      f'D:{d_loss.item():.4f} G_adv:{g_adv_loss.item():.4f} '
                      f'E:{g_edge_loss.item():.4f} F:{g_freq_loss.item():.4f} '
                      f'DR:{real_out.mean():.3f} DF:{fake_out.mean():.3f}')

        # ===== Epoch 结束 =====
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_g_adv = epoch_g_adv / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches

        metrics['epoch'].append(epoch + 1)
        metrics['d_loss'].append(avg_d_loss)
        metrics['g_loss'].append(avg_g_loss)
        metrics['g_adv_loss'].append(avg_g_adv)
        metrics['g_edge_loss'].append(epoch_g_edge / num_batches)
        metrics['g_freq_loss'].append(epoch_g_freq / num_batches)
        metrics['d_real_mean'].append(avg_d_real)
        metrics['d_fake_mean'].append(avg_d_fake)

        # 质量指标
        with torch.no_grad():
            edge_dens = compute_edge_density(fake_img)
            hf_energy = compute_high_freq_energy(fake_img)
        metrics['edge_density'].append(edge_dens)
        metrics['hf_energy'].append(hf_energy)

        epoch_time = time.time() - epoch_start
        metrics['time_per_epoch'].append(epoch_time)

        # 学习率
        d_lr_now = d_scheduler.step(epoch)
        g_lr_now = g_scheduler.step(epoch)

        # 保存样本和权重
        if epoch == 0:
            save_img(real_img[:64], f'{sample_dir}/real_images.png')

        if (epoch + 1) % 50 == 0:
            save_img(fake_img[:64], f'{sample_dir}/fake_images_{epoch+1}.png')
            torch.save(generator.state_dict(),
                       f'{weight_dir}/generator_v2_epoch{epoch+1}.pth')
            torch.save(discriminator.state_dict(),
                       f'{weight_dir}/discriminator_v2_epoch{epoch+1}.pth')

        print(f'[V2] Epoch [{epoch+1}/{epochs}] '
              f'D:{avg_d_loss:.4f} G:{avg_g_loss:.4f} '
              f'DR:{avg_d_real:.3f} DF:{avg_d_fake:.3f} '
              f'LR_d:{d_lr_now:.2e} LR_g:{g_lr_now:.2e} '
              f'Time:{epoch_time:.1f}s')

    total_time = time.time() - total_start
    print(f'\n[V2] 训练完成! 总耗时: {total_time/60:.1f}分钟')

    # 保存最终权重
    torch.save(generator.state_dict(), f'{weight_dir}/generator_v2.pth')
    torch.save(discriminator.state_dict(), f'{weight_dir}/discriminator_v2.pth')

    # 保存指标
    with open(f'{weight_dir}/metrics_v2.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='GAN V2 Hinge Loss 训练')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--d_steps', type=int, default=2, help='每轮D更新次数')
    parser.add_argument('--lambda_edge', type=float, default=0.05, help='边缘损失权重')
    parser.add_argument('--lambda_freq', type=float, default=5.0, help='频域损失权重（已归一化，推荐 3-10）')
    parser.add_argument('--noise_std', type=float, default=0.02, help='实例噪声标准差')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='生成器学习率')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='判别器学习率')
    parser.add_argument('--data_dir', type=str, default=None, help='数据集路径')
    args = parser.parse_args()

    # 项目根目录 (脚本所在目录)
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)  # 切换到项目目录，后续相对路径都基于此

    # 数据准备 (保持64×64)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),  # 在线增强
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = args.data_dir or './data/1_enhanced'
    dataset = My_dataset(data_dir, transform=transform)
    batch_size = 32
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
    print(f'数据集: {len(dataset)} 张, {len(dataloader)} 批次')

    # 模型
    generator = GeneratorV2()
    discriminator = DiscriminatorV2()

    # 配置
    config = {
        'epochs': args.epochs,
        'batch_size': batch_size,
        'd_steps': args.d_steps,
        'g_lr': args.g_lr,
        'd_lr': args.d_lr,
        'warmup_epochs': 20,
        'lambda_edge': args.lambda_edge,
        'lambda_freq': args.lambda_freq,  # 已归一化，默认 5.0
        'noise_std': args.noise_std,
        'sample_dir': './sample/v2',
        'weight_dir': './weights_v2',
    }

    train_v2(generator, discriminator, dataloader, config)

    print('\n训练完成! 运行以下命令做对比:')
    print('  python compare_results_v2.py')


if __name__ == '__main__':
    main()
