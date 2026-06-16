"""
GAN_Task2 V3 - WGAN-GP 训练脚本
================================
核心改进 vs V2:
  1. WGAN-GP loss — 最稳定的 GAN 变体，防模式崩溃
  2. Critic 5次更新 / Generator 1次 — WGAN-GP 标准配比
  3. 梯度惩罚 (GP) 替代 SpectralNorm — 直接强制 Lipschitz 约束
  4. 无 BN/SN 的纯 critic — 保证 GP 计算正确
  5. 多样性监控 — 每10 epoch 检测 cell_means_std

WGAN-GP 原理:
  Critic loss  = E[D(fake)] - E[D(real)] + λ * GP
  Generator loss = -E[D(fake)]
  GP = E[(||∇_x̂ D(x̂)||₂ - 1)²]  on interpolated samples

使用方法:
  python train_v3.py
  python train_v3.py --epochs 800 --n_critic 3
"""

import sys, os
import torch
# === 强制 GPU 检查 ===
if not torch.cuda.is_available():
    raise RuntimeError(
        f"\nCUDA 不可用！当前 Python: {sys.executable}\n"
        f"请改用 .venv\\Scripts\\python.exe 运行\n"
    )
print(f"GPU: {torch.cuda.get_device_name(0)}")
# =====================
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import argparse
import time
import math

from create_dataset import My_dataset, save_img
from model_v3 import GeneratorV3, DiscriminatorV3
from modules import FrequencyLoss, compute_edge_density, compute_high_freq_energy


# ==================== WGAN-GP 损失函数 ====================

def critic_loss_fn(real_out, fake_out):
    """
    Critic 损失
    Critic 希望最小化: D(fake) - D(real)
    即希望 D(real) 尽可能大，D(fake) 尽可能小
    """
    return fake_out.mean() - real_out.mean()


def generator_loss_fn(fake_out):
    """
    Generator 损失
    Generator 希望最小化: -D(fake)
    即希望 D(fake) 尽可能大
    """
    return -fake_out.mean()


def compute_gradient_penalty(critic, real_imgs, fake_imgs):
    """
    WGAN-GP 梯度惩罚
    在真实和生成图像的随机插值点上，
    强制 critic 梯度范数接近 1.0

    技术细节:
      - 逐样本随机插值系数 ε
      - 插值点 x̂ = ε * real + (1-ε) * fake
      - GP = (||∇_x̂ D(x̂)||₂ - 1)² 的均值
    """
    batch_size = real_imgs.size(0)
    device = real_imgs.device

    # 逐样本随机插值系数
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_imgs + (1 - epsilon) * fake_imgs
    interpolated.requires_grad_(True)

    # Critic 输出
    critic_output = critic(interpolated)

    # 计算 D(x̂) 对 x̂ 的梯度
    gradients = torch.autograd.grad(
        outputs=critic_output,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # 逐样本梯度 L2 范数
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # (||grad|| - 1)²
    gp = ((gradient_norm - 1.0) ** 2).mean()
    return gp


# ==================== 学习率调度 ====================

class WarmupCosineScheduler:
    """线性 warmup → 余弦退火"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# ==================== 多样性监控 ====================

@torch.no_grad()
def compute_diversity_metrics(generator, device, num_samples=64, noise_dim=100):
    """
    计算模式崩溃检测指标

    cell_means_std: 64 张生成图像的像素均值标准差
      > 15: 健康多样性
      5-15: 警告
      < 5: 严重模式崩溃

    neighbor_diff_mean: 相邻图像对之间的平均差异
      > 5.0: 健康
      < 2.0: 模式崩溃
    """
    noise = torch.randn(num_samples, noise_dim, 1, 1, device=device)
    fake = generator(noise)

    # 反归一化: [-1,1] → [0,1]
    fake_np = (fake * 0.5 + 0.5).clamp(0, 1)

    # 每张图的平均灰度
    cell_means = fake_np.mean(dim=[1, 2, 3]).cpu().numpy()

    # 相邻图像差异 (逐对比较)
    neighbor_diffs = []
    for i in range(0, num_samples - 1, 2):
        diff = torch.abs(fake[i] - fake[i + 1]).mean().item()
        neighbor_diffs.append(diff)

    return {
        'cell_means_mean': float(cell_means.mean()),
        'cell_means_std': float(cell_means.std()),
        'cell_means_min': float(cell_means.min()),
        'cell_means_max': float(cell_means.max()),
        'neighbor_diff_mean': float(sum(neighbor_diffs) / len(neighbor_diffs)),
    }


# ==================== 训练函数 ====================

def train_v3(generator, critic, dataloader, config):
    """
    WGAN-GP 训练

    每 iteration:
      1. Critic 更新 n_critic 次 (默认 5)
         - 每次用新鲜噪声
         - GP 在 real/fake 插值上计算
      2. Generator 更新 1 次
         - 仅对抗损失 (可选 freq 辅助)

    WGAN-GP 核心指标:
      - GP 值应在 0.1~2.0 范围
      - D(real) 应始终 > D(fake)
      - 健康的 WGAN-GP 中 D_loss 通常在 0 附近振荡
    """
    device = torch.device('cuda')
    generator = generator.to(device)
    critic = critic.to(device)

    epochs = config['epochs']
    batch_size = config['batch_size']
    n_critic = config.get('n_critic', 5)
    g_lr = config.get('g_lr', 1e-4)
    d_lr = config.get('d_lr', 1e-4)
    lambda_gp = config.get('lambda_gp', 10)
    lambda_freq = config.get('lambda_freq', 0.0)
    noise_dim = config.get('noise_dim', 100)
    warmup_epochs = config.get('warmup_epochs', 20)
    diversity_interval = config.get('diversity_interval', 10)
    sample_dir = config['sample_dir']
    weight_dir = config['weight_dir']

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    # 优化器 — WGAN-GP 标准 betas (0.0, 0.9)
    d_optimizer = torch.optim.Adam(critic.parameters(), lr=d_lr, betas=(0.0, 0.9))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.0, 0.9))

    # LR 调度器
    d_scheduler = WarmupCosineScheduler(d_optimizer, warmup_epochs, epochs, d_lr)
    g_scheduler = WarmupCosineScheduler(g_optimizer, warmup_epochs, epochs, g_lr)

    # 可选 freq 辅助损失 (默认关闭)
    freq_loss_fn = FrequencyLoss().to(device) if lambda_freq > 0 else None

    # 指标记录
    metrics = {
        'epoch': [], 'd_loss': [], 'g_loss': [],
        'd_real_mean': [], 'd_fake_mean': [], 'gp_mean': [],
        'cell_means_std': [], 'neighbor_diff_mean': [],
        'edge_density': [], 'hf_energy': [], 'time_per_epoch': [],
    }

    print(f"\n{'='*60}")
    print(f"V3 WGAN-GP 训练")
    print(f"设备: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"学习率: G={g_lr}, D={d_lr} | D步数: {n_critic} | GP λ={lambda_gp}")
    print(f"Warmup: {warmup_epochs}ep | Freq辅助: {lambda_freq}")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        epoch_gp = 0.0
        num_batches = 0

        for i, real_imgs in enumerate(dataloader):
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # ===== 训练 Critic (n_critic 次) =====
            for step in range(n_critic):
                noise = torch.randn(bs, noise_dim, 1, 1, device=device)
                with torch.no_grad():
                    fake_imgs = generator(noise)

                real_out = critic(real_imgs)
                fake_out = critic(fake_imgs)

                # 梯度惩罚
                gp = compute_gradient_penalty(critic, real_imgs, fake_imgs)

                # Critic loss
                d_loss = critic_loss_fn(real_out, fake_out) + lambda_gp * gp

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # ===== 训练 Generator (1 次) =====
            noise = torch.randn(bs, noise_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            fake_out = critic(fake_imgs)

            g_loss = generator_loss_fn(fake_out)

            # 可选 freq 辅助损失
            g_freq_val = 0.0
            if freq_loss_fn is not None:
                g_freq_val = freq_loss_fn(fake_imgs, real_imgs).item()
                g_loss = g_loss + lambda_freq * freq_loss_fn(fake_imgs, real_imgs)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # 记录 (用最后一次 critic step 的值)
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_real += real_out.mean().item()
            epoch_d_fake += fake_out.mean().item()
            epoch_gp += gp.item()
            num_batches += 1

            # 每 5 batch 打印
            if (i + 1) % 5 == 0:
                print(f'[V3] Epoch[{epoch+1}/{epochs}] Batch[{i+1}] '
                      f'D:{d_loss.item():.4f} G:{g_loss.item():.4f} '
                      f'GP:{gp.item():.3f} '
                      f'DR:{real_out.mean():.3f} DF:{fake_out.mean():.3f}')

        # ===== Epoch 结束 =====
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches
        avg_gp = epoch_gp / num_batches

        metrics['epoch'].append(epoch + 1)
        metrics['d_loss'].append(avg_d_loss)
        metrics['g_loss'].append(avg_g_loss)
        metrics['d_real_mean'].append(avg_d_real)
        metrics['d_fake_mean'].append(avg_d_fake)
        metrics['gp_mean'].append(avg_gp)

        # 图像质量
        with torch.no_grad():
            edge_dens = compute_edge_density(fake_imgs)
            hf_energy = compute_high_freq_energy(fake_imgs)
        metrics['edge_density'].append(edge_dens)
        metrics['hf_energy'].append(hf_energy)

        epoch_time = time.time() - epoch_start
        metrics['time_per_epoch'].append(epoch_time)

        # 学习率更新
        d_lr_now = d_scheduler.step(epoch)
        g_lr_now = g_scheduler.step(epoch)

        # 多样性监控
        div_metrics = {}
        if (epoch + 1) % diversity_interval == 0:
            div_metrics = compute_diversity_metrics(generator, device,
                                                     noise_dim=noise_dim)
            metrics['cell_means_std'].append(div_metrics['cell_means_std'])
            metrics['neighbor_diff_mean'].append(div_metrics['neighbor_diff_mean'])

            if div_metrics['cell_means_std'] < 5:
                print(f'[V3] ⚠️  警告: cell_means_std={div_metrics["cell_means_std"]:.2f}'
                      f' — 可能出现模式崩溃!')
        else:
            # 非检测 epoch 用占位
            metrics['cell_means_std'].append(0)
            metrics['neighbor_diff_mean'].append(0)

        # 保存样本和权重
        if epoch == 0:
            save_img(real_imgs[:64], f'{sample_dir}/real_images.png')

        if (epoch + 1) % 50 == 0:
            save_img(fake_imgs[:64], f'{sample_dir}/fake_images_{epoch+1}.png')
            torch.save(generator.state_dict(),
                       f'{weight_dir}/generator_v3_epoch{epoch+1}.pth')
            torch.save(critic.state_dict(),
                       f'{weight_dir}/discriminator_v3_epoch{epoch+1}.pth')

        # Epoch 摘要
        div_str = ''
        if div_metrics:
            div_str = f' div_std={div_metrics["cell_means_std"]:.2f}'
        print(f'[V3] Epoch [{epoch+1}/{epochs}] '
              f'D:{avg_d_loss:.4f} G:{avg_g_loss:.4f} '
              f'GP:{avg_gp:.3f} '
              f'DR:{avg_d_real:.3f} DF:{avg_d_fake:.3f} '
              f'LR:{g_lr_now:.2e}{div_str} '
              f'Time:{epoch_time:.1f}s')

    total_time = time.time() - total_start
    print(f'\n[V3] 训练完成! 总耗时: {total_time/60:.1f}分钟')

    # 保存最终权重
    torch.save(generator.state_dict(), f'{weight_dir}/generator_v3.pth')
    torch.save(critic.state_dict(), f'{weight_dir}/discriminator_v3.pth')

    # 保存指标
    with open(f'{weight_dir}/metrics_v3.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'指标已保存: {weight_dir}/metrics_v3.json')

    return metrics


def main():
    parser = argparse.ArgumentParser(description='GAN V3 WGAN-GP 训练')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n_critic', type=int, default=5,
                        help='每 batch Critic 更新次数')
    parser.add_argument('--lambda_gp', type=float, default=10,
                        help='梯度惩罚权重')
    parser.add_argument('--lambda_freq', type=float, default=0.0,
                        help='频域辅助损失权重 (0=关闭)')
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--diversity_interval', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='./data/1_enhanced')
    args = parser.parse_args()

    # 切换到脚本所在目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # 数据准备
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = My_dataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                            shuffle=True, drop_last=True)
    print(f'数据集: {len(dataset)} 张, {len(dataloader)} 批次')

    # 模型
    generator = GeneratorV3()
    critic = DiscriminatorV3()

    # 打印参数量
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in critic.parameters())
    print(f'G 参数: {g_params/1e6:.1f}M | D 参数: {d_params/1e6:.1f}M | 比值: {g_params/d_params:.2f}')

    # 配置
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'n_critic': args.n_critic,
        'g_lr': args.g_lr,
        'd_lr': args.d_lr,
        'lambda_gp': args.lambda_gp,
        'lambda_freq': args.lambda_freq,
        'warmup_epochs': args.warmup_epochs,
        'diversity_interval': args.diversity_interval,
        'sample_dir': './sample/v3',
        'weight_dir': './weights_v3',
    }

    train_v3(generator, critic, dataloader, config)

    print('\n训练完成! 运行以下命令做评估:')
    print('  python compare_results_v2.py --gen_path ./weights_v3/generator_v3.pth')


if __name__ == '__main__':
    main()
