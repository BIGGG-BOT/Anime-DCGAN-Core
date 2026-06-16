"""
GAN_Task2 V4 - LSGAN 训练脚本
==============================
LSGAN (Least Squares GAN):
  D_loss = 0.5 * MSE(D(real), a) + 0.5 * MSE(D(fake), b)
  G_loss = 0.5 * MSE(D(fake), a)
  其中 a=1.0 (真标签), b=0.0 (假标签)

为什么 LSGAN:
  1. 平滑梯度 — 不饱和，D 再自信也给可用梯度
  2. 天然防模式崩溃 — 惩罚远离目标的输出
  3. 比 WGAN-GP 简单，不需要梯度惩罚计算

训练策略:
  - D/G 更新 1:1, 完全平衡
  - SpectralNorm 稳定 D
  - 轻度实例噪声 (σ=0.01)
  - 标签平滑 (real=0.9, fake=0.1)
  - 可选频域辅助损失 (默认关闭)

使用方法:
  python train_v4.py
  python train_v4.py --epochs 800 --d_lr 5e-5
"""

import sys, os
import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        f"\nCUDA unavailable! Python: {sys.executable}\n"
        f"Use: .venv\\Scripts\\python.exe\n"
    )
print(f"GPU: {torch.cuda.get_device_name(0)}")
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import json, argparse, time, math

from create_dataset import My_dataset, save_img
from model_v4 import GeneratorV4, DiscriminatorV4
from modules import FrequencyLoss, compute_edge_density, compute_high_freq_energy


# ==================== LSGAN 损失 ====================

def lsgan_d_loss(real_out, fake_out, real_label=0.9, fake_label=0.1):
    """D 希望 D(real)→real_label, D(fake)→fake_label"""
    real_loss = F.mse_loss(real_out, torch.full_like(real_out, real_label))
    fake_loss = F.mse_loss(fake_out, torch.full_like(fake_out, fake_label))
    return 0.5 * (real_loss + fake_loss)


def lsgan_g_loss(fake_out, real_label=0.9):
    """G 希望 D(fake)→real_label"""
    return 0.5 * F.mse_loss(fake_out, torch.full_like(fake_out, real_label))


# ==================== 学习率调度 ====================

class WarmupCosineScheduler:
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
    noise = torch.randn(num_samples, noise_dim, 1, 1, device=device)
    fake = generator(noise)
    fake_np = (fake * 0.5 + 0.5).clamp(0, 1)
    cell_means = fake_np.mean(dim=[1, 2, 3]).cpu().numpy()

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


# ==================== 实例噪声 ====================

def add_noise(img, std=0.01):
    return img + torch.randn_like(img) * std


# ==================== 训练 ====================

def train_v4(generator, discriminator, dataloader, config):
    device = torch.device('cuda')
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    epochs = config['epochs']
    batch_size = config['batch_size']
    g_lr = config.get('g_lr', 1e-4)
    d_lr = config.get('d_lr', 1e-4)
    warmup_epochs = config.get('warmup_epochs', 20)
    noise_std = config.get('noise_std', 0.01)
    lambda_freq = config.get('lambda_freq', 0.0)
    noise_dim = config.get('noise_dim', 100)
    diversity_interval = config.get('diversity_interval', 10)
    label_smooth_real = config.get('label_smooth_real', 0.9)
    label_smooth_fake = config.get('label_smooth_fake', 0.1)
    sample_dir = config['sample_dir']
    weight_dir = config['weight_dir']

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))

    d_scheduler = WarmupCosineScheduler(d_optimizer, warmup_epochs, epochs, d_lr)
    g_scheduler = WarmupCosineScheduler(g_optimizer, warmup_epochs, epochs, g_lr)

    freq_loss_fn = FrequencyLoss().to(device) if lambda_freq > 0 else None

    metrics = {
        'epoch': [], 'd_loss': [], 'g_loss': [],
        'd_real_mean': [], 'd_fake_mean': [],
        'cell_means_std': [], 'neighbor_diff_mean': [],
        'edge_density': [], 'hf_energy': [], 'time_per_epoch': [],
    }

    print(f"\n{'='*60}")
    print(f"V4 LSGAN 训练")
    print(f"设备: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"学习率: G={g_lr}, D={d_lr} | 噪声σ={noise_std}")
    print(f"标签平滑: real={label_smooth_real}, fake={label_smooth_fake}")
    print(f"更新: D×1 G×1 (完全平衡) | Warmup: {warmup_epochs}ep")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        num_batches = 0

        for i, real_imgs in enumerate(dataloader):
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # ===== 训练 D (1 次) =====
            noise = torch.randn(bs, noise_dim, 1, 1, device=device)
            with torch.no_grad():
                fake_imgs = generator(noise)

            real_noisy = add_noise(real_imgs, noise_std)
            fake_noisy = add_noise(fake_imgs, noise_std)

            real_out = discriminator(real_noisy)
            fake_out = discriminator(fake_noisy)

            d_loss = lsgan_d_loss(real_out, fake_out,
                                  label_smooth_real, label_smooth_fake)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ===== 训练 G (1 次) =====
            noise = torch.randn(bs, noise_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            fake_out = discriminator(fake_imgs)

            g_loss = lsgan_g_loss(fake_out, label_smooth_real)

            if freq_loss_fn is not None:
                g_loss = g_loss + lambda_freq * freq_loss_fn(fake_imgs, real_imgs)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_real += real_out.mean().item()
            epoch_d_fake += fake_out.mean().item()
            num_batches += 1

            if (i + 1) % 5 == 0:
                print(f'[V4] Epoch[{epoch+1}/{epochs}] Batch[{i+1}] '
                      f'D:{d_loss.item():.4f} G:{g_loss.item():.4f} '
                      f'DR:{real_out.mean():.3f} DF:{fake_out.mean():.3f}')

        # ===== Epoch 结束 =====
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_real = epoch_d_real / num_batches
        avg_d_fake = epoch_d_fake / num_batches

        metrics['epoch'].append(epoch + 1)
        metrics['d_loss'].append(avg_d_loss)
        metrics['g_loss'].append(avg_g_loss)
        metrics['d_real_mean'].append(avg_d_real)
        metrics['d_fake_mean'].append(avg_d_fake)

        with torch.no_grad():
            metrics['edge_density'].append(compute_edge_density(fake_imgs))
            metrics['hf_energy'].append(compute_high_freq_energy(fake_imgs))

        epoch_time = time.time() - epoch_start
        metrics['time_per_epoch'].append(epoch_time)

        d_lr_now = d_scheduler.step(epoch)
        g_lr_now = g_scheduler.step(epoch)

        # 多样性监控
        div_metrics = {}
        if (epoch + 1) % diversity_interval == 0:
            div_metrics = compute_diversity_metrics(generator, device)
            metrics['cell_means_std'].append(div_metrics['cell_means_std'])
            metrics['neighbor_diff_mean'].append(div_metrics['neighbor_diff_mean'])
            if div_metrics['cell_means_std'] < 5:
                print(f'[V4] WARNING: cell_means_std={div_metrics["cell_means_std"]:.2f}'
                      f' - possible mode collapse!')
        else:
            metrics['cell_means_std'].append(0)
            metrics['neighbor_diff_mean'].append(0)

        if epoch == 0:
            save_img(real_imgs[:64], f'{sample_dir}/real_images.png')

        if (epoch + 1) % 50 == 0:
            save_img(fake_imgs[:64], f'{sample_dir}/fake_images_{epoch+1}.png')
            torch.save(generator.state_dict(),
                       f'{weight_dir}/generator_v4_epoch{epoch+1}.pth')
            torch.save(discriminator.state_dict(),
                       f'{weight_dir}/discriminator_v4_epoch{epoch+1}.pth')

        div_str = f' div_std={div_metrics["cell_means_std"]:.2f}' if div_metrics else ''
        print(f'[V4] Epoch [{epoch+1}/{epochs}] '
              f'D:{avg_d_loss:.4f} G:{avg_g_loss:.4f} '
              f'DR:{avg_d_real:.3f} DF:{avg_d_fake:.3f} '
              f'LR:{g_lr_now:.2e}{div_str} '
              f'Time:{epoch_time:.1f}s')

    total_time = time.time() - total_start
    print(f'\n[V4] 训练完成! 总耗时: {total_time/60:.1f}分钟')

    torch.save(generator.state_dict(), f'{weight_dir}/generator_v4.pth')
    torch.save(discriminator.state_dict(), f'{weight_dir}/discriminator_v4.pth')

    with open(f'{weight_dir}/metrics_v4.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='GAN V4 LSGAN 训练')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--lambda_freq', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='./data/1_enhanced')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

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

    generator = GeneratorV4()
    discriminator = DiscriminatorV4()

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f'G 参数: {g_params/1e6:.1f}M | D 参数: {d_params/1e6:.1f}M | D/G: {d_params/g_params:.2f}')

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'g_lr': args.g_lr,
        'd_lr': args.d_lr,
        'noise_std': args.noise_std,
        'lambda_freq': args.lambda_freq,
        'warmup_epochs': args.warmup_epochs,
        'diversity_interval': 10,
        'label_smooth_real': 0.9,
        'label_smooth_fake': 0.1,
        'sample_dir': './sample/v4',
        'weight_dir': './weights_v4',
    }

    train_v4(generator, discriminator, dataloader, config)

    print('\n训练完成! 评估命令:')
    print('  python compare_results_v4.py')


if __name__ == '__main__':
    main()
