"""
V2 模型质量评估脚本
====================
评估维度:
  1. 模式坍塌: 生成图像像素均值标准差 (cell_means std > 15 为健康)
  2. 对比度: 中间灰度像素占比 (mid_ratio < 50% 为健康)
  3. 多样性: 相邻图片差异
  4. 与 teacher_binary 的结构相似度

使用方法:
  python compare_results_v2.py
  python compare_results_v2.py --gen_path ./weights_v2/generator_v2.pth
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
import json
import argparse

from model_v2 import GeneratorV2
from create_dataset import save_img
from modules import compute_edge_density, compute_high_freq_energy


def analyze_generated_samples(gen, device, num_samples=64):
    """分析生成样本的质量"""
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        fake = gen(noise)

    # 保存网格图
    os.makedirs('./sample/v2_eval', exist_ok=True)
    save_img(fake, './sample/v2_eval/eval_grid.png')

    # 提取每张图的统计量
    # fake: [64, 3, 64, 64]
    fake_np = (fake * 0.5 + 0.5).clamp(0, 1)  # denormalize
    cell_means = []
    mid_ratios = []
    dark_ratios = []
    light_ratios = []

    for i in range(num_samples):
        img = fake_np[i].permute(1, 2, 0).cpu().numpy()  # [64, 64, 3]
        gray = img.mean(axis=2)
        cell_means.append(gray.mean())
        dark_ratios.append((gray < 0.31).mean())   # < 80/255
        light_ratios.append((gray > 0.67).mean())   # > 170/255
        mid_ratios.append(1 - dark_ratios[-1] - light_ratios[-1])

    cell_means = np.array(cell_means)
    mid_ratios = np.array(mid_ratios)

    # 相邻图片差异
    neighbor_diffs = []
    for i in range(0, num_samples - 1, 2):
        diff = torch.abs(fake[i] - fake[i + 1]).mean().item()
        neighbor_diffs.append(diff)

    results = {
        'cell_means_mean': float(cell_means.mean()),
        'cell_means_std': float(cell_means.std()),
        'cell_means_min': float(cell_means.min()),
        'cell_means_max': float(cell_means.max()),
        'mid_ratio_mean': float(mid_ratios.mean()),
        'dark_ratio_mean': float(np.mean(dark_ratios)),
        'light_ratio_mean': float(np.mean(light_ratios)),
        'neighbor_diff_mean': float(np.mean(neighbor_diffs)),
        'edge_density': compute_edge_density(fake),
        'hf_energy': compute_high_freq_energy(fake),
    }

    return results, fake_np


def compare_with_teacher(fake_tensor, device):
    """与 teacher_binary 对比"""
    try:
        teacher_bin = np.array(Image.open('./teacher_binary.png'))
    except:
        print('[WARN] teacher_binary.png 未找到，跳过对比')
        return {}

    # teacher_binary 是 1080×580，左侧 530×530 是8×8网格
    left_bin = teacher_bin[:530, :530]

    teacher_cell_stats = []
    for row in range(8):
        for col in range(8):
            r_start = row * 66 + 1
            c_start = col * 66 + 1
            if r_start + 64 <= 530 and c_start + 64 <= 530:
                cell = left_bin[r_start:r_start+64, c_start:c_start+64]
                teacher_cell_stats.append((cell == 0).mean())  # black ratio

    # 生成图像的二值化版本（以中位数阈值）
    fake_np = (fake_tensor * 0.5 + 0.5).clamp(0, 1)
    fake_gray = fake_np.mean(dim=1).cpu().numpy()  # [64, 64, 64]
    gen_black_ratios = []
    for i in range(min(64, fake_gray.shape[0])):
        img = fake_gray[i]
        threshold = np.median(img)
        binary = (img < threshold).astype(float)
        gen_black_ratios.append(binary.mean())

    return {
        'teacher_black_ratio_mean': float(np.mean(teacher_cell_stats)),
        'teacher_black_ratio_std': float(np.std(teacher_cell_stats)),
        'gen_black_ratio_mean': float(np.mean(gen_black_ratios)),
        'gen_black_ratio_std': float(np.std(gen_black_ratios)),
    }


def print_report(results, teacher_comp):
    """打印质量报告"""
    print(f"\n{'='*60}")
    print("V2 模型质量评估报告")
    print(f"{'='*60}")

    # 模式坍塌评估
    std = results['cell_means_std']
    status = '✅ 健康' if std > 15 else ('⚠️ 轻微坍塌' if std > 5 else '❌ 严重坍塌')
    print(f"\n【模式坍塌评估】")
    print(f"  64图像素均值 std: {std:.2f} ({status})")
    print(f"  均值范围: [{results['cell_means_min']:.3f}, {results['cell_means_max']:.3f}]")
    print(f"  相邻图差异: {results['neighbor_diff_mean']:.3f}")

    # 对比度评估
    mid = results['mid_ratio_mean']
    status_c = '✅ 良好' if mid < 0.45 else ('⚠️ 一般' if mid < 0.55 else '❌ 模糊')
    print(f"\n【对比度评估】")
    print(f"  中间灰度占比: {mid*100:.1f}% ({status_c})")
    print(f"  暗区占比: {results['dark_ratio_mean']*100:.1f}%")
    print(f"  亮区占比: {results['light_ratio_mean']*100:.1f}%")

    # 结构指标
    print(f"\n【结构指标】")
    print(f"  边缘密度: {results['edge_density']:.4f}")
    print(f"  高频能量: {results['hf_energy']:.4f}")

    # Teacher 对比
    if teacher_comp:
        print(f"\n【Teacher Binary 对比】")
        print(f"  Teacher black ratio: {teacher_comp['teacher_black_ratio_mean']:.4f} "
              f"± {teacher_comp['teacher_black_ratio_std']:.4f}")
        print(f"  Generated black ratio: {teacher_comp['gen_black_ratio_mean']:.4f} "
              f"± {teacher_comp['gen_black_ratio_std']:.4f}")

    # 综合判定
    print(f"\n【综合判定】")
    checks = []
    checks.append(std > 10)
    checks.append(mid < 0.50)
    checks.append(results['neighbor_diff_mean'] > 5.0)
    passed = sum(checks)
    if passed == 3:
        print("  🎉 所有指标达标！生成质量良好。")
    elif passed >= 2:
        print(f"  ⚠️ {passed}/3 指标达标，建议继续训练或微调超参。")
    else:
        print(f"  ❌ 仅 {passed}/3 指标达标，需要检查训练状态。")


def main():
    parser = argparse.ArgumentParser(description='V2 模型质量评估')
    parser.add_argument('--gen_path', type=str,
                        default='./weights_v2/generator_v2.pth',
                        help='生成器权重路径')
    parser.add_argument('--num_samples', type=int, default=64)
    args = parser.parse_args()

    # 切换到脚本所在目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    # 加载模型
    if not os.path.exists(args.gen_path):
        print(f'[ERR] 权重文件不存在: {args.gen_path}')
        print('请先训练模型: python train_v2.py')
        sys.exit(1)

    gen = GeneratorV2().to(device)
    gen.load_state_dict(torch.load(args.gen_path, map_location=device))
    gen.eval()
    print(f'[OK] 已加载: {args.gen_path}')

    # 分析
    results, fake_np = analyze_generated_samples(gen, device, args.num_samples)
    teacher_comp = compare_with_teacher(fake_np, device)

    # 保存结果
    with open('./sample/v2_eval/metrics.json', 'w') as f:
        json.dump({**results, **teacher_comp}, f, indent=2)

    print_report(results, teacher_comp)
    print(f"\n评估图已保存: ./sample/v2_eval/eval_grid.png")
    print(f"指标已保存: ./sample/v2_eval/metrics.json")


if __name__ == '__main__':
    main()
