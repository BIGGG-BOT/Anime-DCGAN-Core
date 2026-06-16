"""
模型质量评估脚本 (支持 V3/V4)
==============================
评估维度:
  1. 模式坍塌: 像素均值标准差 (cell_means_std > 15 健康)
  2. 对比度: 中间灰度占比 (mid_ratio < 50% 健康)
  3. 多样性: 相邻图片差异 (neighbor_diff > 5.0 健康)
  4. 边缘密度 / 高频能量

使用方法:
  python compare_results_v3.py --model v3
  python compare_results_v3.py --model v4
  python compare_results_v3.py --model v4 --gen_path ./weights_v4/generator_v4_epoch300.pth
"""

import torch
import numpy as np
from PIL import Image
import os, sys, json, argparse

from create_dataset import save_img
from modules import compute_edge_density, compute_high_freq_energy


def load_generator(model_type, weight_path, device):
    """加载不同版本的生成器"""
    if model_type == 'v3':
        from model_v3 import GeneratorV3
        gen = GeneratorV3().to(device)
    elif model_type == 'v4':
        from model_v4 import GeneratorV4
        gen = GeneratorV4().to(device)
    elif model_type == 'v5':
        from model_v5 import GeneratorV5
        gen = GeneratorV5().to(device)
    elif model_type == 'final':
        from model_final import Generator
        gen = Generator().to(device)
    elif model_type == 'v6':
        from model_v6 import GeneratorV6
        gen = GeneratorV6().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}, use v3/v4/v5/final/v6")

    state = torch.load(weight_path, map_location=device)
    gen.load_state_dict(state)
    gen.eval()
    return gen


def analyze_generated_samples(gen, device, num_samples=64):
    """生成样本并分析质量"""
    with torch.no_grad():
        noise = torch.randn(num_samples, 100, 1, 1, device=device)
        fake = gen(noise)

    os.makedirs('./sample/eval', exist_ok=True)
    save_img(fake, './sample/eval/eval_grid.png')

    fake_np = (fake * 0.5 + 0.5).clamp(0, 1)
    cell_means = []
    dark_ratios = []
    light_ratios = []

    for i in range(num_samples):
        gray = fake_np[i].permute(1, 2, 0).cpu().numpy().mean(axis=2)
        cell_means.append(gray.mean())
        dark_ratios.append((gray < 0.31).mean())
        light_ratios.append((gray > 0.67).mean())

    cell_means = np.array(cell_means)
    mid_ratios = 1.0 - np.mean(dark_ratios) - np.mean(light_ratios)

    neighbor_diffs = []
    for i in range(0, num_samples - 1, 2):
        diff = torch.abs(fake[i] - fake[i + 1]).mean().item()
        neighbor_diffs.append(diff)

    # 逐图差异矩阵 (衡量多样性)
    flat_fake = fake.view(num_samples, -1)
    pairwise_dists = []
    for i in range(min(num_samples, 16)):
        for j in range(i + 1, min(num_samples, 16)):
            d = torch.norm(flat_fake[i] - flat_fake[j], p=2).item()
            pairwise_dists.append(d)

    return {
        'cell_means_mean': float(cell_means.mean()),
        'cell_means_std': float(cell_means.std()),
        'cell_means_min': float(cell_means.min()),
        'cell_means_max': float(cell_means.max()),
        'mid_ratio_mean': float(mid_ratios),
        'dark_ratio_mean': float(np.mean(dark_ratios)),
        'light_ratio_mean': float(np.mean(light_ratios)),
        'neighbor_diff_mean': float(np.mean(neighbor_diffs)),
        'pairwise_dist_mean': float(np.mean(pairwise_dists)),
        'pairwise_dist_std': float(np.std(pairwise_dists)),
        'edge_density': compute_edge_density(fake),
        'hf_energy': compute_high_freq_energy(fake),
    }


def compare_with_teacher(fake_tensor):
    """与 teacher 图片对比"""
    teacher_paths = ['./teacher_binary.png', './teacher_requirement.png',
                     './mmexport1780646140696.jpg']
    teacher_bin = None
    for p in teacher_paths:
        if os.path.exists(p):
            img = Image.open(p)
            teacher_bin = np.array(img)
            break

    if teacher_bin is None:
        return {}

    if teacher_bin.ndim == 3:
        teacher_bin = teacher_bin.mean(axis=2)

    h, w = teacher_bin.shape
    side = min(530, h, w)
    left_bin = teacher_bin[:side, :side]

    teacher_cell_stats = []
    for row in range(8):
        for col in range(8):
            r_start = row * 66 + 1
            c_start = col * 66 + 1
            if r_start + 64 <= side and c_start + 64 <= side:
                cell = left_bin[r_start:r_start+64, c_start:c_start+64]
                teacher_cell_stats.append((cell < 128).mean())

    fake_np = (fake_tensor.detach() * 0.5 + 0.5).clamp(0, 1)
    fake_gray = fake_np.mean(dim=1).cpu().numpy()
    gen_black_ratios = []
    for i in range(min(64, fake_gray.shape[0])):
        img = fake_gray[i]
        threshold = np.median(img)
        binary = (img < threshold).astype(float)
        gen_black_ratios.append(binary.mean())

    if teacher_cell_stats:
        return {
            'teacher_black_ratio_mean': float(np.mean(teacher_cell_stats)),
            'teacher_black_ratio_std': float(np.std(teacher_cell_stats)),
            'gen_black_ratio_mean': float(np.mean(gen_black_ratios)),
            'gen_black_ratio_std': float(np.std(gen_black_ratios)),
        }
    return {}


def print_report(results, teacher_comp, model_type):
    """打印评估报告"""
    print(f"\n{'='*60}")
    print(f"Model Quality Report ({model_type.upper()})")
    print(f"{'='*60}")

    # === 模式坍塌 ===
    std = results['cell_means_std']
    if std > 15:
        status = 'HEALTHY'
    elif std > 5:
        status = 'WARNING'
    else:
        status = 'FAILED'
    print(f"\n[1] Mode Collapse Check")
    print(f"    cell_means_std:   {std:.2f}  ({status})")
    print(f"    healthy > 15  |  warning 5-15  |  failed < 5")
    print(f"    per-image mean range: [{results['cell_means_min']:.3f}, {results['cell_means_max']:.3f}]")
    print(f"    neighbor_diff:  {results['neighbor_diff_mean']:.3f}  (healthy > 5.0)")
    print(f"    pairwise_dist:  {results['pairwise_dist_mean']:.1f} +/- {results['pairwise_dist_std']:.1f}")

    # === 对比度 ===
    mid = results['mid_ratio_mean']
    if mid < 0.45:
        sc = 'Good'
    elif mid < 0.55:
        sc = 'OK'
    else:
        sc = 'Blurry'
    print(f"\n[2] Contrast Check")
    print(f"    mid_ratio:    {mid*100:.1f}%  ({sc})")
    print(f"    dark_ratio:   {results['dark_ratio_mean']*100:.1f}%")
    print(f"    light_ratio:  {results['light_ratio_mean']*100:.1f}%")
    print(f"    target: mid_ratio < 45%")

    # === 结构 ===
    print(f"\n[3] Structure")
    print(f"    edge_density:  {results['edge_density']:.4f}")
    print(f"    hf_energy:     {results['hf_energy']:.4f}")

    # === Teacher ===
    if teacher_comp:
        print(f"\n[4] Teacher Comparison")
        print(f"    Teacher black_ratio: {teacher_comp['teacher_black_ratio_mean']:.4f}"
              f" +/- {teacher_comp['teacher_black_ratio_std']:.4f}")
        print(f"    Generated black_ratio: {teacher_comp['gen_black_ratio_mean']:.4f}"
              f" +/- {teacher_comp['gen_black_ratio_std']:.4f}")
        gbs = teacher_comp['gen_black_ratio_std']
        if gbs < 0.01:
            print(f"    WARNING: gen_black_ratio_std = {gbs:.6f} (all images identical!)")

    # === 综合 ===
    checks = [
        ('Diversity  (cell_means_std > 10)', std > 10),
        ('Contrast   (mid_ratio < 50%)', mid < 0.50),
        ('Variation  (neighbor_diff > 3.0)', results['neighbor_diff_mean'] > 3.0),
    ]
    passed = sum(c[1] for c in checks)
    print(f"\n[5] Summary: {passed}/{len(checks)} checks passed")
    for name, ok in checks:
        print(f"    {'[PASS]' if ok else '[FAIL]'} {name}")


def main():
    parser = argparse.ArgumentParser(description='Model Quality Evaluation')
    parser.add_argument('--model', type=str, default='final',
                        help='Model version: v3/v4/v5/final')
    parser.add_argument('--gen_path', type=str, default=None,
                        help='Generator weights path')
    parser.add_argument('--num_samples', type=int, default=64)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 自动推断权重路径
    if args.gen_path is None:
        weight_dir = f'./weights_{args.model}'
        if args.model == 'final':
            gen_path = os.path.join(weight_dir, 'generator.pth')
        else:
            gen_path = os.path.join(weight_dir, f'generator_{args.model}.pth')
    else:
        gen_path = args.gen_path

    if not os.path.exists(gen_path):
        print(f'[ERR] Weight file not found: {gen_path}')
        print(f'      Train first or specify --gen_path')
        sys.exit(1)

    gen = load_generator(args.model, gen_path, device)
    print(f'[OK] Loaded: {gen_path}')

    results = analyze_generated_samples(gen, device, args.num_samples)
    with torch.no_grad():
        teacher_fake = gen(torch.randn(args.num_samples, 100, 1, 1, device=device))
    teacher_comp = compare_with_teacher(teacher_fake)

    os.makedirs('./sample/eval', exist_ok=True)
    with open('./sample/eval/metrics.json', 'w') as f:
        json.dump({**results, **teacher_comp}, f, indent=2)

    print_report(results, teacher_comp, args.model)
    print(f"\nEval grid: ./sample/eval/eval_grid.png")
    print(f"Metrics:   ./sample/eval/metrics.json")


if __name__ == '__main__':
    main()
