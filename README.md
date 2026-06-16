# 🎨 Anime-DCGAN-Core (v3.5 Final Portfolio)

> **A cutting-edge, academic-grade PyTorch implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for 64x64 image synthesis.** This repository serves as an algorithmic sandbox documenting the complete research journey from catastrophic model collapse to a robust Nash equilibrium, integrating multi-scale attention, frequency constraints, and comprehensive quantitative metrics.

## 🔬 Algorithmic Evolution & Failure Analysis
Unlike typical "black-box" repositories, this project explicitly archives the empirical failures and theoretical bottlenecks discovered during the optimization cycle, demonstrating rigorous research diagnostics:

| Phase | Core Mechanism | Optimization Strategy | Empirical Result & Bottleneck Analysis |
| :--- | :--- | :--- | :--- |
| **v1.0** | Vanilla DCGAN | Standard BCE Loss + Naive CNN | **Failure:** Suffered severe path pollution and device hardcoding. Linear networks led to blurry, unaligned structures. |
| **v2.0** | Hinge GAN (`train_v2.py`) | Hinge Adversarial Loss + SpectralNorm (D) + `d_steps=2` | **Catastrophic Collapse:** Disproportionate Discriminator-to-Generator capacity (~6.3M vs ~3.6M). D collapsed as G failed to provide informative gradients under sharp Hinge boundaries. |
| **v3.0** | WGAN-GP (`train_v3.py`) | Wasserstein Distance + Gradient Penalty + `n_critic=5` | **Generator Vanished:** Critic gradient penalty successfully restricted Lipschitz continuity but generated a hyper-powerful Critic (11M params) that completely crushed the Generator. |
| **v4.0** | Balanced LSGAN (`train_v4.py`) | Least Squares GAN (MSE) + 1:1 Cadence + SpectralNorm + CBAM | **SUCCESS (Healthy Equilibrium):** Smooth, non-saturating gradients from LSGAN combined with CBAM attention successfully balanced the game, passing all diversity audits. |

## 🌟 Key Engineering Highlights (v4.0 Production Core)
- **Multi-Scale Attention Mechanics (`modules.py`):** Integrates **CBAM** for dynamic channel/spatial recalibration and **SAGAN-style Self-Attention** for long-range global grid dependencies.
- **Differentiable Frequency & Edge Constraints (`modules.py`):** Deploys a **FrequencyProcessor via 2D Fast Fourier Transform (FFT)** for texture preservation alongside a **Learnable Edge Detector (Sobel/Laplacian initialization)** to eliminate checkerboard artifacts.
- **Automated Audit Pipeline (`compare_results_v3.py`):** Establishes a closed-loop quantitative verification suite analyzing Mode Collapse via `cell_means_std` and benchmarking black-to-white ratios against Teacher Distributions.

## 📁 Repository Structure
```text
Anime-DCGAN-Core/
├── modules.py            # ADVANCED: CBAM, Self-Attention, Learnable Edge Filters, and 2D FFT
├── model_v4.py           # PRODUCTION: LSGAN architecture integrated with SpectralNorm & Attention
├── train_v4.py           # PIPELINE: Balance-optimized v4.0 active training pipeline script
├── train_v3.py           # RESEARCH LOG: WGAN-GP exploratory pipeline (Discriminator overpowered)
├── model_v3.py           # RESEARCH LOG: Pure Conv WGAN Critic structure without normalization
├── train_v2.py           # RESEARCH LOG: Hinge Loss exploratory pipeline (Catastrophic collapse)
├── model_v2.py           # RESEARCH LOG: SpectralNorm + Hinge standard baseline model
├── compare_results_v3.py # AUDIT SUITE: Upgraded evaluation for model v3/v4 telemetry metrics
├── compare_results_v2.py # AUDIT SUITE: Baseline evaluation tracking gray-levels and contrast ratios
├── data_clean.py         # PREPROCESSING: Offline bilateral noise-filtering & USM sharpening pipeline
├── create_dataset.py     # CUSTOM DATA: Core data loaders and denormalized image saving utilities
├── main.py               # LEGACY BASELINE: v2.0 baseline framework with computational graph truncation
├── model64.py            # LEGACY BASELINE: v2.0 basic 64x64 DCGAN network structure
├── model512.py           # LEGACY BASELINE: v2.0 high-resolution upscaled convolutional blocks
├── .gitignore            # SECURITY: Strict rules isolating raw dataset assets from upstream remotes
└── README.md             # MASTER PORTFOLIO: Comprehensive technical documentation
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install torch torchvision numpy opencv-python matplotlib pillow
```

### 2. Deploy Balanced Production Training (v4.0)
```bash
python train_v4.py --epochs 500 --batch_size 32 --g_lr 1e-4 --d_lr 1e-4
```

### 3. Execute Automated Quality Audit
```bash
python compare_results_v3.py --model v4
```

## 📊 Evaluation Report Metrics (Quantitative Audit)
The pipeline systematically prints a structural report to prevent human bias in visual assessment:
- **Mode Collapse Check:** `cell_means_std > 15` denotes healthy structural diversity.
- **Contrast Check:** `mid_ratio < 45%` ensures high-contrast sharp generation over uniform blur.
- **Teacher Alignment:** Cross-compares generated black-ratio standard deviation against `teacher_binary.png` properties to audit global macro-structural alignment.

## 🔮 Future Roadmap
- [ ] Integrate specific Tmall Campus retail-themed datasets for domain-specific data augmentation.
- [ ] Implement Wasserstein GAN with Gradient Penalty (WGAN-GP) constraints on top of the current frequency-domain processor.
```
