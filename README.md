# 🎨 Anime-DCGAN-Core

> A clean, PyTorch-based implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for 64x64 image synthesis, serving as a robust baseline framework for future Tmall campus data augmentation.

## ✨ Key Features
- **Hardware-Agnostic:** Automatically detects and utilizes GPU (CUDA) if available, gracefully falling back to CPU. No more hardcoded device crashes.
- **Modular Architecture:** The Generator and Discriminator networks are clearly decoupled (`model64.py` / `model512.py`) for easy scaling and modification.
- **Clean I/O Management:** Adopts relative pathing and strictly ignores local datasets via `.gitignore` to prevent repository bloat and path pollution.

## 📁 Repository Structure
```text
Anime-DCGAN-Core/
├── create_dataset.py   # Custom PyTorch Dataset loader and image processing
├── main.py             # Main training loop and optimization logic
├── model64.py          # DCGAN architecture optimized for 64x64 generation
├── model512.py         # Advanced architecture scaling up to 512x512
├── .gitignore          # Git ignore rules (protecting raw data)
└── README.md           # Project documentation
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install torch torchvision numpy pillow
```

### 2. Data Preparation
For privacy and repository size limits, the training dataset is not included. Please create a `data/1/` directory in the root folder and place your training images inside.

### 3. Run Training
```bash
python main.py
```
*Generated samples will be automatically saved in the `./sample/` directory during training.*

## 🔮 Future Work
- [ ] Integrate specific Tmall Campus datasets for specialized data augmentation.
- [ ] Implement advanced loss functions (e.g., Wasserstein Loss) to stabilize training.
