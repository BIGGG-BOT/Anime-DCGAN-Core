import os
import cv2
import numpy as np

INPUT_DIR = './data/1'
OUTPUT_DIR = './data/1_enhanced'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def enhance_image(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    # 噪声过滤：双边滤波 (Bilateral Filter)
    filtered = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    # 边缘锐化：USM 锐化 (Unsharp Masking)
    blurred = cv2.GaussianBlur(filtered, (0, 0), 2.0)
    sharpened = cv2.addWeighted(filtered, 1.5, blurred, -0.5, 0)
    cv2.imwrite(save_path, sharpened)

def main():
    valid_exts = ('.jpg', '.png', '.jpeg')
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到原始数据文件夹 {INPUT_DIR}，请先将数据放入该位置！")
        return
        
    img_names = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]
    total = len(img_names)
    print(f"🚀 开始执行离线数据清洗与特征强化，共计 {total} 张样本...")
    
    for i, img_name in enumerate(img_names):
        in_path = os.path.join(INPUT_DIR, img_name)
        out_path = os.path.join(OUTPUT_DIR, img_name)
        enhance_image(in_path, out_path)
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"[{i + 1}/{total}] 数据强化处理完成.")
            
    print(f"✅ 所有特征强化样本已落盘至: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()