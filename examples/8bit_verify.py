import rasterio
import numpy as np
import os
from pathlib import Path

# --- 設定 ---
# 剛轉好的 8-bit 檔案路徑 (挑一張出來測)
TARGET_DIR = "/home/asvserver/TIC/s2_combined/train"

def check_image(path):
    print(f"Checking: {path.name}")
    try:
        with rasterio.open(path) as src:
            data = src.read()
            profile = src.profile
            
            # 1. 檢查 Metadata
            print(f"  - Metadata: {profile['dtype']}, Channels: {profile['count']}")
            if profile['dtype'] != 'uint8':
                print("  [FAIL] Dtype Error! Should be uint8.")
                return
            if profile['count'] != 3:
                print("  [FAIL] Channel Error! Should be 3.")
                return

            # 2. 檢查數值分佈
            print(f"  - Stats: Min={data.min()}, Max={data.max()}, Mean={data.mean():.2f}")
            
            # 3. 邏輯檢查
            if data.max() > 255 or data.min() < 0:
                print("  [FAIL] Value Range Error! Outside 0-255.")
            elif data.max() == 0:
                print("  [WARNING] Image is completely black (all zeros).")
            else:
                print("  [PASS] Data looks valid for 8-bit image.")

    except Exception as e:
        print(f"  [ERROR] {e}")

def main():
    p = Path(TARGET_DIR)
    files = list(p.glob("*.tif"))
    if not files:
        print("No files found!")
        return
    
    # 隨機抽 3 張檢查
    import random
    samples = random.sample(files, min(len(files), 3))
    
    for f in samples:
        check_image(f)
        print("-" * 30)

if __name__ == "__main__":
    main()