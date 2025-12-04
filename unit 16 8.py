import os
import numpy as np
import rasterio
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ==========================================================
# ---                  使用者設定區                      ---
# ==========================================================

# 1. 來源資料夾：您用【平衡劃分腳本】產生的乾淨資料夾
SOURCE_BASE_DIR = r'D:\s2_cloudfree_balanced_split'

# 2. 輸出資料夾：一個全新的、乾淨的資料夾，用來存放最終的 8-bit 視覺化圖片
OUTPUT_BASE_DIR = r'D:\s2_cloudfree_balanced_split_visual_uint8'

# 3. 定義要選取的 RGB 波段 (紅, 綠, 藍)
RGB_BANDS = [4, 3, 2] # Sentinel-2 標準真彩色 (B4, B3, B2)

# 4. 多處理程序設定
num_processes = max(1, cpu_count() // 2)
maxtasksperchild = 1000

# ==========================================================
# ---             全局統計數據 (無需修改)                ---
# ==========================================================
S2L1C_MEAN_FULL = [2607.345, 2393.068, 2320.225, 2373.963, 2562.536, 3110.071, 3392.832, 3321.154, 3583.77, 1838.712, 1021.753, 3205.112, 2545.798]
S2L1C_STD_FULL  = [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036]

# 根據 RGB_BANDS = [4, 3, 2] 的順序提取對應的 Mean 和 Std
# DATASET_MEANS 的順序將是 [B4_mean, B3_mean, B2_mean]
DATASET_MEANS = np.array([S2L1C_MEAN_FULL[3], S2L1C_MEAN_FULL[2], S2L1C_MEAN_FULL[1]])
DATASET_STDS  = np.array([S2L1C_STD_FULL[3],  S2L1C_STD_FULL[2],  S2L1C_STD_FULL[1]])


# ==========================================================
# ---              核心轉換函式 (無需修改)               ---
# ==========================================================
def convert_to_visual_uint8(image_data, means, stds):
    """
    使用【資料集全局】的 mean 和 std 進行標準化，並轉換成適合觀看的 8-bit 數據。
    """
    means = means[:, np.newaxis, np.newaxis]
    stds = stds[:, np.newaxis, np.newaxis]

    # 1. 進行標準化 (Z-score)
    standardized_data = (image_data.astype(np.float32) - means) / stds

    # 2. 裁切 (Clip) 到 -2.5 到 +2.5 個標準差，以增強視覺對比度
    clipped_data = np.clip(standardized_data, -2.5, 2.5)

    # 3. 將 [-2.5, 2.5] 的範圍線性縮放到 [0, 1]
    scaled_data = (clipped_data + 2.5) / 5.0

    # 4. 轉換成 8位元 (0-255)
    img_uint8 = (scaled_data * 255).astype(np.uint8)

    return img_uint8


def process_single_file(src_path: Path):
    """處理單一檔案的全部邏輯。"""
    try:
        split_name = src_path.parent.name
        file_name = src_path.name
        output_dir = Path(OUTPUT_BASE_DIR) / split_name
        output_path = output_dir / file_name

        with rasterio.open(src_path) as src:
            rgb_data_uint16 = src.read(RGB_BANDS)
            profile = src.profile

        img_uint8 = convert_to_visual_uint8(rgb_data_uint16, DATASET_MEANS, DATASET_STDS)

        # 【修改點】更新 profile 以符合 8位元 RGB 輸出，並移除壓縮
        profile.update({
            'dtype': 'uint8',       # <--- 輸出為 8-bit
            'count': 3,
            'photometric': 'RGB',
            'compress': None,       # <--- 設置為 None 來禁用壓縮
            'predictor': 1          # 確保 predictor 為預設值
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(img_uint8)

        return None
    except Exception as e:
        return f"處理 {src_path} 時發生錯誤: {e}"


# ==========================================================
# ---                  主程式 (無需修改)                 ---
# ==========================================================
def main_process():
    """主執行函式"""
    print("--- 模式：轉換為【視覺化 TIF】(8位元，全局色彩一致) ---")
    print(f"來源資料夾: {SOURCE_BASE_DIR}")
    print(f"輸出資料夾: {OUTPUT_BASE_DIR}\n")
    print(f"使用的 Mean: {DATASET_MEANS}")
    print(f"使用的 Std:  {DATASET_STDS}\n")

    all_source_files = []
    splits = ['train', 'val', 'test']
    for split_name in splits:
        source_dir = Path(SOURCE_BASE_DIR) / split_name
        output_dir = Path(OUTPUT_BASE_DIR) / split_name
        os.makedirs(output_dir, exist_ok=True)

        files_in_split = list(source_dir.glob('*.tif'))
        if not files_in_split:
            print(f"警告：在 {source_dir} 中找不到 .tif 檔案，跳過。")
            continue
        all_source_files.extend(files_in_split)

    if not all_source_files:
        print("在所有資料夾中都找不到 .tif 檔案，程式結束。")
        return

    print(f"找到 {len(all_source_files)} 個檔案，將使用 {num_processes} 個 CPU 核心進行處理...")

    with Pool(processes=num_processes, maxtasksperchild=maxtasksperchild) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, all_source_files), total=len(all_source_files)))

    print("\n\n✅ 所有檔案已成功轉換並儲存為 8位元 RGB TIF！")

    errors = [r for r in results if r is not None]
    if errors:
        print(f"\n處理過程中發生了 {len(errors)} 個錯誤：")
        for err in errors[:10]:
            print(err)


if __name__ == '__main__':
    main_process()