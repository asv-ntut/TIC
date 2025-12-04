import os
import numpy as np
import rasterio
from pathlib import Path

# ==========================================================
# ---                  使用者設定區                      ---
# ==========================================================

# 1. 來源檔案：您想轉換的【單一張】16位元 TIF 檔案路徑
SOURCE_FILE_PATH = r'D:\s2_cloudfree\ROIs1158_spring_1_p496.tif' # <--- 請修改這裡

# 2. 輸出檔案：您想儲存的 8位元 TIF 檔案路徑
OUTPUT_FILE_PATH = r'D:\s2_cloudfree\ROIs1158_spring_1_p496_visual_8bit.tif'  # <--- 請修改這裡

# ==========================================================
# ---                  核心邏輯 (無需修改)               ---
# ==========================================================

# --- 1. 定義要選取的 RGB 波段與全局統計數據 ---
# (索引從 1 開始，對應 Sentinel-2 的 B4, B3, B2)
# 如果您的波段順序不是 B4,B3,B2，請修改這裡
RGB_BANDS = [4, 3, 2]

# 全局統計數據 (來自您的原始碼)
S2L1C_MEAN_FULL = [2607.345, 2393.068, 2320.225, 2373.963, 2562.536, 3110.071, 3392.832, 3321.154, 3583.77, 1838.712,
                   1021.753, 3205.112, 2545.798]
S2L1C_STD_FULL = [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626,
                  1340.347, 1145.036]

# 根據 RGB_BANDS 的順序提取對應的 Mean 和 Std
# 注意：Python 索引從 0 開始，所以 Band 4 是索引 3，依此類推。
band_indices = [b - 1 for b in RGB_BANDS]  # 將 1-based 轉為 0-based
DATASET_MEANS = np.array([S2L1C_MEAN_FULL[i] for i in band_indices])
DATASET_STDS = np.array([S2L1C_STD_FULL[i] for i in band_indices])


# --- 2. 標準化函式 (與您提供的一致) ---
def standardize_for_visual(image_data, means, stds):
    """
    使用【資料集全局】的 mean 和 std 進行標準化，並轉換成適合觀看的 8-bit 數據。
    image_data 的維度應為 (C, H, W)，也就是 (3, height, width)
    """
    # 1. 確保維度正確以便進行廣播 (broadcasting)
    means = means[:, np.newaxis, np.newaxis]
    stds = stds[:, np.newaxis, np.newaxis]

    # 2. 進行標準化 (Z-score)
    standardized_data = (image_data.astype(np.float32) - means) / stds

    # 3. 裁切 (Clip) 到 -2.5 到 +2.5 個標準差
    clipped_data = np.clip(standardized_data, -2.5, 2.5)

    # 4. 將 [-2.5, 2.5] 的範圍線性縮放到 [0, 1]
    scaled_data = (clipped_data + 2.5) / 5.0

    # 5. 轉換成 8位元 (0-255)
    img_uint8 = (scaled_data * 255).astype(np.uint8)

    return img_uint8


# ==========================================================
# ---                  主程式執行區                      ---
# ==========================================================
if __name__ == '__main__':
    print("--- 單張 TIF 影像標準化轉換 (16-bit -> 8-bit) ---")

    # 檢查來源檔案是否存在
    if not os.path.exists(SOURCE_FILE_PATH):
        print(f"\n錯誤：找不到來源檔案！\n請檢查路徑是否正確: {SOURCE_FILE_PATH}")
    else:
        try:
            print(f"\n讀取來源檔案:\n  {SOURCE_FILE_PATH}")
            print(f"將使用的 Mean: {DATASET_MEANS}")
            print(f"將使用的 Std:  {DATASET_STDS}")

            # 確保輸出資料夾存在
            output_dir = Path(OUTPUT_FILE_PATH).parent
            os.makedirs(output_dir, exist_ok=True)

            # --- 核心處理流程 ---
            # 1. 讀取 16-bit 原始 TIF
            with rasterio.open(SOURCE_FILE_PATH) as src:
                rgb_data_uint16 = src.read(RGB_BANDS)
                profile = src.profile

            # 2. 呼叫標準化函式
            img_uint8 = standardize_for_visual(rgb_data_uint16, DATASET_MEANS, DATASET_STDS)

            # 3. 更新 TIF 檔案資訊以符合 8-bit RGB 輸出
            profile.update({
                'dtype': 'uint8',
                'count': 3,
                'photometric': 'RGB'
            })

            # 4. 寫入新的 8-bit TIF 檔案
            with rasterio.open(OUTPUT_FILE_PATH, 'w', **profile) as dst:
                dst.write(img_uint8)

            print(f"\n✅ 成功轉換影像並儲存至:\n  {OUTPUT_FILE_PATH}")

        except Exception as e:
            print(f"\n處理過程中發生錯誤: {e}")