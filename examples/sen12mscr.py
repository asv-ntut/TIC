import os
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
import glob

# --- 1. 來自你提供程式碼的關鍵設定 ---

# 選擇 RGB 通道 (0-indexed)
# Sentinel-2: [3, 2, 1] -> B4 (Red), B3 (Green), B2 (Blue)
RGB_BANDS = [3, 2, 1]

# 像素值的裁剪與縮放範圍
CLIP_MIN = 0
CLIP_MAX = 10000
SCALE = 10000.0

# --- 2. 設定輸入與輸出路徑 ---

# [!!] 請修改: 你的原始 TIF 影像根目錄 (包含 test, train, val)
INPUT_ROOT_DIR = r"D:\s2_cloudfree_balanced_split"

# [!!] 請修改: 你要存放 8-bit RGB 影像的輸出根目錄
OUTPUT_ROOT_DIR = r"D:\s2_cloudfree_8bit_RGB_TIF_noGeo"  # 我改了資料夾名稱以示區別


# --- 3. 核心處理函式 ---

def process_image(tif_path, output_path):
    """
    讀取 TIF 影像，套用正規化邏輯，並儲存為 8-bit RGB TIF 影像 (不含地理資訊)
    """
    try:
        with rasterio.open(tif_path) as src:
            # 讀取所有通道
            raw_data = src.read().astype(np.float32)

            # 參考原程式碼: 處理 NaN 值 (用該通道的平均值取代)
            if np.isnan(raw_data).any():
                for i in range(raw_data.shape[0]):
                    band = raw_data[i]
                    if np.isnan(band).any():
                        band_mean = np.nanmean(band)
                        band[np.isnan(band)] = band_mean
                        raw_data[i] = band

            # 1. 選取 RGB 通道
            # (C, H, W) -> (3, H, W)
            rgb_data = raw_data[RGB_BANDS, :, :]

            # 2. 裁剪 (Clip)
            clipped_data = np.clip(rgb_data, CLIP_MIN, CLIP_MAX)

            # 3. 縮放 (Scale) 至 [0.0, 1.0]
            normalized_data = clipped_data / SCALE

            # 4. 轉換為 8-bit [0, 255]
            image_8bit = (normalized_data * 255).astype(np.uint8)

            # 5. 轉換維度 (C, H, W) -> (H, W, C) 以便 PIL 儲存
            image_8bit_hwc = image_8bit.transpose(1, 2, 0)

            # 6. 儲存影像
            img = Image.fromarray(image_8bit_hwc)

            # [!!] 修改點 1: 將 "PNG" 改為 "TIFF"
            img.save(output_path, "TIFF")

            return True

    except Exception as e:
        print(f"\n[錯誤] 處理 {tif_path} 失敗: {e}")
        return False


# --- 4. 主程式 ---

if __name__ == "__main__":
    print(f"開始處理 (輸出為 TIF, 不含地理資訊)...")
    print(f"輸入資料夾: {INPUT_ROOT_DIR}")
    print(f"輸出資料夾: {OUTPUT_ROOT_DIR}")

    # 使用 glob 遞迴搜尋所有 .tif 檔案
    # ** 代表搜尋所有子資料夾
    search_pattern = os.path.join(INPUT_ROOT_DIR, '**', '*.tif*')  # 支援 .tif 和 .tiff
    tif_files = glob.glob(search_pattern, recursive=True)

    if not tif_files:
        print(f"在 {INPUT_ROOT_DIR} 中找不到任何 .tif 檔案。")
        exit()

    print(f"共找到 {len(tif_files)} 個 TIF 檔案。")

    success_count = 0
    fail_count = 0

    # 使用 tqdm 顯示進度條
    for input_path in tqdm(tif_files, desc="Processing images"):

        # 建立對應的輸出路徑
        # 1. 取得相對於 INPUT_ROOT_DIR 的路徑 (例如 "test/ROIs1158_spring_s2_14.tif")
        relative_path = os.path.relpath(input_path, INPUT_ROOT_DIR)

        # 2. 建立輸出資料夾 (例如 "D:/s2_cloudfree_8bit_RGB_TIF_noGeo/test")
        output_dir = os.path.join(OUTPUT_ROOT_DIR, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)

        # 3. 更改副檔名為 .tif
        # [!!] 修改點 2: 將 ".png" 改為 ".tif"
        base_name = os.path.basename(input_path)
        output_name = os.path.splitext(base_name)[0] + ".tif"
        output_path = os.path.join(output_dir, output_name)

        # 4. 執行處理
        if process_image(input_path, output_path):
            success_count += 1
        else:
            fail_count += 1

    print("\n--- 處理完成 ---")
    print(f"成功: {success_count} 張")
    print(f"失敗: {fail_count} 張")
    print(f"8-bit RGB TIF 影像已儲存至: {OUTPUT_ROOT_DIR}")