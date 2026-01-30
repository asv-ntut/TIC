import os
import random
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 設定區 (請確認路徑正確)
# ==========================================
INPUT_DIR = "/home/asvserver/TIC/s2_combined/test"  # 來源 16-bit 資料夾
OUTPUT_DIR = "/home/asvserver/TIC/s2_combined/test_8bit"  # 輸出 8-bit 資料夾

TIF_EXTENSIONS = {'.tif', '.tiff'}

# --- 轉換參數 ---
RGB_BANDS = [3, 2, 1]  # Sentinel-2: B4(Red), B3(Green), B2(Blue)
CLIP_MIN = 0.0
CLIP_MAX = 10000.0
SCALE = 10000.0

# ==========================================
# 核心轉換函式
# ==========================================
def process_and_save_tif(src_path, dst_path):
    """
    讀取 16-bit Sentinel-2 TIF，轉為 8-bit RGB TIF 並存檔。
    保留原始的 CRS (座標系統) 與 Transform (地理位置)。
    """
    try:
        with rasterio.open(src_path) as src:
            # 1. 讀取並轉為 float32
            raw_data = src.read().astype(np.float32)
            
            # 獲取原始 profile 用於寫入，但需要修改部分參數
            out_profile = src.profile.copy()

        # 2. 處理 NaN (自動補值為該波段的平均值)
        if np.isnan(raw_data).any():
            for i in range(raw_data.shape[0]):
                band = raw_data[i]
                if np.isnan(band).any():
                    band_mean = np.nanmean(band)
                    band[np.isnan(band)] = band_mean
                    raw_data[i] = band

        # 3. 波段選擇 (RGB)
        if raw_data.shape[0] > max(RGB_BANDS):
            rgb_data = raw_data[RGB_BANDS, :, :]
        else:
            # Fallback 若波段不足
            rgb_data = raw_data[:3, :, :]

        # 4. 裁切與正規化 (0 ~ 10000 -> 0.0 ~ 1.0)
        clipped_data = np.clip(rgb_data, CLIP_MIN, CLIP_MAX)
        normalized_data = clipped_data / SCALE

        # 5. 轉換為 8-bit (0.0 ~ 1.0 -> 0 ~ 255)
        img_8bit = (normalized_data * 255).astype(np.uint8)

        # 6. 更新 Profile 以符合 8-bit 輸出
        out_profile.update({
            'dtype': 'uint8',       # 資料型態改為 8-bit
            'count': 3,             # 通道數固定為 3 (RGB)
            'driver': 'GTiff',      # 確保是 GeoTIFF
            'photometric': 'RGB'    # 設定色彩空間為 RGB
        })

        # 7. 寫入新的 TIF 檔案
        with rasterio.open(dst_path, 'w', **out_profile) as dst:
            dst.write(img_8bit)

        return True

    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

# ==========================================
# 主程式 (含斷點續傳機制)
# ==========================================
def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 掃描所有檔案
    all_files = [f for f in input_path.rglob("*") if f.suffix.lower() in TIF_EXTENSIONS]
    
    if not all_files:
        print(f"在 {INPUT_DIR} 找不到任何 TIF 檔案。")
        return

    # 2. 隨機抽樣 (目前設定是全部 100%)
    sample_size = int(len(all_files))
    if sample_size < 1: sample_size = 1 # 至少一張
    sampled_files = random.sample(all_files, sample_size)

    print(f"總檔案數: {len(all_files)}，將處理 {len(sampled_files)} 筆檔案...")
    print(f"輸出格式: 8-bit GeoTIFF (保留原始座標資訊)")

    # 3. 執行轉換 (加入斷點續傳)
    success_count = 0
    skip_count = 0  # 紀錄跳過了多少張
    
    for file_path in tqdm(sampled_files, desc="Converting 16-to-8 bit"):
        # 保持原始檔名，但存到新資料夾
        save_name = file_path.name 
        save_path = output_path / save_name
        
        # ========================================================
        # [斷點續傳防呆] 檢查檔案是否存在且大於 0 byte
        # ========================================================
        if save_path.exists() and save_path.stat().st_size > 0:
            success_count += 1
            skip_count += 1
            continue  # 檔案已存在，直接跳過不做處理
        
        # 執行轉換
        if process_and_save_tif(file_path, save_path):
            success_count += 1

    print(f"\n✅ 完成！成功轉換/略過 {success_count} 張影像至 {OUTPUT_DIR}")
    if skip_count > 0:
        print(f"  (本次執行極速略過了 {skip_count} 張已經轉好的圖片)")

if __name__ == "__main__":
    # 確保 rasterio 讀寫安全
    os.environ["GDAL_PAM_ENABLED"] = "NO"
    main()