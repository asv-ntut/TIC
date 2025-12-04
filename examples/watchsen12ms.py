import rasterio
import numpy as np
import os

# --- 1. 設定你的 TIF 檔案路徑 ---
# (請確保路徑正確)
file_path = r"C:\Users\Matt\Downloads\ROIs1158_spring_1_p30.tif"

# --- 2. 設定要儲存的新 TIF 檔名 ---
output_file = "rgb_image_16bit.tif"

# --- 3. 定義 Sentinel-2 的 RGB 波段 (1-based index) ---
# B4 (Red) = 索引 4
# B3 (Green) = 索引 3
# B2 (Blue) = 索引 2
RGB_BANDS_TO_READ = [4, 3, 2]

try:
    with rasterio.open(file_path) as src:

        # --- 4. 讀取 TIF 檔案的 "元資料" (metadata) ---
        # 這是關於影像的所有設定資訊 (如座標、尺寸等)
        profile = src.profile

        # --- 5. 只讀取 RGB 三個波段 ---
        # 讀取出來的資料 *仍然是 uint16* (16-bit)
        # 形狀是 (3, H, W)
        rgb_image_16bit = src.read(RGB_BANDS_TO_READ)

    print(f"成功讀取檔案，影像維度: {rgb_image_16bit.shape}")
    print(f"原始資料型態: {rgb_image_16bit.dtype}")

    # --- 6. 更新 "元資料" 以符合新的 3 通道影像 ---
    # 告訴 rasterio 我們的新檔案只有 3 個通道
    profile.update({
        'count': 3,
        'dtype': 'uint16',  # 確保儲存時也是 16-bit
        'driver': 'GTiff'  # 儲存成 TIF 格式
    })

    # --- 7. 寫入新的 16-bit TIF 檔案 ---
    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(rgb_image_16bit)

    print(f"\n成功！")
    print(f"已從 '{os.path.basename(file_path)}' 中提取 RGB 波段。")
    print(f"新的 16-bit 影像已儲存為: {output_file}")
    print("這張 .tif 影像可以用 Windows 相片檢視器打開，並且保留了 16-bit 的原始資料。")

except FileNotFoundError:
    print(f"錯誤：找不到檔案！")
    print(f"請檢查你的路徑是否正確: {file_path}")
except Exception as e:
    print(f"發生錯誤: {e}")
    print("請確認你已安裝 'rasterio' 和 'numpy' 函式庫。")