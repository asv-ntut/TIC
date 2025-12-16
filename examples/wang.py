import rasterio
import numpy as np
from PIL import Image
import os

print("--- Sentinel-2 True Color PNG Generator (v4 - Fixed Range Stretch) ---")

# ==========================================
# 1. 設定路徑 (請修改這裡)
# ==========================================

# 輸入資料夾：包含 B02.tif, B03.tif, B04.tif 的資料夾
INPUT_DIR = r"C:\Users\GGBOY\Downloads\Tainan"

# 輸出資料夾：您想儲存 PNG 的位置
OUTPUT_DIR = r"C:\Users\GGBOY\OneDrive\桌面\除10000"

# 輸出的檔名
OUTPUT_FILENAME = 'Sentinel2_TrueColor_Tainan10000.png'

# ==========================================
# 2. 設定拉伸參數 (關鍵參數)
# ==========================================
# Sentinel-2 陸地反射率通常在 500~3000 之間。
# 設定 3500 為白點，可以讓陸地變亮，同時忽略極亮的雲 (數值>8000)。
STRETCH_MIN = 0
STRETCH_MAX = 10000
print(f"設定拉伸範圍: {STRETCH_MIN} (黑) -> {STRETCH_MAX} (白)")


# ==========================================
# 3. 函式定義
# ==========================================

def scale_band_to_8bit(band_data, fixed_min, fixed_max):
    """
    使用「固定」的數值範圍將 16-bit 波段拉伸到 8-bit。
    數值 > fixed_max 的會被截斷為 255 (純白)。
    """
    # 使用線性插值進行拉伸 (np.interp 自動處理截斷)
    scaled_data = np.interp(band_data, (fixed_min, fixed_max), (0, 255))

    # 確保原始數據中的 No Data (0) 保持為 0 (黑色)
    scaled_data[band_data == 0] = 0

    return scaled_data.astype(np.uint8)


# ==========================================
# 4. 主程式執行
# ==========================================

# 組合檔案路徑
try:
    blue_path = os.path.join(INPUT_DIR, 'B02.tif')
    green_path = os.path.join(INPUT_DIR, 'B03.tif')
    red_path = os.path.join(INPUT_DIR, 'B04.tif')

    # 檢查檔案是否存在
    for p in [blue_path, green_path, red_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到檔案: {p}")

    print("成功找到波段檔案，開始讀取...")

except FileNotFoundError as e:
    print(f"錯誤：{e}")
    print(f"請確認路徑是否正確：{INPUT_DIR}")
    exit()

# 讀取波段數據
with rasterio.open(red_path) as src:
    red_band = src.read(1)
with rasterio.open(green_path) as src:
    green_band = src.read(1)
with rasterio.open(blue_path) as src:
    blue_band = src.read(1)

print("正在處理波段亮度...")

# 執行拉伸轉換
red_8bit = scale_band_to_8bit(red_band, STRETCH_MIN, STRETCH_MAX)
green_8bit = scale_band_to_8bit(green_band, STRETCH_MIN, STRETCH_MAX)
blue_8bit = scale_band_to_8bit(blue_band, STRETCH_MIN, STRETCH_MAX)

# 堆疊成 RGB 影像
print("正在合併為 RGB 影像...")
rgb_image = np.dstack((red_8bit, green_8bit, blue_8bit))

# 建立輸出資料夾 (如果不存在)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 儲存圖片
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
print(f"正在儲存至: {output_path}")

image = Image.fromarray(rgb_image)
image.save(output_path)

print("\n✅ 轉換成功！")
print("請檢查您的輸出圖片，陸地應該會變亮且清晰。")