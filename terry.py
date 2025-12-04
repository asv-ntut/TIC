import rasterio
import numpy as np
from pathlib import Path
import random

# ==========================================================
# ---                  使用者設定區                      ---
# ==========================================================

# 1. 來源大圖的檔案名稱
SOURCE_IMAGE_FILENAME = 'C:\\Users\\Matt\\Downloads\\Tokyo.tif'

# 2. 輸出資料夾的名稱 (程式會自動建立)
OUTPUT_DIR_NAME = 'Tokyo_Patches'

# 3. 要裁切的小圖尺寸
PATCH_WIDTH = 256
PATCH_HEIGHT = 256

# 4. 要裁切的小圖數量
NUM_PATCHES = 10


# ==========================================================
# ---                  主程式                            ---
# ==========================================================

def crop_random_patches():
    """
    從一張大的 TIF 影像中，隨機裁切出數個小補丁 (patches)。
    """
    # --- 步驟 1: 檢查來源檔案是否存在 ---
    source_path = Path(SOURCE_IMAGE_FILENAME)
    if not source_path.exists():
        print(f"錯誤：找不到來源檔案 '{SOURCE_IMAGE_FILENAME}'！")
        print("請確保此腳本與您的 TIF 檔案放在同一個資料夾中。")
        return

    # --- 步驟 2: 建立輸出資料夾 ---
    output_dir = Path(OUTPUT_DIR_NAME)
    output_dir.mkdir(exist_ok=True)
    print(f"圖片將儲存至: {output_dir.resolve()}")

    # --- 步驟 3: 讀取大圖的尺寸和資訊 ---
    with rasterio.open(source_path) as src:
        # 獲取原始圖片的寬度和高度
        original_width = src.width
        original_height = src.height

        # 獲取原始圖片的檔案資訊 (profile)，方便後續儲存
        original_profile = src.profile

        print(f"成功讀取 '{SOURCE_IMAGE_FILENAME}'，尺寸為 {original_width} x {original_height} 像素。")

        # --- 步驟 4: 檢查尺寸是否足夠裁切 ---
        if original_width < PATCH_WIDTH or original_height < PATCH_HEIGHT:
            print(f"錯誤：原始圖片尺寸小於要裁切的 {PATCH_WIDTH}x{PATCH_HEIGHT} 尺寸！")
            return

        # --- 步驟 5: 產生隨機座標並裁切儲存 ---
        print(f"\n開始隨機裁切 {NUM_PATCHES} 張 {PATCH_WIDTH}x{PATCH_HEIGHT} 的小圖...")

        generated_coords = set()  # 用來確保座標不重複

        for i in range(NUM_PATCHES):
            # 為了避免座標重複，我們用一個 while 迴圈來確保每次都拿到新的
            while True:
                # 計算可以放置裁切框左上角的最大 X 和 Y 座標
                max_x = original_width - PATCH_WIDTH
                max_y = original_height - PATCH_HEIGHT

                # 隨機產生左上角的座標
                random_x = random.randint(0, max_x)
                random_y = random.randint(0, max_y)

                if (random_x, random_y) not in generated_coords:
                    generated_coords.add((random_x, random_y))
                    break

            # 定義要讀取的視窗 (window)
            window = rasterio.windows.Window(random_x, random_y, PATCH_WIDTH, PATCH_HEIGHT)

            # 從大圖中讀取這個視窗的數據
            patch_data = src.read(window=window)

            # 更新檔案資訊以符合新的小圖尺寸
            profile = original_profile.copy()
            profile.update({
                'width': PATCH_WIDTH,
                'height': PATCH_HEIGHT,
                'transform': rasterio.windows.transform(window, src.transform)
            })

            # 產生新的檔名
            output_filename = f"patch_{i + 1:02d}_{random_x}_{random_y}.tif"
            output_path = output_dir / output_filename

            # 儲存裁切後的小圖
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(patch_data)

            print(f"  - 已儲存: {output_filename}")

    print(f"\n✅ 成功完成！共擷取出 {NUM_PATCHES} 張圖片。")


if __name__ == '__main__':
    crop_random_patches()