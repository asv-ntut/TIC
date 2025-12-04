import os
import random
from PIL import Image

# ================= 設定區 =================
# 1. 大圖路徑
IMAGE_PATH = r"C:\Users\Matt\Desktop\wang\Sentinel2_TrueColor.png"

# 2. 輸出資料夾
OUTPUT_DIR = r"C:\Users\Matt\Desktop\wang\256"

# 3. 裁切設定
CROP_SIZE = 256  # 寬高
NUM_CROPS = 100  # 張數
# =========================================

# 解除大圖限制 (防止報錯)
Image.MAX_IMAGE_PIXELS = None


def main():
    # 1. 檢查檔案
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 找不到檔案：{IMAGE_PATH}")
        return

    # 2. 建立資料夾
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已建立/確認資料夾：{OUTPUT_DIR}")

    print("🚀 正在讀取大圖...")
    try:
        big_img = Image.open(IMAGE_PATH)
        img_w, img_h = big_img.size
        print(f"✅ 圖片讀取成功！尺寸：{img_w} x {img_h}")
    except Exception as e:
        print(f"❌ 讀取失敗：{e}")
        return

    # 檢查是否夠切
    if img_w < CROP_SIZE or img_h < CROP_SIZE:
        print("⚠️ 圖片太小了，切不了 256x256！")
        return

    print(f"✂️ 開始執行「完全隨機」裁切 ({NUM_CROPS} 張)...")

    for i in range(NUM_CROPS):
        # 1. 隨機產生座標 (範圍：0 ~ 圖片寬度-256)
        x = random.randint(0, img_w - CROP_SIZE)
        y = random.randint(0, img_h - CROP_SIZE)

        # 2. 直接裁切 (不過濾任何內容)
        crop = big_img.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))

        # 3. 存檔
        save_name = f"crop_{i:03d}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        crop.save(save_path)

        # 簡單顯示進度
        if (i + 1) % 10 == 0:
            print(f"   已完成 {i + 1}/{NUM_CROPS} 張...")

    print(f"\n✅ 全部完成！100 張圖片已存入：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()