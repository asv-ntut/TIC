import os
from PIL import Image

# ================= 設定區 =================
# 圖片路徑 (根據您的截圖設定)
image_path = r"C:\Users\Matt\Desktop\wang\10000\taipei.png"

# 輸出資料夾 (您指定的存放位置)
output_folder = r"C:\Users\Matt\Desktop\wang\4096 3072"

# 目標切割大小
CROP_WIDTH = 4096
CROP_HEIGHT = 3072


# =========================================

def split_image_strict(image_path, output_folder, crop_w, crop_h):
    # 如果輸出資料夾不存在，自動建立
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 解除像素限制 (避免讀取大圖時報錯)
    Image.MAX_IMAGE_PIXELS = None

    try:
        print(f"正在讀取: {image_path} ...")

        if not os.path.exists(image_path):
            print(f"錯誤: 找不到檔案，請檢查路徑是否正確: {image_path}")
            return

        img = Image.open(image_path)
        img_w, img_h = img.size
        print(f"原始尺寸: {img_w} x {img_h}")

        # 計算橫向和縱向可以切出幾張「完整」的圖片 (無條件捨去小數點，捨棄邊緣)
        num_cols = img_w // crop_w
        num_rows = img_h // crop_h

        print(f"嚴格模式計算結果: 橫向 {num_cols} 張, 縱向 {num_rows} 張")
        print(f"預計產出: {num_cols * num_rows} 張 (其餘邊緣將捨棄)")

        count = 0

        # 使用計算好的數量跑迴圈
        for row in range(num_rows):
            for col in range(num_cols):
                # 計算座標
                x = col * crop_w
                y = row * crop_h

                box = (x, y, x + crop_w, y + crop_h)
                tile = img.crop(box)

                # 檔名標記座標，例如 hualien_r0_c0.png
                # 使用 base filename 來命名，比較直觀
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = f"{base_name}_tile_r{row}_c{col}.png"
                output_path = os.path.join(output_folder, output_filename)

                tile.save(output_path)
                print(f"已儲存: {output_filename}")
                count += 1

        print(f"\n處理完成！共產生 {count} 張嚴格尺寸 ({crop_w}x{crop_h}) 的圖片。")
        print(f"檔案已存放於: {output_folder}")

    except Exception as e:
        print(f"發生錯誤: {e}")


if __name__ == "__main__":
    split_image_strict(image_path, output_folder, CROP_WIDTH, CROP_HEIGHT)