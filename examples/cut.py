import os
import sys
import random  # 新增 random 模組

# 嘗試匯入 PIL
try:
    from PIL import Image
except ImportError:
    print("錯誤：尚未安裝 Pillow 套件。")
    print("請打開 CMD 輸入指令安裝： pip install pillow")
    sys.exit(1)


def main():
    # ==========================================
    # 1. 設定檔案路徑 (已根據你的截圖修改)
    # ==========================================
    # 輸入: 你的 TIF 檔案位置
    input_path = r"C:\Users\Matt\Desktop\wang\taiwan\Taipei\TCI.tif"

    # 輸出: 存在桌面的 wang 資料夾下，檔名加上 random
    output_path = r"C:\Users\Matt\Desktop\wang\taiwan\Taipei\4096x3072taipei.png"

    # ==========================================
    # 2. 設定要裁切的大小
    # ==========================================
    TARGET_WIDTH = 4096
    TARGET_HEIGHT = 3072

    # 檢查輸入檔案是否存在
    if not os.path.exists(input_path):
        print(f"錯誤：找不到檔案！請確認路徑是否正確：\n{input_path}")
        return

    try:
        # 解除像素限制 (因為衛星圖通常很大)
        Image.MAX_IMAGE_PIXELS = None

        print(f"正在讀取圖片：{input_path} ...")

        with Image.open(input_path) as img:
            orig_w, orig_h = img.size
            print(f"原始圖片尺寸：{orig_w} x {orig_h}")

            # 確保原始圖片夠大
            if orig_w < TARGET_WIDTH or orig_h < TARGET_HEIGHT:
                print("❌ 錯誤：原始圖片比目標尺寸還小，無法裁切！")
                return

            # ==========================================
            # 3. 計算隨機座標 (關鍵修改)
            # ==========================================
            # 可移動的最大 X 和 Y 範圍
            max_x = orig_w - TARGET_WIDTH
            max_y = orig_h - TARGET_HEIGHT

            # 隨機選一個起點
            left = random.randint(0, max_x)
            top = random.randint(0, max_y)

            right = left + TARGET_WIDTH
            bottom = top + TARGET_HEIGHT

            crop_box = (left, top, right, bottom)

            print(f"隨機裁切點：(x={left}, y={top})")
            print(f"裁切範圍：{crop_box}")

            # 執行裁切
            cropped_img = img.crop(crop_box)

            # ==========================================
            # 4. 儲存檔案
            # ==========================================
            print(f"正在儲存至：{output_path}")
            # 雖然來源是 TIF，但我們可以存成 PNG 比較方便看
            cropped_img.save(output_path, format="PNG")

            print("-" * 30)
            print("✅ 處理完成！")

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")


if __name__ == "__main__":
    main()