import os
import glob
import shutil
from tqdm import tqdm
import rasterio
import numpy as np
from PIL import Image  # 引入 PIL (Pillow) 函式庫來儲存 PNG

# ==========================================================
# ---               使用者設定區 (請修改這裡)            ---
# ==========================================================

# 1. 來源資料夾：您存放【原始 16位元 TIF】的 train/val/test 的上層資料夾
SOURCE_BASE_DIR = r'D:\s2_cloudfree_split_60k'

# 2. 輸出資料夾：處理完的【8位元 PNG】圖片要存在哪裡？
#    建議是一個全新的、乾淨的資料夾
OUTPUT_BASE_DIR = r'D:\s2_cloudfree_visual_png_60k'

# 3. 定義要選取的 RGB 波段 (紅, 綠, 藍)
RGB_BANDS = [4, 3, 2]


# ==========================================================
# ---                  程式碼主體 (無需修改)             ---
# ==========================================================

def normalize_for_visual(image_data):
    """將 uint16 數據轉換成適合觀看的 0-1 float 數據"""
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val > min_val:
        scaled_data = (image_data - min_val) / (max_val - min_val)
    else:
        scaled_data = image_data - min_val  # 避免除以零
    return np.clip(scaled_data, 0, 1)


def process_and_save_as_png():
    """
    遍歷來源 TIF 檔案，進行視覺化處理，並儲存成 8位元 PNG 檔案。
    """
    print("--- 模式：儲存【視覺化 PNG】(8位元，看起來漂亮，數據有損) ---")
    print(f"來源資料夾: {SOURCE_BASE_DIR}")
    print(f"輸出資料夾: {OUTPUT_BASE_DIR}\n")

    try:
        splits = ['train', 'val', 'test']
        for split_name in splits:
            source_dir = os.path.join(SOURCE_BASE_DIR, split_name)
            output_dir = os.path.join(OUTPUT_BASE_DIR, split_name)
            os.makedirs(output_dir, exist_ok=True)

            source_files = glob.glob(os.path.join(source_dir, '*.tif'))
            if not source_files:
                print(f"警告：在 {source_dir} 中找不到 .tif 檔案，跳過。")
                continue

            print(f"正在處理 '{split_name}' 資料夾...")
            for src_path in tqdm(source_files, desc=f"處理 {split_name}"):
                with rasterio.open(src_path) as src:
                    # 讀取 uint16 格式的 RGB 數據
                    rgb_data_uint16 = src.read(RGB_BANDS)

                # --- 進行視覺化處理 ---
                # 1. 轉成 float32 並用最小-最大值正規化到 0-1
                normalized_data = normalize_for_visual(rgb_data_uint16.astype(np.float32))

                # 2. 轉換維度 (C, H, W) -> (H, W, C) 以符合 PIL 格式
                display_img = np.transpose(normalized_data, (1, 2, 0))

                # 3. 轉換成 8位元 (0-255)
                img_uint8 = (display_img * 255).astype(np.uint8)

                # --- 儲存成 PNG 檔案 ---
                file_name_without_ext = os.path.splitext(os.path.basename(src_path))[0]
                output_path = os.path.join(output_dir, f"{file_name_without_ext}.png")

                Image.fromarray(img_uint8).save(output_path)

        print("\n\n✅ 所有檔案已成功轉換並儲存為 PNG！")

    except Exception as e:
        print(f"\n❌ 處理過程中發生錯誤: {e}")


if __name__ == '__main__':
    process_and_save_as_png()