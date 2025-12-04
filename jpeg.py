import cv2
import numpy as np
import os
import glob
from skimage.metrics import structural_similarity as ssim

# --- 參數設定 ---
# 1. 輸入圖片的資料夾路徑
input_folder = "C:/Users/Matt/Desktop/0921"

# 2. 【這裡已修改】儲存壓縮後 JPEG 的資料夾路徑 (桌面上的 "JPEG" 資料夾)
output_folder = os.path.join(os.path.expanduser("~"), "Desktop", "JPEG")

# 3. 設定 JPEG 壓縮品質 (0-100)
jpeg_quality = 90

# --- 參數設定結束 ---


def calculate_metrics(original_path, compressed_path):
    """
    計算並回傳 BPP, PSNR, SSIM
    """
    original_image = cv2.imread(original_path)
    compressed_image = cv2.imread(compressed_path)

    if original_image is None or compressed_image is None:
        print(f"錯誤: 無法讀取 {original_path} 或 {compressed_path}")
        return None, None, None

    height, width, _ = original_image.shape
    total_pixels = width * height
    compressed_size_bytes = os.path.getsize(compressed_path)
    bpp = (compressed_size_bytes * 8) / total_pixels
    psnr = cv2.PSNR(original_image, compressed_image)
    ssim_value, _ = ssim(original_image, compressed_image, full=True, channel_axis=-1, data_range=255)

    return bpp, psnr, ssim_value


def main():
    if not os.path.isdir(input_folder):
        print(f"錯誤: 輸入資料夾 '{input_folder}' 不存在。")
        return

    # os.makedirs 會自動建立資料夾，exist_ok=True 表示如果資料夾已存在也不會報錯
    os.makedirs(output_folder, exist_ok=True)
    print(f"壓縮後的 JPEG 將儲存至: {output_folder}\n")

    print(f"正在 '{input_folder}' 中尋找 .tif 和 .tiff 檔案...")
    tif_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    tiff_paths = glob.glob(os.path.join(input_folder, "*.tiff"))
    image_paths = tif_paths + tiff_paths

    if not image_paths:
        print(f"在 '{input_folder}' 中找不到任何 TIF 檔案。")
        return

    print(f"\n--- 開始處理，JPEG 品質設定為: {jpeg_quality} ---")
    for original_path in image_paths:
        base_name = os.path.basename(original_path)
        name_without_ext = os.path.splitext(base_name)[0]

        print(f"\n處理中: {base_name}")

        original_image = cv2.imread(original_path)
        if original_image is None:
            print(f"  跳過: 無法讀取檔案。")
            continue

        compressed_path = os.path.join(output_folder, f"{name_without_ext}_quality_{jpeg_quality}.jpg")

        cv2.imwrite(compressed_path, original_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        print(f"  -> 已壓縮並儲存至: {os.path.basename(compressed_path)}")

        bpp, psnr, ssim_val = calculate_metrics(original_path, compressed_path)

        if bpp is not None:
            print(f"  [評估結果]")
            print(f"    BPP  : {bpp:.4f} bits/pixel")
            print(f"    PSNR : {psnr:.2f} dB")
            print(f"    SSIM : {ssim_val:.4f}")

    print("\n--- 所有圖片處理完畢 ---")


if __name__ == "__main__":
    main()