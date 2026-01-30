import os
import glob
import io
import torch
import math
from PIL import Image
# OpenJPEG 支援 JPEG2000
# Libjpeg 支援 JPEG
FFmpeg
from torchvision.transforms import ToTensor
from pytorch_msssim import ms_ssim
from tqdm import tqdm
import pandas as pd

# ==========================================
# 參數設定
# ==========================================
# 修改為您的圖片資料夾路徑
INPUT_DIR = "/home/asvserver/TIC/s2_combined/test_8bit" 
FILE_EXT = "*.tif"

# 測試參數 (RD Curve 取樣點)
JPEG_QUALITIES = [10, 30, 50, 70, 90]
JP2_RATIOS = [100, 60, 40, 20, 10]

# ==========================================
# 評估指標函式 (PSNR, MS-SSIM, BPP)
# ==========================================
def calculate_psnr(tensor_orig, tensor_comp):
    """計算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((tensor_orig - tensor_comp) ** 2)
    if mse == 0:
        return float('inf')
    # 因為 ToTensor 已經將像素值正規化到 [0, 1]，所以 MAX=1.0
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def calculate_metrics(img_orig_pil, img_comp_pil, compressed_size_bytes):
    """計算 BPP, PSNR, MS-SSIM"""
    # 1. 計算 BPP (Bits Per Pixel)
    width, height = img_orig_pil.size
    total_pixels = width * height
    bpp = (compressed_size_bytes * 8) / total_pixels
    
    # 將 PIL 轉為 RGB 並正規化成 [1, C, H, W] 的 PyTorch Tensor
    img_orig = img_orig_pil.convert('RGB')
    img_comp = img_comp_pil.convert('RGB')
    tensor_orig = ToTensor()(img_orig).unsqueeze(0)
    tensor_comp = ToTensor()(img_comp).unsqueeze(0)

    # 2. 計算 PSNR
    psnr_val = calculate_psnr(tensor_orig, tensor_comp)

    # 3. 計算 MS-SSIM
    ssim_val = ms_ssim(tensor_orig, tensor_comp, data_range=1.0, size_average=True).item()
    
    return bpp, psnr_val, ssim_val

# ==========================================
# 主程式
# ==========================================
def main():
    image_paths = glob.glob(os.path.join(INPUT_DIR, FILE_EXT))
    num_images = len(image_paths)

    if num_images == 0:
        print(f"錯誤: 在 {INPUT_DIR} 找不到任何 {FILE_EXT} 檔案。")
        return

    print(f"找到 {num_images} 張圖片，準備開始大規模測試...")
    all_results = []

    # --- 1. 測試 JPEG 系列 ---
    print("\n[1/2] 正在測試 JPEG 系列...")
    for q in JPEG_QUALITIES:
        total_bpp = total_psnr = total_msssim = 0
        
        # 使用 tqdm 顯示單個參數的進度
        for path in tqdm(image_paths, desc=f"JPEG Q={q}", leave=False):
            with Image.open(path) as img:
                orig = img.convert('RGB')
                buf = io.BytesIO()
                orig.save(buf, format="JPEG", quality=q)
                size_bytes = len(buf.getvalue())
                
                buf.seek(0)
                comp = Image.open(buf)
                
                bpp, psnr, msssim = calculate_metrics(orig, comp, size_bytes)
                total_bpp += bpp
                total_psnr += psnr
                total_msssim += msssim

        # 儲存平均值
        all_results.append({
            "Method": "JPEG",
            "Parameter": f"Q={q}",
            "Avg_BPP": total_bpp / num_images,
            "Avg_PSNR": total_psnr / num_images,
            "Avg_MS-SSIM": total_msssim / num_images
        })

    # --- 2. 測試 JPEG2000 系列 ---
    print("\n[2/2] 正在測試 JPEG2000 系列...")
    for r in JP2_RATIOS:
        total_bpp = total_psnr = total_msssim = 0
        
        for path in tqdm(image_paths, desc=f"JP2 Ratio={r}", leave=False):
            with Image.open(path) as img:
                orig = img.convert('RGB')
                buf = io.BytesIO()
                orig.save(buf, format="JPEG2000", quality_layers=[r])
                size_bytes = len(buf.getvalue())
                
                buf.seek(0)
                comp = Image.open(buf)
                
                bpp, psnr, msssim = calculate_metrics(orig, comp, size_bytes)
                total_bpp += bpp
                total_psnr += psnr
                total_msssim += msssim

        # 儲存平均值
        all_results.append({
            "Method": "JPEG2000",
            "Parameter": f"Ratio={r}",
            "Avg_BPP": total_bpp / num_images,
            "Avg_PSNR": total_psnr / num_images,
            "Avg_MS-SSIM": total_msssim / num_images
        })

    # --- 3. 輸出最終報告 ---
    df = pd.DataFrame(all_results)
    # 以 BPP 排序，方便後續畫圖對齊
    df = df.sort_values(by=['Method', 'Avg_BPP']).reset_index(drop=True)
    
    print("\n" + "="*70)
    print(f" 大規模測試結果 (共處理 {num_images} 張影像) ")
    print("="*70)
    pd.set_option('display.unicode.east_asian_width', True)
    # 格式化輸出小數點位數
    pd.options.display.float_format = '{:.4f}'.format
    print(df.to_string(index=False))
    
    # 存檔成 CSV (論文常用)
    csv_name = "benchmark_results.csv"
    df.to_csv(csv_name, index=False)
    print("="*70)
    print(f"結果已匯出至: {csv_name}")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()