import os
import glob
import io
import torch
import math
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch_msssim import ms_ssim
from tqdm import tqdm
import pandas as pd

# ==========================================
# 參數設定
# ==========================================
INPUT_DIR = "/home/asvserver/TIC/s2_combined/test_8bit" 
FILE_EXT = "*.tif"

# JPEG2000 測試參數 (目標壓縮倍率)
JP2_RATIOS = [100, 60, 40, 20, 10]

# ==========================================
# 評估指標函式
# ==========================================
def calculate_psnr(tensor_orig, tensor_comp):
    """計算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((tensor_orig - tensor_comp) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()

def calculate_metrics(img_orig_pil, img_comp_pil, compressed_size_bytes):
    """計算 BPP, PSNR, MS-SSIM"""
    width, height = img_orig_pil.size
    total_pixels = width * height
    bpp = (compressed_size_bytes * 8) / total_pixels
    
    img_orig = img_orig_pil.convert('RGB')
    img_comp = img_comp_pil.convert('RGB')
    tensor_orig = ToTensor()(img_orig).unsqueeze(0)
    tensor_comp = ToTensor()(img_comp).unsqueeze(0)

    psnr_val = calculate_psnr(tensor_orig, tensor_comp)
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

    print(f"找到 {num_images} 張圖片，準備開始 JPEG2000 有損(Lossy) 專項測試...")
    all_results = []

    # --- 測試 JPEG2000 系列 ---
    for r in JP2_RATIOS:
        total_bpp = total_psnr = total_msssim = 0
        
        for path in tqdm(image_paths, desc=f"JP2 Ratio={r} (Lossy)", leave=False):
            with Image.open(path) as img:
                orig = img.convert('RGB')
                buf = io.BytesIO()
                
                # 【關鍵修改】加入 irreversible=True，強制使用有損的 9/7 DWT 小波轉換
                orig.save(buf, format="JPEG2000", quality_layers=[r], irreversible=True)
                
                size_bytes = len(buf.getvalue())
                
                buf.seek(0)
                comp = Image.open(buf)
                
                bpp, psnr, msssim = calculate_metrics(orig, comp, size_bytes)
                total_bpp += bpp
                total_psnr += psnr
                total_msssim += msssim

        # 儲存平均值
        all_results.append({
            "Method": "JPEG2000 (Lossy)",
            "Ratio": r,
            "Avg_BPP": total_bpp / num_images,
            "Avg_PSNR": total_psnr / num_images,
            "Avg_MS-SSIM": total_msssim / num_images
        })

    # --- 輸出最終報告 ---
    df = pd.DataFrame(all_results)
    
    # 計算並新增實際壓縮率欄位 (Assuming 24 bpp originally)
    df.insert(2, 'Actual_CR', 24.0 / df['Avg_BPP'])

    print("\n" + "="*80)
    print(f" JPEG2000 有損模式測試結果 (共處理 {num_images} 張影像) ")
    print("="*80)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.options.display.float_format = '{:.4f}'.format
    print(df.to_string(index=False))
    
    # 存檔成 CSV
    csv_name = "_jp2_lossy_results.csv"
    df.to_csv(csv_name, index=False)
    print("="*80)
    print(f"結果已匯出至: {csv_name}")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()