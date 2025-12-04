import argparse
import json
import math
import os
import sys
import time
import subprocess
import shutil
from collections import defaultdict, OrderedDict
from typing import List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==============================================================================
# 設定：請確認這裡的檔名與您的訓練檔名一致
# ==============================================================================
try:
    from conv2 import SimpleConvStudentModel
except ImportError:
    print("錯誤: 找不到 conv2.py，請確認您的訓練程式碼檔名是否為 conv2.py")
    sys.exit(1)

# ==============================================================================
# 指定輸出路徑 (User Request)
# ==============================================================================
# 使用 raw string (r"...") 避免 Windows 路徑的反斜線被轉義
DEFAULT_OUTPUT_DIR = r"C:\Users\Matt\Desktop\研究進度\1130"


# --- 定義要注入的方法 (Monkey Patching) ---
def compress_method(self, x):
    y = self.g_a(x)
    z = self.h_a(y)
    z_strings = self.entropy_bottleneck.compress(z)
    z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    gaussian_params = self.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = self.gaussian_conditional.build_indexes(scales_hat)
    y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
    return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


def decompress_method(self, strings, shape):
    assert isinstance(strings, list) and len(strings) == 2
    z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    gaussian_params = self.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = self.gaussian_conditional.build_indexes(scales_hat)
    y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
    x_hat = self.g_s(y_hat).clamp_(0, 1)
    return {"x_hat": x_hat}


# --- 執行注入 ---
print("正在載入模型壓縮方法...")
SimpleConvStudentModel.compress = compress_method
SimpleConvStudentModel.decompress = decompress_method

# 嘗試匯入 rasterio
try:
    import rasterio
except ImportError:
    rasterio = None

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
IMG_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def collect_images(rootpath: str) -> List[str]:
    # 如果輸入是單一檔案
    if os.path.isfile(rootpath):
        return [rootpath]
    # 如果輸入是資料夾
    return [os.path.join(rootpath, f) for f in os.listdir(rootpath) if
            os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse) if mse > 0 else float('inf')


# ==============================================================================
# 讀圖函式
# ==============================================================================
def read_image_patch(filepath: str, crop_box=None) -> torch.Tensor:
    """
    讀取影像並正規化為 0~1 的 Tensor。
    crop_box: (left, upper, right, lower)
    """
    ext = os.path.splitext(filepath)[-1].lower()

    # TIF 處理
    if ext in ['.tif', '.tiff']:
        if rasterio is None: raise RuntimeError("需安裝 rasterio")
        SCALE = 10000.0
        with rasterio.open(filepath) as src:
            if crop_box:
                # window: col_off, row_off, width, height
                left, upper, right, lower = crop_box
                window = rasterio.windows.Window(left, upper, right - left, lower - upper)
                raw_data = src.read(window=window).astype(np.float32)
            else:
                raw_data = src.read().astype(np.float32)

        if np.isnan(raw_data).any(): raw_data = np.nan_to_num(raw_data)
        if raw_data.shape[0] >= 3:
            rgb_data = raw_data[:3, :, :]
        else:
            rgb_data = raw_data

        clipped_data = np.clip(rgb_data, 0.0, 10000.0)
        return torch.from_numpy(clipped_data / SCALE)

    # PNG/JPG 處理
    else:
        img = Image.open(filepath).convert("RGB")
        if crop_box:
            img = img.crop(crop_box)
        return transforms.ToTensor()(img)


# ==============================================================================
# 推論核心
# ==============================================================================
@torch.no_grad()
def inference(model, x, save=False, output_dir=None, base_filename=None):
    # Padding 至 64 倍數 (雖然 256 本身就是 64 倍數，但為了保險保留此邏輯)
    h, w = x.size(2), x.size(3)
    p = 64
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom))

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    if save and output_dir is not None:
        # 儲存二進位檔 (.bin)
        bin_path = os.path.join(output_dir, f"{base_filename}.bin")
        try:
            with open(bin_path, "wb") as f:
                shape = out_enc["shape"]
                f.write(shape[0].to_bytes(2, 'little'))
                f.write(shape[1].to_bytes(2, 'little'))
                y_str = out_enc["strings"][0][0]
                z_str = out_enc["strings"][1][0]
                f.write(len(z_str).to_bytes(4, 'little'))
                f.write(z_str)
                f.write(len(y_str).to_bytes(4, 'little'))
                f.write(y_str)
        except Exception as e:
            print(f"Error saving bin: {e}")

        # 儲存還原的小圖 (.png)
        png_path = os.path.join(output_dir, f"{base_filename}_rec.png")
        rec_img = out_dec["x_hat"].squeeze().cpu().clamp(0, 1)
        transforms.ToPILImage()(rec_img).save(png_path)

    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "x_hat": out_dec["x_hat"]  # 回傳還原圖以便拼貼
    }


# ==============================================================================
# 切割、推論與拼貼
# ==============================================================================
def eval_model(model, filepaths, output_dir=None):
    device = next(model.parameters()).device
    PATCH_SIZE = 256

    print(f"輸出目錄設定為: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for i, f in enumerate(filepaths):
        print(f"\n[{i + 1}/{len(filepaths)}] 正在處理: {os.path.basename(f)}")
        base_filename = os.path.splitext(os.path.basename(f))[0]

        # 1. 取得影像原始尺寸
        if rasterio and os.path.splitext(f)[-1].lower() in ['.tif', '.tiff']:
            with rasterio.open(f) as src:
                img_w, img_h = src.width, src.height
        else:
            with Image.open(f) as img:
                img_w, img_h = img.size

        print(f"  原始尺寸: {img_w} x {img_h}")

        # 2. 計算切割數量 (預期 4096x3072 會切成 16x12=192 張)
        num_cols = img_w // PATCH_SIZE  # 4096 / 256 = 16
        num_rows = img_h // PATCH_SIZE  # 3072 / 256 = 12
        total_patches = num_cols * num_rows

        print(f"  將分割為: {num_rows} (列) x {num_cols} (行) = {total_patches} 張小圖 (256x256)")

        # 為這張圖建立專屬資料夾
        current_save_dir = os.path.join(output_dir, base_filename)
        os.makedirs(current_save_dir, exist_ok=True)

        # 準備一張大畫布來拼貼還原後的圖 (用於視覺化確認)
        full_recon_img = Image.new('RGB', (img_w, img_h))

        metrics = defaultdict(float)

        # 3. 雙層迴圈進行切割與推論
        count = 0
        for row in range(num_rows):
            for col in range(num_cols):
                count += 1
                # 計算裁切範圍
                left = col * PATCH_SIZE
                upper = row * PATCH_SIZE
                right = left + PATCH_SIZE
                lower = upper + PATCH_SIZE
                box = (left, upper, right, lower)

                # 讀取小圖
                x = read_image_patch(f, box).unsqueeze(0).to(device)

                # 命名: 加上 row/col 讓你知道這是哪一塊
                patch_name = f"{base_filename}_row{row}_col{col}"

                # 執行推論
                rv = inference(model, x, save=True, output_dir=current_save_dir, base_filename=patch_name)

                # 累加數據
                metrics["psnr"] += rv["psnr"]
                metrics["bpp"] += rv["bpp"]
                metrics["ms-ssim"] += rv["ms-ssim"]

                # 將還原的小圖貼回大畫布
                rec_tensor = rv["x_hat"].squeeze().cpu().clamp(0, 1)
                rec_patch_pil = transforms.ToPILImage()(rec_tensor)
                full_recon_img.paste(rec_patch_pil, (left, upper))

                print(f"  處理進度: {count}/{total_patches} | Patch ({row},{col}) PSNR: {rv['psnr']:.2f} dB", end='\r')

        # 4. 儲存完整拼貼圖
        full_recon_path = os.path.join(output_dir, f"{base_filename}_FULL_RECONSTRUCTED.png")
        full_recon_img.save(full_recon_path)
        print(f"\n  已儲存完整還原圖至: {full_recon_path}")

        # 計算平均數據
        avg_psnr = metrics["psnr"] / total_patches
        avg_bpp = metrics["bpp"] / total_patches
        avg_ssim = metrics["ms-ssim"] / total_patches

        print(f"  > 平均 PSNR: {avg_psnr:.2f} dB")
        print(f"  > 平均 BPP:  {avg_bpp:.4f}")
        print(f"  > 平均 SSIM: {avg_ssim:.4f}")


# ==============================================================================
# Main
# ==============================================================================
def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 簡單假設
    N, M = 128, 192
    try:
        N = new_state_dict['g_a.0.weight'].size(0)
        keys = sorted([k for k in new_state_dict.keys() if 'g_a' in k and 'weight' in k])
        M = new_state_dict[keys[-1]].size(0)
    except:
        pass

    model = SimpleConvStudentModel(N=N, M=M)
    model.load_state_dict(new_state_dict, strict=False)
    return model.eval()


def main(argv):
    parser = argparse.ArgumentParser()
    # 這裡改成可以接受單張圖片路徑或資料夾
    parser.add_argument("input_path", type=str, help="Path to input image or folder")
    parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to .pth.tar")
    parser.add_argument("--cuda", action="store_true", default=True)
    args = parser.parse_args(argv)

    if not os.path.exists(args.input_path):
        print(f"Input path not found: {args.input_path}")
        sys.exit(1)

    filepaths = collect_images(args.input_path)
    print(f"Found {len(filepaths)} images to process.")

    model = load_checkpoint(args.checkpoint)
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()

    print("Updating entropy model CDFs...")
    model.update(force=True)

    # 直接使用寫死的路徑
    eval_model(model, filepaths, output_dir=DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main(sys.argv[1:])