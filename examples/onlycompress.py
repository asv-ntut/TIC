import argparse
import os
import sys
import glob
import re
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==============================================================================
# 設定：匯入模型
# ==============================================================================
try:
    from conv2 import SimpleConvStudentModel
except ImportError:
    print("錯誤: 找不到 conv2.py，請確認檔案位置。")
    sys.exit(1)

# 嘗試匯入 rasterio (若原始圖是 TIF 需要)
try:
    import rasterio
except ImportError:
    rasterio = None
    # 嘗試匯入 ms_ssim (用於計算分數)
try:
    from pytorch_msssim import ms_ssim

    HAS_MSSSIM = True
except ImportError:
    print("警告: 未安裝 pytorch_msssim，將跳過 SSIM 計算。(可執行 pip install pytorch-msssim 安裝)")
    HAS_MSSSIM = False


# ==============================================================================
# Monkey Patching: 注入解壓縮方法
# ==============================================================================
def decompress_method(self, strings, shape):
    assert isinstance(strings, list) and len(strings) == 2
    z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    gaussian_params = self.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = self.gaussian_conditional.build_indexes(scales_hat)
    y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
    x_hat = self.g_s(y_hat).clamp_(0, 1)
    return {"x_hat": x_hat}


SimpleConvStudentModel.decompress = decompress_method


# ==============================================================================
# 工具函式
# ==============================================================================
def load_bin_file(bin_path):
    """讀取 .bin 檔案並還原成 strings 和 shape"""
    with open(bin_path, "rb") as f:
        # 讀取 shape
        h = int.from_bytes(f.read(2), 'little')
        w = int.from_bytes(f.read(2), 'little')
        shape = (h, w)

        # 讀取 z_string
        len_z = int.from_bytes(f.read(4), 'little')
        z_str = f.read(len_z)

        # 讀取 y_string
        len_y = int.from_bytes(f.read(4), 'little')
        y_str = f.read(len_y)

    return {"strings": [[y_str], [z_str]], "shape": shape}


@torch.no_grad()
def process_decompress(model, bin_path, device):
    data = load_bin_file(bin_path)
    out_dec = model.decompress(data["strings"], data["shape"])
    x_hat = out_dec["x_hat"]

    # 處理 Patch 大小 (預設 256)
    target_h, target_w = 256, 256
    curr_h, curr_w = x_hat.size(2), x_hat.size(3)

    if curr_h != target_h or curr_w != target_w:
        padding_left = (curr_w - target_w) // 2
        padding_top = (curr_h - target_h) // 2
        x_hat = x_hat[:, :, padding_top:padding_top + target_h, padding_left:padding_left + target_w]

    return x_hat


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """計算 PSNR"""
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse) if mse > 0 else float('inf')


def read_original_image(filepath: str) -> torch.Tensor:
    """讀取原始圖片並轉為 Tensor (與你原本的邏輯一致)"""
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.tif', '.tiff']:
        if rasterio is None: raise RuntimeError("需安裝 rasterio 才能讀取 TIF")
        SCALE = 10000.0
        with rasterio.open(filepath) as src:
            raw_data = src.read().astype(np.float32)
        if np.isnan(raw_data).any(): raw_data = np.nan_to_num(raw_data)
        rgb_data = raw_data[:3, :, :] if raw_data.shape[0] >= 3 else raw_data
        clipped_data = np.clip(rgb_data, 0.0, 10000.0)
        return torch.from_numpy(clipped_data / SCALE)
    else:
        img = Image.open(filepath).convert("RGB")
        return transforms.ToTensor()(img)


def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

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


# ==============================================================================
# 主程式
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Image Decompression & PSNR Tool")
    parser.add_argument("bin_dir", type=str, help="Directory containing .bin files")
    parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to .pth model")
    # 新增參數：原始圖片路徑
    parser.add_argument("--original", type=str, default=None, help="Path to original image (for PSNR calculation)")
    parser.add_argument("--cuda", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    PATCH_SIZE = 256

    # 1. 搜尋 .bin 檔案
    bin_files = glob.glob(os.path.join(args.bin_dir, "*.bin"))
    if not bin_files:
        print(f"在 {args.bin_dir} 找不到任何 .bin 檔案")
        sys.exit(1)

    print(f"找到 {len(bin_files)} 個壓縮檔，準備解壓縮...")

    # 2. 分析檔名
    max_row, max_col = 0, 0
    pattern = re.compile(r"_row(\d+)_col(\d+)\.bin$")
    valid_files = []
    base_name = ""

    for f in bin_files:
        match = pattern.search(f)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            max_row = max(max_row, r)
            max_col = max(max_col, c)
            valid_files.append((r, c, f))
            if base_name == "":
                base_name = os.path.basename(f).replace(match.group(0), "")

    # 計算畫布大小
    canvas_w = (max_col + 1) * PATCH_SIZE
    canvas_h = (max_row + 1) * PATCH_SIZE
    print(f"偵測到矩陣: {max_row + 1}x{max_col + 1} | 重建畫布: {canvas_w}x{canvas_h}")

    full_recon_img = Image.new('RGB', (canvas_w, canvas_h))

    # 3. 載入模型
    model = load_checkpoint(args.checkpoint).to(device)
    model.update(force=True)

    # 4. 解壓縮並拼貼
    count = 0
    for r, c, fpath in valid_files:
        count += 1
        print(f"解壓縮: {os.path.basename(fpath)} ({count}/{len(valid_files)})", end='\r')
        x_hat = process_decompress(model, fpath, device)
        rec_tensor = x_hat.squeeze().cpu().clamp(0, 1)
        rec_patch_pil = transforms.ToPILImage()(rec_tensor)

        left = c * PATCH_SIZE
        upper = r * PATCH_SIZE
        full_recon_img.paste(rec_patch_pil, (left, upper))

    print("\n解壓縮完成，正在儲存大圖...")

    output_filename = f"{base_name}_RECONSTRUCTED.png"
    output_path = os.path.join(args.bin_dir, output_filename)
    full_recon_img.save(output_path)
    print(f"完整還原圖已儲存至: {output_path}")

    # ==========================================================================
    # 5. 計算 PSNR (如果使用者有提供原始圖)
    # ==========================================================================
    if args.original:
        print("-" * 40)
        print("正在計算 PSNR...")

        if not os.path.exists(args.original):
            print(f"錯誤: 找不到原始圖片 {args.original}")
            return

        # 讀取原始圖
        try:
            # 使用與訓練時相同的讀取邏輯 (正規化到 0-1)
            gt_tensor = read_original_image(args.original)

            # 讀取剛重建好的圖 (轉為 Tensor 0-1)
            rec_tensor = transforms.ToTensor()(full_recon_img)

            # 確保尺寸一致 (針對邊緣可能被裁切的情況)
            # 如果重建圖比原始圖小 (因為 patch 沒切滿)，則裁切原始圖來對齊
            # 如果重建圖比原始圖大 (因為 padding)，則裁切重建圖
            h_gt, w_gt = gt_tensor.shape[1], gt_tensor.shape[2]
            h_rec, w_rec = rec_tensor.shape[1], rec_tensor.shape[2]

            min_h = min(h_gt, h_rec)
            min_w = min(w_gt, w_rec)

            gt_tensor = gt_tensor[:, :min_h, :min_w]
            rec_tensor = rec_tensor[:, :min_h, :min_w]

            # 計算數值
            val_psnr = psnr(gt_tensor, rec_tensor)
            val_msssim = ms_ssim(gt_tensor.unsqueeze(0), rec_tensor.unsqueeze(0), data_range=1.0).item()

            print(f"原始圖尺寸: {w_gt}x{h_gt}")
            print(f"重建圖尺寸: {w_rec}x{h_rec}")
            print(f"比對區域:   {min_w}x{min_h}")
            print("-" * 40)
            print(f"🚀 PSNR:    {val_psnr:.4f} dB")
            print(f"🚀 MS-SSIM: {val_msssim:.4f}")
            print("-" * 40)

        except Exception as e:
            print(f"計算 PSNR 時發生錯誤: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()