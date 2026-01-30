import argparse
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
import struct
import zlib  # 用於 CRC32
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==============================================================================
# 設定：匯入模型
# ==============================================================================
try:
    from gdn import TIC as SimpleConvStudentModel
except ImportError as e:
    print(f"錯誤: 找不到 gdn.py，或其依賴套件載入失敗。\n詳細錯誤: {e}")
    sys.exit(1)

# 嘗試匯入 tifffile (取代 rasterio，較容易在 ARM 平台安裝)
try:
    import tifffile
except ImportError:
    tifffile = None


# ==============================================================================
# Monkey Patching: 注入壓縮方法
# ==============================================================================
def compress_method(self, x):
    y = self.g_a(x)
    z = self.h_a(torch.abs(y))  # TIC 使用 abs(y) 作為 h_a 輸入
    
    # Force CPU
    self.entropy_bottleneck.cpu()
    self.h_s.cpu()
    self.gaussian_conditional.cpu()
    z = z.cpu()
    
    # =========================================================================
    # [優化修改區] 分開執行：一路產字串，一路產 z_hat
    # =========================================================================
    
    # 1. 產出 Bitstream (這是要寫入檔案的，不能省)
    z_strings = self.entropy_bottleneck.compress(z)
    
    # 2. 產出 z_hat (用數學運算直接算，跳過 decompress)
    # 必須取得 EntropyBottleneck 內部的 medians (因為 z 分佈中心不是 0)
    medians = self.entropy_bottleneck._get_medians().detach()
    
    # 處理維度擴展 (參考 CompressAI 原始碼邏輯)
    spatial_dims = len(z.size()) - 2
    medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
    medians = medians.expand(z.size(0), *([-1] * (spatial_dims + 1)))
    
    # 直接做量化 (Mode="dequantize" 會做 Round + 加回 medians)
    # 這樣得到的 z_hat 跟 decompress 出來的是一模一樣的，但速度快很多
    z_hat = self.entropy_bottleneck.quantize(z, "dequantize", medians)
    
    # =========================================================================

    # [關鍵修正] 
    # 1. 將 h_s 層轉為 Double (雙倍精度)
    self.h_s = self.h_s.double()
    
    # 2. 輸入的 z_hat 也要轉成 Double，才能跟 h_s 匹配
    z_hat_double = z_hat.double()
    
    # 3. 運算 (現在 Input 和 Weight 都是 Double 了)
    gaussian_params = self.h_s(z_hat_double)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)

    # 4. 轉回 Float
    scales_hat = scales_hat.float()
    means_hat = means_hat.float()

    # [V11] Cross-Platform Deterministic Quantization
    scales_clamped = scales_hat.clamp(0.5, 32.0)
    scale_indices = torch.floor((scales_clamped - 0.5) * 2 + 0.5).to(torch.int64).clamp(0, 63)
    scales_hat = 0.5 + scale_indices.float() * 0.5
    means_hat = torch.floor(means_hat * 10 + 0.5) / 10
    
    indexes = self.gaussian_conditional.build_indexes(scales_hat).to(torch.int32).contiguous()
    y_cpu = y.detach().cpu().contiguous()
    
    if torch.isnan(y_cpu).any():
        raise ValueError("FATAL: NaN detected in Latent y.")
        
    y_strings = self.gaussian_conditional.compress(y_cpu, indexes, means=means_hat.contiguous())
    
    return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

# 套用修改
SimpleConvStudentModel.compress = compress_method

from gdn import get_scale_table

# ==============================================================================
# 衛星通訊專用封包函式
# ==============================================================================

def save_satellite_packet(out_enc, output_path, img_id, row, col):
    """
    儲存封包: Header + Y_Strings + Z_Strings (AI Compressed) + CRC32
    Format (Little Endian <):
    [Magic:3] 'TIC'
    [ImgID:1]
    [Row:1]
    [Col:1]
    [H:2]
    [W:2]
    [LenY:4]
    [LenZ:4]
    [Payload Y]
    [Payload Z]
    [CRC32:4]
    """
    y_strings = out_enc["strings"][0]
    z_strings = out_enc["strings"][1]  # AI Compressed Bytes
    shape = out_enc["shape"]

    # 1. 準備 Payload Data
    # Y-String
    if isinstance(y_strings[0], (bytes, bytearray)):
         y_str_payload = b''.join(y_strings)
    else:
         y_str_payload = b''.join(y_strings[0])
         
    # Z-String (AI Compressed)
    if isinstance(z_strings[0], (bytes, bytearray)):
         z_str_payload = b''.join(z_strings)
    else:
         z_str_payload = b''.join(z_strings[0])

    # 2. 準備 Header (18 bytes)
    magic = b'TIC'
    h, w = shape
    len_y = len(y_str_payload)
    len_z = len(z_str_payload)

    header = struct.pack('<3sBBBHHII', 
                         magic, img_id, row, col, h, w, len_y, len_z)

    # 3. 組合完整封包 (Header + Payloads)
    packet_content = header + y_str_payload + z_str_payload

    # 4. 計算 CRC32
    crc = zlib.crc32(packet_content) & 0xffffffff
    footer = struct.pack('<I', crc)

    # 5. 寫入檔案 (含 fsync)
    with open(output_path, "wb") as f:
        f.write(packet_content)
        f.write(footer)
        f.flush()
        os.fsync(f.fileno())  # 確保寫入磁碟


# ==============================================================================
# 工具函式
# ==============================================================================
def read_image_patch(filepath: str, crop_box=None) -> torch.Tensor:
    """讀取影像區塊並轉為 Tensor"""
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.tif', '.tiff']:
        if tifffile is None: 
            raise RuntimeError("需安裝 tifffile (pip install tifffile)")
        
        # 使用 tifffile 讀取（保留原始 dtype 資訊）
        raw_data_original = tifffile.imread(filepath)
        original_dtype = raw_data_original.dtype
        raw_data = raw_data_original.astype(np.float32)
        
        # 處理維度順序：tifffile 讀出可能是 (H, W, C) 或 (C, H, W)
        if raw_data.ndim == 3:
            # 如果最後一維是通道數（較小），轉為 (C, H, W)
            if raw_data.shape[2] <= 4:  # 假設通道數不超過 4
                raw_data = np.transpose(raw_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        elif raw_data.ndim == 2:
            # 灰階圖，擴展為 (1, H, W)
            raw_data = np.expand_dims(raw_data, axis=0)
        
        # 裁切區塊
        if crop_box:
            left, upper, right, lower = crop_box
            raw_data = raw_data[:, upper:lower, left:right]
        
        if np.isnan(raw_data).any(): 
            raw_data = np.nan_to_num(raw_data)
        
        # 取前 3 通道作為 RGB
        rgb_data = raw_data[:3, :, :] if raw_data.shape[0] >= 3 else raw_data
        
        # 根據數據類型自動選擇縮放方式
        if original_dtype == np.uint8:
            # 已正規化的圖片 (0~255)
            clipped_data = np.clip(rgb_data, 0.0, 255.0)
            return torch.from_numpy(clipped_data / 255.0)
        else:
            # 衛星原始數據 (0~10000 或更大)
            SCALE = 10000.0
            clipped_data = np.clip(rgb_data, 0.0, 10000.0)
            return torch.from_numpy(clipped_data / SCALE)
    else:
        img = Image.open(filepath).convert("RGB")
        if crop_box:
            img = img.crop(crop_box)
        return transforms.ToTensor()(img)


@torch.no_grad()
def process_compress(model, x, output_path, img_id, row, col):
    # Padding 至 64 倍數
    h, w = x.size(2), x.size(3)
    p = 64
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top

    x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)

    # 執行壓縮
    out_enc = model.compress(x_padded)

    # 改用衛星封包格式儲存
    save_satellite_packet(out_enc, output_path, img_id, row, col)

    # 計算 bpp 供參考
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    total_bits = os.path.getsize(output_path) * 8.0  # 使用實際檔案大小(含header/crc)計算
    return total_bits / num_pixels


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
    model.load_state_dict(new_state_dict, strict=True)
    
    # ==========================================================================
    # 量化策略: 強制統一 Scale Table (Coarse Grid)
    # ==========================================================================
    # 使用粗刻度 0.5 ~ 32.0 (共 64 階)
    scale_table = torch.linspace(0.5, 32.0, 64)
    
    # 強制更新模型內的表和 CDF
    # model.gaussian_conditional.update_scale_table(scale_table, force=True)
    # model.update(force=True)
    
    # ==========================================================================
    # 強制統一 EntropyBottleneck CDFs & Medians
    # ==========================================================================
    try:
        from fixed_cdfs import FIXED_EB_CDF, FIXED_EB_OFFSET, FIXED_EB_LENGTH, FIXED_EB_MEDIANS
        from fixed_cdfs import FIXED_GC_CDF, FIXED_GC_OFFSET, FIXED_GC_LENGTH, FIXED_GC_SCALE_TABLE
        
        eb = model.entropy_bottleneck
        gc = model.gaussian_conditional
        device = eb._quantized_cdf.device
        
        # EntropyBottleneck CDFs
        eb._quantized_cdf.resize_(torch.tensor(FIXED_EB_CDF).shape).copy_(
            torch.tensor(FIXED_EB_CDF, device=device, dtype=torch.int32))
        eb._offset.resize_(torch.tensor(FIXED_EB_OFFSET).shape).copy_(
            torch.tensor(FIXED_EB_OFFSET, device=device, dtype=torch.int32))
        eb._cdf_length.resize_(torch.tensor(FIXED_EB_LENGTH).shape).copy_(
            torch.tensor(FIXED_EB_LENGTH, device=device, dtype=torch.int32))
        fixed_medians = torch.tensor(FIXED_EB_MEDIANS, device=device)
        eb.quantiles.data[:, 0, 1] = fixed_medians.squeeze()
        
        # GaussianConditional CDFs
        gc._quantized_cdf.resize_(torch.tensor(FIXED_GC_CDF).shape).copy_(
            torch.tensor(FIXED_GC_CDF, device=device, dtype=torch.int32))
        gc._offset.resize_(torch.tensor(FIXED_GC_OFFSET).shape).copy_(
            torch.tensor(FIXED_GC_OFFSET, device=device, dtype=torch.int32))
        gc._cdf_length.resize_(torch.tensor(FIXED_GC_LENGTH).shape).copy_(
            torch.tensor(FIXED_GC_LENGTH, device=device, dtype=torch.int32))
        gc.scale_table = torch.tensor(FIXED_GC_SCALE_TABLE, device=device)
            

    except ImportError:
        print("[WARNING] fixed_cdfs.py not found or incomplete!")
    except Exception as e:
        print(f"[WARNING] Failed to overwrite CDFs: {e}")
    # ==========================================================================

    return model.eval()

def compress_single_image(model, input_path, output_dir, img_id, device, PATCH_SIZE=256):
    """壓縮單張圖片的核心邏輯"""
    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(output_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n輸出資料夾: {save_dir}")

    # 取得影像尺寸
    if tifffile and os.path.splitext(input_path)[-1].lower() in ['.tif', '.tiff']:
        raw_data = tifffile.imread(input_path)
        if raw_data.ndim == 3:
            if raw_data.shape[2] <= 4:  # (H, W, C)
                img_h, img_w = raw_data.shape[0], raw_data.shape[1]
            else:  # (C, H, W)
                img_h, img_w = raw_data.shape[1], raw_data.shape[2]
        else:  # (H, W)
            img_h, img_w = raw_data.shape[0], raw_data.shape[1]
        del raw_data  # 釋放記憶體
    else:
        with Image.open(input_path) as img:
            img_w, img_h = img.size

    num_cols = img_w // PATCH_SIZE
    num_rows = img_h // PATCH_SIZE

    # 邊界處理：如果有餘數，需要多加一行/列
    if img_w % PATCH_SIZE != 0: num_cols += 1
    if img_h % PATCH_SIZE != 0: num_rows += 1

    total_patches = num_cols * num_rows

    print(f"原始影像: {img_w}x{img_h}")
    print(f"分割模式（segmentation）: {num_rows}x{num_cols} (共 {total_patches} 個區塊)")
    print(f"Image ID: {img_id}")

    start_time = time.time()
    total_bpp = 0

    # 開始切割與壓縮
    count = 0
    for row in range(num_rows):
        for col in range(num_cols):
            count += 1
            left = col * PATCH_SIZE
            upper = row * PATCH_SIZE
            right = left + PATCH_SIZE
            lower = upper + PATCH_SIZE
            box = (left, upper, right, lower)

            try:
                x = read_image_patch(input_path, box).unsqueeze(0).to(device)
            except Exception as e:
                print(f"\n[ERROR] 讀取 row={row}, col={col} 失敗: {e}")
                continue

            # 確保輸入是 256x256 (若在邊緣可能變小)
            if x.size(2) < PATCH_SIZE or x.size(3) < PATCH_SIZE:
                x = F.pad(x, (0, PATCH_SIZE - x.size(3), 0, PATCH_SIZE - x.size(2)))

            bin_filename = f"{base_name}_row{row}_col{col}.bin"
            bin_path = os.path.join(save_dir, bin_filename)

            bpp = process_compress(model, x, bin_path, img_id, row, col)
            total_bpp += bpp

            print(f"壓縮進度: {count}/{total_patches} | Row:{row} Col:{col} | bpp: {bpp:.4f}", end='\r')

    avg_bpp = total_bpp / total_patches if total_patches > 0 else 0
    elapsed = time.time() - start_time
    print(f"\n壓縮完成! 平均 bpp: {avg_bpp:.4f} | 耗時: {elapsed:.2f} 秒")
    return avg_bpp, elapsed

# ==============================================================================
# 主程式
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Satellite Image Compression Tool (with CRC32)")
    parser.add_argument("input_path", type=str, nargs='+', 
                        help="Path to input image(s) or directory containing images")
    parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to .pth model")
    parser.add_argument("-o", "--output_dir", type=str, default="output_satellite",
                        help="Output directory")
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--id", type=int, default=1, help="Image ID (0-255)")
    args = parser.parse_args()

    # 初始化設定
    PATCH_SIZE = 256
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 載入模型 (只載入一次!)
    model = load_checkpoint(args.checkpoint).to(device)

    # 收集所有要處理的圖片
    image_files = []
    for path in args.input_path:
        if os.path.isdir(path):
            # 如果是資料夾，找出所有圖片檔
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.PNG', '*.JPG', '*.TIF', '*.TIFF']:
                image_files.extend(glob.glob(os.path.join(path, ext)))
        elif os.path.isfile(path):
            image_files.append(path)
        else:
            print(f"[WARNING] 找不到: {path}")

    if not image_files:
        print("錯誤: 沒有找到任何圖片檔案")
        sys.exit(1)

    # 排序以確保順序一致
    image_files = sorted(set(image_files))
    
    print(f"=== 準備處理 {len(image_files)} 張圖片 ===")

    
    total_start = time.time()
    all_bpp = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*50}")
        print(f"[{idx}/{len(image_files)}] 處理: {os.path.basename(img_path)}")
        print('='*50)
        
        try:
            avg_bpp, _ = compress_single_image(
                model, img_path, args.output_dir, args.id, device, PATCH_SIZE
            )
            all_bpp.append(avg_bpp)
        except Exception as e:
            print(f"[ERROR] 處理失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 總結
    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"=== 全部完成 ===")
    print(f"處理圖片數: {len(all_bpp)}/{len(image_files)}")
    if all_bpp:
        print(f"整體平均 bpp: {sum(all_bpp)/len(all_bpp):.4f}")
    print(f"總耗時: {total_elapsed:.2f} 秒")
    print('='*50)


if __name__ == "__main__":
    main()