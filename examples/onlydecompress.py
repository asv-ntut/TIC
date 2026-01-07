import argparse
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import math
import struct
import zlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ==============================================================================
# 設定：匯入模型
# ==============================================================================
try:
    from compressai.models.tic import TIC
except ImportError as e:
    print(f"錯誤: 找不到 conv2.py，或其依賴套件載入失敗。\n詳細錯誤: {e}")
    sys.exit(1)

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    rasterio = None

try:
    from pytorch_msssim import ms_ssim

    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


# ==============================================================================
# Monkey Patching: 注入解壓縮方法
# ==============================================================================
def decompress_method(self, strings, shape):
    assert isinstance(strings, list) and len(strings) == 2
    
    # 1. Force CPU
    self.entropy_bottleneck.cpu()
    self.h_s.cpu()
    self.gaussian_conditional.cpu()
    
    # Z-String Padding (for robustness)
    z_str_list = strings[1]
    if isinstance(z_str_list, list) and isinstance(z_str_list[0], (bytes, bytearray)):
         z_str_list[0] = z_str_list[0] + b'\x00' * 4096 
            
    # 2. Decompress Z
    z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    if z_hat.device.type != 'cpu': z_hat = z_hat.cpu()
    
    # 3. TIC model: h_s outputs only scales (M channels), no means
    scales_hat = self.h_s(z_hat.float())
    
    # Build indexes for entropy coding
    indexes = self.gaussian_conditional.build_indexes(scales_hat)

    # Y-String Padding (for robustness)
    y_str_list = strings[0]
    if isinstance(y_str_list, list) and isinstance(y_str_list[0], (bytes, bytearray)):
        y_str_list[0] = y_str_list[0] + b'\x00' * 4096
            
    # 4. Decompress Y (no means for TIC model)
    try:
        torch.set_num_threads(1)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes.cpu())
    except Exception as e:
        print(f"[ERROR] Block Corruption: {e}")
        # Fallback: create zero tensor
        y_hat = torch.zeros_like(scales_hat)

    # Sanitizer: clamp extreme values
    if torch.isnan(y_hat).any() or y_hat.abs().max() > 100.0:
        mask_bad = (y_hat.abs() > 100.0) | torch.isnan(y_hat)
        y_hat[mask_bad] = 0.0
        
    device = next(self.g_s.parameters()).device
    y_hat = y_hat.to(device).float()
    
    x_hat = self.g_s(y_hat).clamp_(0, 1)
    return {"x_hat": x_hat}

TIC.decompress = decompress_method


def parse_payload_bytes(payload_data):
    """
    解析 Payload 的二進位內容，還原成 strings 和 shape
    格式: Shape(2+2) + LenZ(4) + Z + LenY(4) + Y
    """
    cursor = 0

    # 讀取 shape
    h = int.from_bytes(payload_data[cursor:cursor + 2], 'little')
    cursor += 2
    w = int.from_bytes(payload_data[cursor:cursor + 2], 'little')
    cursor += 2
    shape = (h, w)

    # 讀取 z_string
    len_z = int.from_bytes(payload_data[cursor:cursor + 4], 'little')
    cursor += 4
    z_str = payload_data[cursor:cursor + len_z]
    cursor += len_z

    # 讀取 y_string
    len_y = int.from_bytes(payload_data[cursor:cursor + 4], 'little')
    cursor += 4
    y_str = payload_data[cursor:cursor + len_y]

    return {"strings": [[y_str], [z_str]], "shape": shape}


from compressai.models.tic import get_scale_table

def load_satellite_packet(bin_path):
    """
    讀取封包: Header + Y_Strings + Z_Strings (AI Compressed) + CRC32
    Format (Little Endian <):
    [Magic:3][ID:1][Row:1][Col:1][H:2][W:2][LenY:4][LenZ:4] ... [CRC:4]
    """
    import numpy as np
    try:
        with open(bin_path, "rb") as f:
            data = f.read()

        # 最小長度檢查: Header(18) + CRC(4) = 22 bytes
        if len(data) < 22: 
            print(f"[Error] File too small: {os.path.basename(bin_path)}")
            return None

        # 1. CRC Check (最後 4 bytes)
        received_crc = struct.unpack('<I', data[-4:])[0]
        content = data[:-4]
        calc_crc = zlib.crc32(content) & 0xffffffff

        if received_crc != calc_crc:
            print(f"[Corrupt] CRC Fail: {os.path.basename(bin_path)}")
            return None

        # 2. Parse Header (18 bytes)
        # Magic(3) + ID(1) + Row(1) + Col(1) + H(2) + W(2) + LenY(4) + LenZ(4)
        header_size = 18
        header_data = data[:header_size]
        
        magic, img_id, row, col, h, w, len_y, len_z = struct.unpack('<3sBBBHHII', header_data)
        
        # Magic Check
        if magic != b'TIC':
            print(f"[Error] Invalid Magic: {magic}")
            return None
            
        # 3. Parse Payloads
        # Payload starts at 18
        cursor = header_size
        
        if len(data) < cursor + len_y + len_z + 4:
            print(f"[Error] Incomplete Payload")
            return None
            
        y_str = data[cursor : cursor + len_y]
        cursor += len_y
        
        z_str = data[cursor : cursor + len_z]
        
        # 回傳原始字串，讓 decompress_method 去解碼
        return {
            "row": row, "col": col, "img_id": img_id,
            "strings": [[y_str], [z_str]], 
            "shape": (h, w)
        }

    except Exception as e:
        print(f"[Read Error] {bin_path}: {e}")
        return None


@torch.no_grad()
def process_decompress_packet(model, packet_data):
    out_dec = model.decompress(packet_data["strings"], packet_data["shape"])
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
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse) if mse > 0 else float('inf')


def read_original_image(filepath: str) -> torch.Tensor:
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.tif', '.tiff']:
        if rasterio is None: raise RuntimeError("需安裝 rasterio 才能讀取 TIF")
        with rasterio.open(filepath) as src:
            raw_data_original = src.read()
            original_dtype = raw_data_original.dtype
            raw_data = raw_data_original.astype(np.float32)
        if np.isnan(raw_data).any(): raw_data = np.nan_to_num(raw_data)
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

    model = TIC(N=N, M=M)
    model.load_state_dict(new_state_dict, strict=True)
    
    # ==========================================================================
    # 量化策略: 強制統一 Scale Table (Coarse Grid)
    # ==========================================================================
    # 使用粗刻度 0.5 ~ 32.0 (共 64 階)
    scale_table = torch.linspace(0.5, 32.0, 64)
    
    # 強制更新模型內的表和 CDF
    model.gaussian_conditional.update_scale_table(scale_table, force=True)
    model.update(force=True)
    
    # ==========================================================================
    # 強制統一 EntropyBottleneck & GaussianConditional CDFs
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

    return model.eval()


# ==============================================================================
# 主程式
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Satellite Image Decompression (CRC32 Verified)")
    parser.add_argument("bin_dir", type=str, help="Directory containing .bin files")
    parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--original", type=str, default=None, help="Original image for PSNR")
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--target_id", type=int, default=None, help="Only reconstruct packets with this Image ID")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    PATCH_SIZE = 256

    # 1. 搜尋所有 .bin 檔案 (不再依賴檔名格式)
    bin_files = glob.glob(os.path.join(args.bin_dir, "*.bin"))
    if not bin_files:
        print(f"在 {args.bin_dir} 找不到任何 .bin 檔案")
        sys.exit(1)

    print(f"找到 {len(bin_files)} 個檔案，開始驗證並解壓縮...")

    # 2. 載入模型
    model = load_checkpoint(args.checkpoint).to(device)
    # [CRITICAL] DO NOT call model.update() here! It will overwrite the fixed_cdfs loaded in load_checkpoint
    # model.update(force=True)  # REMOVED - This was causing z_hat mismatch across platforms

    # 3. 預掃描：決定畫布大小
    # 因為沒有檔名告訴我們大小，我們必須掃描有效封包的 header 來找出最大的 row/col
    max_row, max_col = 0, 0
    valid_packets = []

    for f in bin_files:
        # 讀取並驗證封包
        packet = load_satellite_packet(f)

        if packet is None:
            # 驗證失敗，直接跳過 (這就是掉包/壞包處理)
            continue

        # 如果指定了 ID，過濾掉不符的
        if args.target_id is not None and packet['img_id'] != args.target_id:
            continue

        max_row = max(max_row, packet['row'])
        max_col = max(max_col, packet['col'])
        valid_packets.append(packet)

    if not valid_packets:
        print("沒有找到有效的封包 (可能全部損毀或 ID 不符)。")
        sys.exit(1)

    # 計算畫布大小
    canvas_w = (max_col + 1) * PATCH_SIZE
    canvas_h = (max_row + 1) * PATCH_SIZE
    print(f"最大矩陣: Row {max_row}, Col {max_col} | 原始大小: {canvas_w}x{canvas_h}")
    print(f"有效封包數: {len(valid_packets)} / {len(bin_files)} (遺失/損毀: {len(bin_files) - len(valid_packets)})")

    # 建立黑色畫布 (RGB 0,0,0)
    full_recon_img = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))

    # 4. 解壓縮並拼貼
    count = 0
    for packet in valid_packets:
        count += 1
        r, c = packet['row'], packet['col']
        print(f"還原區塊: ({r}, {c}) - ID:{packet['img_id']} ({count}/{len(valid_packets)})", end='\r')

        try:
            x_hat = process_decompress_packet(model, packet)
            rec_tensor = x_hat.squeeze().cpu().clamp(0, 1)
            rec_patch_pil = transforms.ToPILImage()(rec_tensor)

            left = c * PATCH_SIZE
            upper = r * PATCH_SIZE
            full_recon_img.paste(rec_patch_pil, (left, upper))
        except Exception as e:
            print(f"\n[解碼失敗] 區塊 ({r},{c}) 資料異常: {e}")

    print("\n還原完成...")

    # ==========================================================================
    # 5. 計算 PSNR (使用原始重建圖，不含亮度調整)
    # ==========================================================================
    if args.original:
        print("-" * 40)
        if not os.path.exists(args.original):
            print(f"錯誤: 找不到原始圖片 {args.original}")
            return

        try:
            gt_tensor = read_original_image(args.original)
            rec_tensor = transforms.ToTensor()(full_recon_img)

            # 對齊尺寸
            h_gt, w_gt = gt_tensor.shape[1], gt_tensor.shape[2]
            h_rec, w_rec = rec_tensor.shape[1], rec_tensor.shape[2]
            min_h = min(h_gt, h_rec)
            min_w = min(w_gt, w_rec)

            gt_tensor = gt_tensor[:, :min_h, :min_w]
            rec_tensor = rec_tensor[:, :min_h, :min_w]

            val_psnr = psnr(gt_tensor, rec_tensor)
            val_msssim = 0.0
            if HAS_MSSSIM:
                val_msssim = ms_ssim(gt_tensor.unsqueeze(0), rec_tensor.unsqueeze(0), data_range=1.0).item()

            print(f"PSNR:    {val_psnr:.4f} dB")
            if HAS_MSSSIM:
                print(f"MS-SSIM: {val_msssim:.4f}")
            print("-" * 40)

        except Exception as e:
            print(f"計算 PSNR 時發生錯誤: {e}")

    # ==========================================================================
    # 6. 調亮並儲存影像
    # ==========================================================================
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(full_recon_img)
    full_recon_img = enhancer.enhance(1.8)  # 亮度調亮 80%
    
    output_path = os.path.join(args.bin_dir, "RECONSTRUCTED_SATELLITE.png")
    full_recon_img.save(output_path)
    print(f"結果已儲存: {output_path}")


if __name__ == "__main__":
    main()