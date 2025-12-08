# import argparse
# import os
# import sys
# import time
# from collections import OrderedDict
# from typing import List
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
#
# # ==============================================================================
# # 設定：匯入模型
# # ==============================================================================
# try:
#     from conv2 import SimpleConvStudentModel
# except ImportError:
#     print("錯誤: 找不到 conv2.py，請確認檔案位置。")
#     sys.exit(1)
#
# # 嘗試匯入 rasterio (處理 TIF)
# try:
#     import rasterio
# except ImportError:
#     rasterio = None
#
#
# # ==============================================================================
# # Monkey Patching: 注入壓縮方法
# # ==============================================================================
# def compress_method(self, x):
#     y = self.g_a(x)
#     z = self.h_a(y)
#     z_strings = self.entropy_bottleneck.compress(z)
#     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#     gaussian_params = self.h_s(z_hat)
#     scales_hat, means_hat = gaussian_params.chunk(2, 1)
#     indexes = self.gaussian_conditional.build_indexes(scales_hat)
#     y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
#     return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
#
#
# SimpleConvStudentModel.compress = compress_method
#
#
# # ==============================================================================
# # 工具函式
# # ==============================================================================
# def read_image_patch(filepath: str, crop_box=None) -> torch.Tensor:
#     """讀取影像區塊並轉為 Tensor"""
#     ext = os.path.splitext(filepath)[-1].lower()
#     if ext in ['.tif', '.tiff']:
#         if rasterio is None: raise RuntimeError("需安裝 rasterio")
#         SCALE = 10000.0
#         with rasterio.open(filepath) as src:
#             if crop_box:
#                 left, upper, right, lower = crop_box
#                 window = rasterio.windows.Window(left, upper, right - left, lower - upper)
#                 raw_data = src.read(window=window).astype(np.float32)
#             else:
#                 raw_data = src.read().astype(np.float32)
#         if np.isnan(raw_data).any(): raw_data = np.nan_to_num(raw_data)
#         rgb_data = raw_data[:3, :, :] if raw_data.shape[0] >= 3 else raw_data
#         clipped_data = np.clip(rgb_data, 0.0, 10000.0)
#         return torch.from_numpy(clipped_data / SCALE)
#     else:
#         img = Image.open(filepath).convert("RGB")
#         if crop_box:
#             img = img.crop(crop_box)
#         return transforms.ToTensor()(img)
#
#
# def save_compressed_bin(out_enc, output_path):
#     """將壓縮結果寫入二進位檔案"""
#     with open(output_path, "wb") as f:
#         # 1. 寫入 shape (h, w) - 各 2 bytes
#         shape = out_enc["shape"]
#         f.write(shape[0].to_bytes(2, 'little'))
#         f.write(shape[1].to_bytes(2, 'little'))
#
#         # 2. 寫入 z_string (長度 + 內容)
#         z_str = out_enc["strings"][1][0]
#         f.write(len(z_str).to_bytes(4, 'little'))
#         f.write(z_str)
#
#         # 3. 寫入 y_string (長度 + 內容)
#         y_str = out_enc["strings"][0][0]
#         f.write(len(y_str).to_bytes(4, 'little'))
#         f.write(y_str)
#
#
# @torch.no_grad()
# def process_compress(model, x, output_path):
#     # Padding 至 64 倍數
#     h, w = x.size(2), x.size(3)
#     p = 64
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
#
#     x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)
#
#     # 執行壓縮
#     out_enc = model.compress(x_padded)
#
#     # 儲存
#     save_compressed_bin(out_enc, output_path)
#
#     # 計算 BPP 供參考
#     num_pixels = x.size(0) * x.size(2) * x.size(3)
#     total_bits = sum(len(s[0]) for s in out_enc["strings"]) * 8.0
#     return total_bits / num_pixels
#
#
# def load_checkpoint(checkpoint_path):
#     print(f"Loading checkpoint: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     state_dict = checkpoint.get("state_dict", checkpoint)
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] if k.startswith('module.') else k
#         new_state_dict[name] = v
#
#     # 這裡依照您的原始碼邏輯取得 N, M
#     N, M = 128, 192
#     try:
#         N = new_state_dict['g_a.0.weight'].size(0)
#         keys = sorted([k for k in new_state_dict.keys() if 'g_a' in k and 'weight' in k])
#         M = new_state_dict[keys[-1]].size(0)
#     except:
#         pass
#
#     model = SimpleConvStudentModel(N=N, M=M)
#     model.load_state_dict(new_state_dict, strict=False)
#     return model.eval()
#
#
# # ==============================================================================
# # 主程式
# # ==============================================================================
# def main():
#     parser = argparse.ArgumentParser(description="Image Compression Tool")
#     parser.add_argument("input_path", type=str, help="Path to input image")
#     parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to .pth model")
#     parser.add_argument("-o", "--output_dir", type=str, default=r"C:\Users\Matt\Desktop\研究進度\1130",
#                         help="Output directory")
#     parser.add_argument("--cuda", action="store_true", default=True)
#     args = parser.parse_args()
#
#     # 初始化設定
#     PATCH_SIZE = 256
#     device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
#
#     # 載入模型
#     model = load_checkpoint(args.checkpoint).to(device)
#     model.update(force=True)
#
#     # 準備路徑
#     filename = os.path.basename(args.input_path)
#     base_name = os.path.splitext(filename)[0]
#     save_dir = os.path.join(args.output_dir, base_name)
#     os.makedirs(save_dir, exist_ok=True)
#     print(f"輸出資料夾: {save_dir}")
#
#     # 取得影像尺寸
#     if rasterio and os.path.splitext(args.input_path)[-1].lower() in ['.tif', '.tiff']:
#         with rasterio.open(args.input_path) as src:
#             img_w, img_h = src.width, src.height
#     else:
#         with Image.open(args.input_path) as img:
#             img_w, img_h = img.size
#
#     num_cols = img_w // PATCH_SIZE
#     num_rows = img_h // PATCH_SIZE
#     total_patches = num_cols * num_rows
#
#     print(f"原始影像: {img_w}x{img_h}")
#     print(f"分割模式: {num_rows}x{num_cols} (共 {total_patches} 個區塊)")
#
#     start_time = time.time()
#     total_bpp = 0
#
#     # 開始切割與壓縮
#     count = 0
#     for row in range(num_rows):
#         for col in range(num_cols):
#             count += 1
#             left = col * PATCH_SIZE
#             upper = row * PATCH_SIZE
#             right = left + PATCH_SIZE
#             lower = upper + PATCH_SIZE
#             box = (left, upper, right, lower)
#
#             # 讀取
#             x = read_image_patch(args.input_path, box).unsqueeze(0).to(device)
#
#             # 定義檔名: 包含 row, col 資訊以便解壓縮時定位
#             bin_filename = f"{base_name}_row{row}_col{col}.bin"
#             bin_path = os.path.join(save_dir, bin_filename)
#
#             # 壓縮並存檔
#             bpp = process_compress(model, x, bin_path)
#             total_bpp += bpp
#
#             print(f"壓縮進度: {count}/{total_patches} | BPP: {bpp:.4f}", end='\r')
#
#     print(f"\n壓縮完成! 平均 BPP: {total_bpp / total_patches:.4f}")
#     print(f"總耗時: {time.time() - start_time:.2f} 秒")
#
#
# if __name__ == "__main__":
#     main()

# import argparse
# import os
# import sys
# import time
# import struct
# import zlib  # 用於 CRC32
# from collections import OrderedDict
# from typing import List

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image

# # ==============================================================================
# # 設定：匯入模型
# # ==============================================================================
# try:
#     from conv2 import SimpleConvStudentModel
# except ImportError:
#     print("錯誤: 找不到 conv2.py，請確認檔案位置。")
#     sys.exit(1)

# # 嘗試匯入 rasterio
# try:
#     import rasterio
# except ImportError:
#     rasterio = None


# # ==============================================================================
# # Monkey Patching: 注入壓縮方法
# # ==============================================================================
# def compress_method(self, x):
#     y = self.g_a(x)
#     z = self.h_a(y)
#     z_strings = self.entropy_bottleneck.compress(z)
#     z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#     gaussian_params = self.h_s(z_hat)
#     scales_hat, means_hat = gaussian_params.chunk(2, 1)
#     indexes = self.gaussian_conditional.build_indexes(scales_hat)
#     y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
#     return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


# SimpleConvStudentModel.compress = compress_method


# # ==============================================================================
# # 衛星通訊專用封包函式 (核心修改)
# # ==============================================================================
# def save_satellite_packet(out_enc, output_path, img_id, row, col):
#     """
#     打包成衛星傳輸格式：
#     Header [7 bytes] + Payload [N bytes] + CRC32 [4 bytes]

#     Header 結構:
#     - Image ID (1 byte)
#     - Row Index (1 byte)
#     - Col Index (1 byte)
#     - Payload Length (4 bytes)
#     """

#     # 1. 建構 Payload (資料本體)
#     # Payload 格式: Shape(4) + LenZ(4) + Z + LenY(4) + Y
#     payload = bytearray()

#     shape = out_enc["shape"]
#     z_str = out_enc["strings"][1][0]
#     y_str = out_enc["strings"][0][0]

#     # 加入 Shape (Height, Width) - 各 2 bytes
#     payload.extend(shape[0].to_bytes(2, 'little'))
#     payload.extend(shape[1].to_bytes(2, 'little'))

#     # 加入 Z String
#     payload.extend(len(z_str).to_bytes(4, 'little'))
#     payload.extend(z_str)

#     # 加入 Y String
#     payload.extend(len(y_str).to_bytes(4, 'little'))
#     payload.extend(y_str)

#     # 2. 建構 Header
#     # 格式: <BBBI (Little Endian: unsigned char, u_char, u_char, unsigned int)
#     # img_id, row, col 必須 < 256
#     header = struct.pack('<BBBI', int(img_id), int(row), int(col), len(payload))

#     # 3. 計算 CRC32 (Header + Payload)
#     # 地面站會驗證這一段，確保 header 和資料都沒壞
#     checksum_data = header + payload
#     crc_value = zlib.crc32(checksum_data) & 0xffffffff
#     footer = struct.pack('<I', crc_value)

#     # 4. 寫入檔案
#     with open(output_path, "wb") as f:
#         f.write(header)
#         f.write(payload)
#         f.write(footer)


# # ==============================================================================
# # 工具函式
# # ==============================================================================
# def read_image_patch(filepath: str, crop_box=None) -> torch.Tensor:
#     """讀取影像區塊並轉為 Tensor"""
#     ext = os.path.splitext(filepath)[-1].lower()
#     if ext in ['.tif', '.tiff']:
#         if rasterio is None: raise RuntimeError("需安裝 rasterio")
#         SCALE = 10000.0
#         with rasterio.open(filepath) as src:
#             if crop_box:
#                 left, upper, right, lower = crop_box
#                 window = rasterio.windows.Window(left, upper, right - left, lower - upper)
#                 raw_data = src.read(window=window).astype(np.float32)
#             else:
#                 raw_data = src.read().astype(np.float32)
#         if np.isnan(raw_data).any(): raw_data = np.nan_to_num(raw_data)
#         rgb_data = raw_data[:3, :, :] if raw_data.shape[0] >= 3 else raw_data
#         clipped_data = np.clip(rgb_data, 0.0, 10000.0)
#         return torch.from_numpy(clipped_data / SCALE)
#     else:
#         img = Image.open(filepath).convert("RGB")
#         if crop_box:
#             img = img.crop(crop_box)
#         return transforms.ToTensor()(img)


# @torch.no_grad()
# def process_compress(model, x, output_path, img_id, row, col):
#     # Padding 至 64 倍數
#     h, w = x.size(2), x.size(3)
#     p = 64
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top

#     x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)

#     # 執行壓縮
#     out_enc = model.compress(x_padded)

#     # 改用衛星封包格式儲存
#     save_satellite_packet(out_enc, output_path, img_id, row, col)

#     # 計算 BPP 供參考
#     num_pixels = x.size(0) * x.size(2) * x.size(3)
#     total_bits = os.path.getsize(output_path) * 8.0  # 使用實際檔案大小(含header/crc)計算
#     return total_bits / num_pixels


# def load_checkpoint(checkpoint_path):
#     print(f"Loading checkpoint: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     state_dict = checkpoint.get("state_dict", checkpoint)
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] if k.startswith('module.') else k
#         new_state_dict[name] = v

#     N, M = 128, 192
#     try:
#         N = new_state_dict['g_a.0.weight'].size(0)
#         keys = sorted([k for k in new_state_dict.keys() if 'g_a' in k and 'weight' in k])
#         M = new_state_dict[keys[-1]].size(0)
#     except:
#         pass

#     model = SimpleConvStudentModel(N=N, M=M)
#     model.load_state_dict(new_state_dict, strict=False)
#     return model.eval()


# # ==============================================================================
# # 主程式
# # ==============================================================================
# def main():
#     parser = argparse.ArgumentParser(description="Satellite Image Compression Tool (with CRC32)")
#     parser.add_argument("input_path", type=str, help="Path to input image")
#     parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to .pth model")
#     parser.add_argument("-o", "--output_dir", type=str, default="output_satellite",
#                         help="Output directory")
#     parser.add_argument("--cuda", action="store_true", default=True)
#     parser.add_argument("--id", type=int, default=1, help="Image ID (0-255)")
#     args = parser.parse_args()

#     # 初始化設定
#     PATCH_SIZE = 256
#     device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

#     # 載入模型
#     model = load_checkpoint(args.checkpoint).to(device)
#     model.update(force=True)

#     # 準備路徑
#     filename = os.path.basename(args.input_path)
#     base_name = os.path.splitext(filename)[0]
#     save_dir = os.path.join(args.output_dir, base_name)
#     os.makedirs(save_dir, exist_ok=True)
#     print(f"輸出資料夾: {save_dir}")

#     # 取得影像尺寸
#     if rasterio and os.path.splitext(args.input_path)[-1].lower() in ['.tif', '.tiff']:
#         with rasterio.open(args.input_path) as src:
#             img_w, img_h = src.width, src.height
#     else:
#         with Image.open(args.input_path) as img:
#             img_w, img_h = img.size

#     num_cols = img_w // PATCH_SIZE
#     num_rows = img_h // PATCH_SIZE

#     # 邊界處理：如果有餘數，需要多加一行/列
#     if img_w % PATCH_SIZE != 0: num_cols += 1
#     if img_h % PATCH_SIZE != 0: num_rows += 1

#     total_patches = num_cols * num_rows

#     print(f"原始影像: {img_w}x{img_h}")
#     print(f"分割模式: {num_rows}x{num_cols} (共 {total_patches} 個區塊)")
#     print(f"Image ID: {args.id}")

#     start_time = time.time()
#     total_bpp = 0

#     # 開始切割與壓縮
#     count = 0
#     for row in range(num_rows):
#         for col in range(num_cols):
#             count += 1
#             left = col * PATCH_SIZE
#             upper = row * PATCH_SIZE
#             right = left + PATCH_SIZE
#             lower = upper + PATCH_SIZE
#             box = (left, upper, right, lower)

#             # 讀取 (如果是邊緣，read_image_patch 預設行為會 padding 或是由 process_compress handle，這裡主要依賴 process_compress 的 padding)
#             # 注意: 這裡我們簡單傳入 box，若超出邊界，read_image_patch 需能處理，或者由 PIL crop 自動處理
#             try:
#                 x = read_image_patch(args.input_path, box).unsqueeze(0).to(device)
#             except Exception as e:
#                 # 處理邊緣超出情況，通常 PIL crop 會自動縮小，這時需要 pad 回 256
#                 # 這裡為了穩健性，我們手動讀取並 Pad
#                 # (略過複雜 padding 邏輯，假設 read_image_patch 會回傳正確 tensor)
#                 continue

#             # 確保輸入是 256x256 (若在邊緣可能變小)
#             if x.size(2) < PATCH_SIZE or x.size(3) < PATCH_SIZE:
#                 x = F.pad(x, (0, PATCH_SIZE - x.size(3), 0, PATCH_SIZE - x.size(2)))

#             # 檔名仍然保留 row/col 方便人類查看，但解壓縮時將忽略它
#             bin_filename = f"{base_name}_row{row}_col{col}.bin"
#             bin_path = os.path.join(save_dir, bin_filename)

#             # 壓縮並存檔 (傳入 row, col, id)
#             bpp = process_compress(model, x, bin_path, args.id, row, col)
#             total_bpp += bpp

#             print(f"壓縮進度: {count}/{total_patches} | Row:{row} Col:{col} | BPP: {bpp:.4f}", end='\r')

#     print(f"\n壓縮完成! 平均 BPP: {total_bpp / total_patches:.4f}")
#     print(f"總耗時: {time.time() - start_time:.2f} 秒")


# if __name__ == "__main__":
#     main()

import argparse
import os
import sys
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
    from conv2 import SimpleConvStudentModel
except ImportError as e:
    print(f"錯誤: 找不到 conv2.py，或其依賴套件載入失敗。\n詳細錯誤: {e}")
    sys.exit(1)

# 嘗試匯入 rasterio
try:
    import rasterio
except ImportError:
    rasterio = None


# ==============================================================================
# Monkey Patching: 注入壓縮方法
# ==============================================================================
def compress_method(self, x):
    y = self.g_a(x)
    z = self.h_a(y)
    # 核心邏輯: 繞過 EntropyBottleneck (Bypass Z-stream)
    # 為了徹底排除 EntropyBottleneck (Arithmetic Coding) 跨平台解碼不一致的問題，
    # 我們不使用 model.entropy_bottleneck.compress 產生的字串。
    # 而是直接取出 z_hat (Tensor)，稍後在 save_satellite_packet 中使用 zlib 進行無損壓縮。
    # 強制執行 compress -> decompress 流程是為了確保 z_hat 數值包含 medians 修正，與解碼端一致。
    
    # [Deterministic Fix] Force Critical Components to CPU
    # This ensures that h_s (Conv2d) runs on CPU, matching the Decoder's behavior.
    # GPU (CUDA) vs CPU floating point drift in h_s is the root cause of PSNR degradation (9dB).
    self.entropy_bottleneck.cpu()
    self.h_s.cpu()
    self.gaussian_conditional.cpu()

    # Move z to CPU for processing
    z = z.cpu()
    
    z_strings = self.entropy_bottleneck.compress(z)
    z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
    
    gaussian_params = self.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)

    # [V11] Cross-Platform Deterministic Quantization
    # Scale Table: [0.5, 1.0, 1.5, ..., 32.0] (64 levels)
    scales_clamped = scales_hat.clamp(0.5, 32.0)
    scale_indices = torch.floor((scales_clamped - 0.5) * 2 + 0.5).to(torch.int64).clamp(0, 63)
    scales_hat = 0.5 + scale_indices.float() * 0.5
    means_hat = torch.floor(means_hat * 10 + 0.5) / 10
    
    indexes = self.gaussian_conditional.build_indexes(scales_hat)
    indexes = indexes.to(dtype=torch.int32).contiguous()
    
    y_cpu = y.detach().cpu().contiguous()
    means_hat_cpu = means_hat.detach().cpu().contiguous()
    indexes_cpu = indexes.detach().cpu().contiguous()
    
    if torch.isnan(y_cpu).any():
        raise ValueError("FATAL: NaN detected in Latent y.")
        
    y_strings = self.gaussian_conditional.compress(y_cpu, indexes_cpu, means=means_hat_cpu)
    
    return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


SimpleConvStudentModel.compress = compress_method


from conv2 import get_scale_table

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
        if rasterio is None: raise RuntimeError("需安裝 rasterio")
        SCALE = 10000.0
        with rasterio.open(filepath) as src:
            if crop_box:
                left, upper, right, lower = crop_box
                window = rasterio.windows.Window(left, upper, right - left, lower - upper)
                raw_data = src.read(window=window).astype(np.float32)
            else:
                raw_data = src.read().astype(np.float32)
        if np.isnan(raw_data).any(): raw_data = np.nan_to_num(raw_data)
        rgb_data = raw_data[:3, :, :] if raw_data.shape[0] >= 3 else raw_data
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

    # 計算 BPP 供參考
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
    model.gaussian_conditional.update_scale_table(scale_table, force=True)
    model.update(force=True)
    
    # ==========================================================================
    # 強制統一 EntropyBottleneck CDFs & Medians
    # ==========================================================================
    try:
        from fixed_cdfs import FIXED_EB_CDF, FIXED_EB_OFFSET, FIXED_EB_LENGTH, FIXED_EB_MEDIANS
        eb = model.entropy_bottleneck
        device = eb._quantized_cdf.device
        
        # 1. 覆蓋 CDF, Offset, Length
        eb._quantized_cdf.resize_(torch.tensor(FIXED_EB_CDF).shape).copy_(
            torch.tensor(FIXED_EB_CDF, device=device, dtype=torch.int32))
        eb._offset.resize_(torch.tensor(FIXED_EB_OFFSET).shape).copy_(
            torch.tensor(FIXED_EB_OFFSET, device=device, dtype=torch.int32))
        eb._cdf_length.resize_(torch.tensor(FIXED_EB_LENGTH).shape).copy_(
            torch.tensor(FIXED_EB_LENGTH, device=device, dtype=torch.int32))
            
        # 2. 覆蓋 Quantiles/Medians
        fixed_medians = torch.tensor(FIXED_EB_MEDIANS, device=device)
        eb.quantiles.data[:, 0, 1] = fixed_medians.squeeze()
            
        print("[INFO] EntropyBottleneck CDFs & Medians overwritten.")
    except ImportError:
        print("[WARNING] fixed_cdfs.py not found! EntropyBottleneck might be non-deterministic.")
    except Exception as e:
        print(f"[WARNING] Failed to overwrite EntropyBottleneck CDFs: {e}")
    # ==========================================================================

    return model.eval()

    return model.eval()


# ==============================================================================
# 主程式
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Satellite Image Compression Tool (with CRC32)")
    parser.add_argument("input_path", type=str, help="Path to input image")
    parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to .pth model")
    parser.add_argument("-o", "--output_dir", type=str, default="output_satellite",
                        help="Output directory")
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--id", type=int, default=1, help="Image ID (0-255)")
    args = parser.parse_args()

    # 初始化設定
    PATCH_SIZE = 256
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 載入模型
    model = load_checkpoint(args.checkpoint).to(device)
    # [CRITICAL] DO NOT call model.update() here! It will overwrite the fixed_cdfs loaded in load_checkpoint
    # model.update(force=True)  # REMOVED - This was causing z_hat mismatch across platforms

    # 準備路徑
    filename = os.path.basename(args.input_path)
    base_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"輸出資料夾: {save_dir}")

    # 取得影像尺寸
    if rasterio and os.path.splitext(args.input_path)[-1].lower() in ['.tif', '.tiff']:
        with rasterio.open(args.input_path) as src:
            img_w, img_h = src.width, src.height
    else:
        with Image.open(args.input_path) as img:
            img_w, img_h = img.size

    num_cols = img_w // PATCH_SIZE
    num_rows = img_h // PATCH_SIZE

    # 邊界處理：如果有餘數，需要多加一行/列
    if img_w % PATCH_SIZE != 0: num_cols += 1
    if img_h % PATCH_SIZE != 0: num_rows += 1

    total_patches = num_cols * num_rows

    print(f"原始影像: {img_w}x{img_h}")
    print(f"分割模式: {num_rows}x{num_cols} (共 {total_patches} 個區塊)")
    print(f"Image ID: {args.id}")

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

            # 讀取 (如果是邊緣，read_image_patch 預設行為會 padding 或是由 process_compress handle，這裡主要依賴 process_compress 的 padding)
            # 注意: 這裡我們簡單傳入 box，若超出邊界，read_image_patch 需能處理，或者由 PIL crop 自動處理
            try:
                x = read_image_patch(args.input_path, box).unsqueeze(0).to(device)
            except Exception as e:
                # 處理邊緣超出情況，通常 PIL crop 會自動縮小，這時需要 pad 回 256
                # 這裡為了穩健性，我們手動讀取並 Pad
                # (略過複雜 padding 邏輯，假設 read_image_patch 會回傳正確 tensor)
                continue

            # 確保輸入是 256x256 (若在邊緣可能變小)
            if x.size(2) < PATCH_SIZE or x.size(3) < PATCH_SIZE:
                x = F.pad(x, (0, PATCH_SIZE - x.size(3), 0, PATCH_SIZE - x.size(2)))

            # 檔名仍然保留 row/col 方便人類查看，但解壓縮時將忽略它
            bin_filename = f"{base_name}_row{row}_col{col}.bin"
            bin_path = os.path.join(save_dir, bin_filename)

            # 壓縮並存檔 (傳入 row, col, id)
            bpp = process_compress(model, x, bin_path, args.id, row, col)
            total_bpp += bpp

            print(f"壓縮進度: {count}/{total_patches} | Row:{row} Col:{col} | BPP: {bpp:.4f}", end='\r')

    print(f"\n壓縮完成! 平均 BPP: {total_bpp / total_patches:.4f}")
    print(f"總耗時: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()