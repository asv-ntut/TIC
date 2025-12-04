import argparse
import os
import sys
import math
import time
from collections import defaultdict, OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- 導入 compressai 相關模組 ---
try:
    import compressai
    from compressai.ans import BufferedRansEncoder, RansDecoder
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
    from compressai.models.utils import conv, update_registered_buffers
    from compressai.zoo import load_state_dict
except ImportError:
    print("\n錯誤：找不到 'compressai' 函式庫。")
    sys.exit(1)

# ==============================================================================
# ( 1. 你的模型定義 - 必須和 vek1116.py 100% 相同 )
# ==============================================================================

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        output_padding=stride - 1, padding=kernel_size // 2,
    )


def deconv_pixelshuffle(in_channels, out_channels, kernel_size=5, stride=2):
    internal_channels = out_channels * (stride ** 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels, internal_channels, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2,
        ),
        nn.PixelShuffle(upscale_factor=stride)
    )


class SimpleConvStudentModel(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.N, self.M = N, M
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, M, kernel_size=3, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv_pixelshuffle(M, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            deconv_pixelshuffle(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            deconv_pixelshuffle(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            deconv_pixelshuffle(N, 3, kernel_size=5, stride=2),
        )
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2)
        )
        self.h_s = nn.Sequential(
            deconv_pixelshuffle(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            deconv_pixelshuffle(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            conv(N, M * 2, kernel_size=3, stride=1)
        )
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        # (解壓縮時不需要 self.apply(self._init_weights))

    def update(self, scale_table=None, force=False):
        if scale_table is None: scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = any(m.update(force=force) for m in self.modules() if isinstance(m, EntropyBottleneck))
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.entropy_bottleneck, "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"], state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        # (使用和 vek1116.py 100% 相同的 from_state_dict 邏輯)
        try:
            first_g_a_weight_key = 'g_a.0.weight'
            if first_g_a_weight_key not in state_dict:
                g_a_weight_keys = sorted([k for k in state_dict if k.startswith('g_a.') and k.endswith('.weight')])
                if not g_a_weight_keys: raise KeyError("No g_a weights found in state_dict for N")
                first_g_a_weight_key = g_a_weight_keys[0]
            N = state_dict[first_g_a_weight_key].size(0)
        except Exception as e:
            print(f"Error inferring N in from_state_dict: {e}. Assuming default N=128.")
            N = 128
        try:
            g_a_weight_keys = sorted([k for k in state_dict if k.startswith('g_a.') and k.endswith('.weight')])
            if not g_a_weight_keys: raise KeyError("No g_a weights found in state_dict for M")
            last_g_a_weight_key = g_a_weight_keys[-1]
            expected_key = 'g_a.9.weight'
            if expected_key in state_dict:
                M = state_dict[expected_key].size(0)
            else:
                print(
                    f"Warning: Expected key '{expected_key}' not found. Inferring M from last g_a key: '{last_g_a_weight_key}'")
                M = state_dict[last_g_a_weight_key].size(0)
        except Exception as e:
            print(f"Error inferring M in from_state_dict: {e}. Assuming default M=192.")
            M = 192
        print(f"偵測到模型結構: N={N}, M={M}")
        net = cls(N, M)
        net.load_state_dict(state_dict, strict=False)
        return net

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


# ==============================================================================
# ( 2. 載入權重的輔助函式 )
# ==============================================================================
architectures = {
    "simple_conv_student": SimpleConvStudentModel
}


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    print(f"正在從 {checkpoint_path} 載入權重...")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        clean_state_dict[name] = v
    if any(k.startswith('module.') for k in state_dict):
        print("偵測到 'module.' 前綴，已自動移除。")
    if arch not in architectures:
        print(f"錯誤: 架構 '{arch}' 未在 'architectures' 字典中註冊。")
        sys.exit(1)
    model = architectures[arch].from_state_dict(clean_state_dict).eval()
    model.update(force=True)
    print("模型權重載入並更新完畢。")
    return model


# ==============================================================================
# ✨✨✨ ( 3. 讀取 .bin 檔案的輔助函式 ) (修正版) ✨✨✨
# ==============================================================================
def read_bin_file(filepath):
    """讀取我們自訂的 (多字串) .bin 檔案"""
    with open(filepath, 'rb') as f:
        # 1. 讀取 Shape
        shape_h = int.from_bytes(f.read(2), 'little', signed=False)
        shape_w = int.from_bytes(f.read(2), 'little', signed=False)
        shape = (shape_h, shape_w)

        # 2. 讀取 Z 字串列表
        z_strings_list = []
        num_z_strings = int.from_bytes(f.read(2), 'little', signed=False) # 讀取 Z 列表長度
        for _ in range(num_z_strings):
            s_len = int.from_bytes(f.read(4), 'little', signed=False)   # 讀取字串長度
            s = f.read(s_len)                                          # 讀取字串
            z_strings_list.append(s)

        # 3. 讀取 Y 字串列表
        y_strings_list = []
        num_y_strings = int.from_bytes(f.read(2), 'little', signed=False) # 讀取 Y 列表長度
        for _ in range(num_y_strings):
            s_len = int.from_bytes(f.read(4), 'little', signed=False)   # 讀取字串長度
            s = f.read(s_len)                                          # 讀取字串
            y_strings_list.append(s)

        # 傳回 decompress() 期望的格式: [List[y_strings], List[z_strings]]
        return [y_strings_list, z_strings_list], shape


# ==============================================================================
# ✨✨✨ 主函式 - 拼圖邏輯 (修正版：移除 Padding) ✨✨✨
# ==============================================================================
def main(argv):
    parser = argparse.ArgumentParser(description="Decompress and tile patches into a large image.")
    # ... (parser.add_argument... 保持不變) ...
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Path to the directory containing .bin patches")
    parser.add_argument("-p", "--path", type=str, required=True, help="Checkpoint path (model.pth.tar)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the final reconstructed PNG")
    parser.add_argument("-a", "--architecture", type=str, default="simple_conv_student", help="Model architecture")
    parser.add_argument("--size", type=int, nargs=2, default=(4096, 3072), help="Size of the final large image (W H)")
    parser.add_argument("--patch-size", type=int, default=256, help="Size of the square patches")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for decompression")
    parser.add_argument("--half", action="store_true", help="Use FP16 (half) precision")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dataset):
        print(f"錯誤: 輸入路徑 '{args.dataset}' 不是一個有效的資料夾。")
        sys.exit(1)

    # --- 1. 載入模型 ---
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.architecture, args.path)
    model = model.to(device)
    if args.half:
        model = model.half()
    print(f"模型已移至 {device} 並設定為評估模式。")

    # --- 2. 準備畫布 ---
    W_LARGE, H_LARGE = args.size
    P_SIZE = args.patch_size
    num_x = W_LARGE // P_SIZE  # 16
    num_y = H_LARGE // P_SIZE  # 12
    total_patches = num_y * num_x
    print(f"準備拼圖: {num_x} (寬) x {num_y} (高) = {total_patches} 塊 (每塊 {P_SIZE}x{P_SIZE})")

    canvas = torch.zeros(3, H_LARGE, W_LARGE, dtype=torch.float32)
    dir_name = os.path.basename(os.path.normpath(args.dataset))
    patch_index = 0
    found_count = 0
    start_time = time.time()

    # --- 3. (移除 Padding 計算，因為 256 是 64 的倍數) ---

    with torch.no_grad():
        for y_idx in range(num_y):  # 0 to 11 (高)
            for x_idx in range(num_x):  # 0 to 15 (寬)

                patch_base_filename = f"{dir_name}_patch_{patch_index:03d}"
                bin_filename = f"{patch_base_filename}.bin"
                bin_filepath = os.path.join(args.dataset, bin_filename)

                if os.path.exists(bin_filepath):
                    try:
                        # 【關鍵修正】
                        strings, shape = read_bin_file(bin_filepath)

                        out_dec = model.decompress(strings, shape)

                        # x_hat 現在就是 [1, 3, 256, 256]
                        x_hat = out_dec["x_hat"].to(device)

                        if args.half:
                            x_hat = x_hat.half()

                        # (移除 F.pad 裁切)

                        y_start = y_idx * P_SIZE
                        x_start = x_idx * P_SIZE

                        canvas[:, y_start:y_start + P_SIZE, x_start:x_start + P_SIZE] = x_hat.squeeze(0).cpu().float()
                        found_count += 1

                    except Exception as e:
                        print(f"\n錯誤: 處理 {bin_filename} 失敗: {e}. 留黑。")

                else:
                    print(f"\n警告: 區塊 {bin_filename} 不存在 (掉包)。將在該位置留黑。")

                patch_index += 1

                sys.stdout.write(f"\r  處理中... {patch_index}/{total_patches}")
                sys.stdout.flush()

    end_time = time.time()
    print(f"\n--- 拼圖完成 (耗時: {end_time - start_time:.2f} 秒) ---")
    print(f"  成功找到並解壓縮 {found_count} / {total_patches} 個區塊。")
    if total_patches != found_count:
        print(f"  遺失了 {total_patches - found_count} 個區塊 (已顯示為黑色)。")

    # --- 5. 儲存最終影像 ---
    try:
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(canvas)
        pil_img.save(args.output)
        print(f"✅ 成功儲存拼圖影像至: {args.output}")
    except Exception as e:
        print(f"錯誤: 儲存最終影像失敗: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])