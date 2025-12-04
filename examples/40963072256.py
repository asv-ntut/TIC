import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict, OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

# import compressai
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.datasets import ImageFolder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# ✨ 修改：從 compressai.models.utils 導入 conv
from compressai.models.utils import conv, update_registered_buffers
from compressai.zoo import image_models as pretrained_models
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as architectures

# ==============================================================================
# (您的模型定義和輔助函式... 保持不變)
# ==============================================================================

# --- 模型 update 時需要的輔助函式 ---
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


# --- (輔助函式) conv 和 deconv ---
def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


# --- ✨ 新增：從訓練腳本複製過來的 PixelShuffle 輔助函式 ---
def deconv_pixelshuffle(in_channels, out_channels, kernel_size=5, stride=2):
    internal_channels = out_channels * (stride ** 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            internal_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ),
        nn.PixelShuffle(upscale_factor=stride)
    )


# --- ✨ 替換：使用訓練腳本的模型定義 ---

class SimpleConvStudentModel(nn.Module):
    def __init__(self, N=128, M=192):  # 使用訓練腳本的預設值
        super().__init__()
        self.N, self.M = N, M

        # g_a: Encoder (與訓練腳本一致，包含 BatchNorm)
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, M, kernel_size=3, stride=2),  # Output is M channels
        )

        # g_s: Decoder (與訓練腳本一致，使用 PixelShuffle 和 BatchNorm)
        self.g_s = nn.Sequential(
            deconv_pixelshuffle(M, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),
            nn.ReLU(inplace=True),

            deconv_pixelshuffle(N, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),
            nn.ReLU(inplace=True),

            deconv_pixelshuffle(N, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),
            nn.ReLU(inplace=True),

            deconv_pixelshuffle(N, 3, kernel_size=5, stride=2),
        )

        # h_a: Hyper Encoder (與訓練腳本一致)
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1), nn.ReLU(inplace=True),  # Input is M channels
            conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2)  # Output is N channels
        )

        # h_s: Hyper-Prior Decoder (與訓練腳本一致，使用 PixelShuffle)
        self.h_s = nn.Sequential(
            deconv_pixelshuffle(N, N, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            deconv_pixelshuffle(N, N, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            conv(N, M * 2, kernel_size=3, stride=1)  # Output is 2*M channels
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.g_a(x)  # [B, M, H/16, W/16]
        z = self.h_a(y)  # [B, N, H/64, W/64]
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)  # [B, 2*M, H/16, W/16]
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)  # [B, M, H/16, W/16]
        x_hat = self.g_s(y_hat)  # [B, 3, H, W]

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_hat": y_hat,
            "z_hat": z_hat
        }

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def update(self, scale_table=None, force=False):
        if scale_table is None: scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = any(m.update(force=force) for m in self.modules() if isinstance(m, EntropyBottleneck))
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
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

        print(f"偵測到模型結構: N={N}, M={M} (使用 BatchNorm 和 PixelShuffle)")
        net = cls(N, M)
        net.load_state_dict(state_dict, strict=False)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

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
# (輔助函式... 保持不變)
# ==============================================================================

architectures["simple_conv_student"] = SimpleConvStudentModel
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def collect_images(rootpath: str) -> List[str]:
    return [os.path.join(rootpath, f) for f in os.listdir(rootpath) if
            os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse) if mse > 0 else float('inf')


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


# ==============================================================================
# ✨✨✨【函式修改 1：inference】✨✨✨
# 1. 參數增加了 base_filename 和 patch_index
# 2. 儲存 .bin 時，在最前面寫入 patch_index (2 bytes)
# ==============================================================================
@torch.no_grad()
def inference(model, x, save=False, output_dir=None, base_filename=None, patch_index=None):
    # x 應該是 [3, 256, 256] 的 TENSOR

    x = x.unsqueeze(0)  # 變成 [1, 3, 256, 256]

    h, w = x.size(2), x.size(3)  # h=256, w=256

    # 你的模型需要 64 的倍數，所以 256 會被填充到 320
    p = 64
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top

    # x_padded 變成 [1, 3, 320, 320]
    x_padded = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode="constant", value=0)

    start = time.time()
    # 1. 呼叫 compress 函式
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    # 2. 呼叫 decompress 函式
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    # 裁剪回 256x256
    out_dec["x_hat"] = F.pad(out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom))

    # 計算 bpp (基於 256x256)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    # 3. 檢查是否需要儲存檔案
    if save and output_dir is not None and base_filename is not None:

        # --- 儲存解壓縮的 .png 檔案 ---
        reconstructed_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze().cpu())
        output_filepath = os.path.join(output_dir, f"{base_filename}_reconstructed.png")
        reconstructed_img.save(output_filepath)

        # --- 儲存 .bin 檔案 ---
        bin_filepath = os.path.join(output_dir, f"{base_filename}.bin")

        # 從壓縮輸出中提取資料
        y_string = out_enc["strings"][0][0]
        z_string = out_enc["strings"][1][0]
        shape = out_enc["shape"]  # 這是 z.size()[-2:]

        try:
            with open(bin_filepath, "wb") as f:

                # ✨✨✨【你的新需求：Header】✨✨✨
                # 1. 寫入 區塊編號 (Patch Index)，使用 2 bytes (uint16_t)
                #    (0 到 191，2 bytes 綽綽有餘)
                f.write(patch_index.to_bytes(2, 'little', signed=False))

                # --- (原始 header) ---
                # 2. 寫入 shape (H, W)，各 2 bytes (uint16_t)
                f.write(shape[0].to_bytes(2, 'little', signed=False))
                f.write(shape[1].to_bytes(2, 'little', signed=False))

                # 3. 寫入 z_string: 長度 4 bytes (uint32_t) + 原始資料
                f.write(len(z_string).to_bytes(4, 'little', signed=False))
                f.write(z_string)

                # 4. 寫入 y_string: 長度 4 bytes (uint32_t) + 原始資料
                f.write(len(y_string).to_bytes(4, 'little', signed=False))
                f.write(y_string)

        except Exception as e:
            print(f"Error saving bitstream: {e}")
        # --- 儲存邏輯結束 ---

    return {"psnr": psnr(x, out_dec["x_hat"]), "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
            "bpp": bpp, "encoding_time": enc_time, "decoding_time": dec_time}


# ==============================================================================
# (inference_entropy_estimation 函式保持不變)
# ==============================================================================
@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)
    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in
        out_net["likelihoods"].values())
    return {"psnr": psnr(x, out_net["x_hat"]), "bpp": bpp.item(), "encoding_time": elapsed_time / 2.0,
            "decoding_time": elapsed_time / 2.0}


# ==============================================================================
# (load_pretrained 和 load_checkpoint 函式保持不變)
# ==============================================================================
def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](quality=quality, metric=metric, pretrained=True).eval()


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    print(f"正在從 {checkpoint_path} 載入權重...")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint.get("state_dict", checkpoint)

    clean_state_dict = OrderedDict()
    is_dataparallel = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            is_dataparallel = True
            name = k[7:]  # remove 'module.'
        else:
            name = k
        clean_state_dict[name] = v

    if is_dataparallel:
        print("偵測到 'module.' 前綴，已自動移除。")

    return architectures[arch].from_state_dict(clean_state_dict).eval()


# ==============================================================================
# ✨✨✨【函式修改：eval_model (低記憶體 Petalinux 版)】✨✨✨
# 1. 改用 Pillow.open() (低記憶體)
# 2. 在迴圈中 "手刻" crop box
# 3. 每次迴圈只將 256x256 區塊轉為 Tensor
# ==============================================================================
def eval_model(model, filepaths, entropy_estimation=False, half=False, save=False, output_dir=None):
    device = next(model.parameters()).device

    # 這個 metrics 將儲存所有 "大型圖片" 的 "平均" 指標
    metrics = defaultdict(float)

    PATCH_SIZE = 256
    EXPECTED_W = 4096
    EXPECTED_H = 3072

    # 取得 Pytorch 的 ToTensor 轉換器 (我們只在迴圈內使用)
    to_tensor_transform = transforms.ToTensor()

    # 迴圈遍歷你 "dataset" 資料夾中的每一張 "大型圖片"
    for i, f in enumerate(filepaths):
        print(f"\n--- 正在開啟大型影像 {i + 1}/{len(filepaths)}: {os.path.basename(f)} ---")

        img_large = None
        try:
            # ✨【重點】使用 Pillow 開啟影像 (lazy loading，不佔記憶體)
            img_large = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"  錯誤：無法讀取影像 {f}。錯誤訊息: {e}。跳過此檔案。")
            if img_large: img_large.close()
            continue

        # 獲取影像尺寸 (Pillow .size 是 (W, H))
        img_w, img_h = img_large.size

        if img_w != EXPECTED_W or img_h != EXPECTED_H:
            print(f"  警告：影像 '{os.path.basename(f)}' 的大小不是 {EXPECTED_W}x{EXPECTED_H} (而是 {img_w}x{img_h})。")
            print("  將繼續嘗試切割，但可能不是 192 張。")

        num_patches_x = img_w // PATCH_SIZE  # 4096 // 256 = 16
        num_patches_y = img_h // PATCH_SIZE  # 3072 // 256 = 12
        total_patches = num_patches_y * num_patches_x  # 12 * 16 = 192

        if total_patches == 0:
            print(f"  錯誤：影像太小，無法切割成 {PATCH_SIZE}x{PATCH_SIZE} 的區塊。跳過此影像。")
            img_large.close()
            continue

        print(f"  影像將被切割成 {num_patches_y} (高) x {num_patches_x} (寬) = {total_patches} 張 256x256 區塊。")

        # 取得大圖的基礎檔名 (例如 "Tokyo_cropped_4096x3072")
        base_filename_large = os.path.splitext(os.path.basename(f))[0]

        # 儲存這張大圖 (192個區塊) 的所有指標
        patch_metrics_list = []
        patch_index = 0

        # ✨【重點】巢狀迴圈：從上到下 (y), 從左到右 (x)
        try:
            for y_idx in range(num_patches_y):  # 0 到 11
                for x_idx in range(num_patches_x):  # 0 到 15

                    print(f"    處理區塊 {patch_index + 1}/{total_patches} (Row {y_idx}, Col {x_idx})...", end="")

                    # 1. ✨【手刻】計算裁切座標 (left, upper, right, lower)
                    left = x_idx * PATCH_SIZE
                    upper = y_idx * PATCH_SIZE
                    right = left + PATCH_SIZE
                    lower = upper + PATCH_SIZE
                    box = (left, upper, right, lower)

                    # 2. ✨【手刻】使用 Pillow 裁切 256x256 區塊
                    #    (這時才真正從磁碟讀取這 256x256 的像素)
                    patch_img = img_large.crop(box)  # 這是一個 PIL Image 物件

                    # 3. ✨【手刻】只將這個小區塊轉為 Tensor
                    x_patch = to_tensor_transform(patch_img).to(device)  # [3, 256, 256]

                    if half:
                        model = model.half()
                        x_patch = x_patch.half()

                    # 4. 定義這個區塊的檔案名稱
                    patch_base_filename = f"{base_filename_large}_patch_{patch_index:03d}"

                    # 5. 執行推論 (壓縮與解壓縮)
                    if not entropy_estimation:
                        # 呼叫我們修改過的 inference 函式
                        rv = inference(model,
                                       x_patch,
                                       save,  # 傳遞 --save 參數
                                       output_dir,  # 傳遞輸出資料夾
                                       patch_base_filename,  # ✨ 傳遞 "區塊檔名"
                                       patch_index)  # ✨ 傳遞 "區塊編號"
                    else:
                        rv = inference_entropy_estimation(model, x_patch)

                    print(f" PSNR: {rv['psnr']:.2f} dB, BPP: {rv['bpp']:.4f}")
                    patch_metrics_list.append(rv)
                    patch_index += 1

                    # (手動釋放 patch_img 和 x_patch 記憶體，雖然 Python 會自動回收，
                    #  但在嵌入式系統上明確 del 總是有益的)
                    del patch_img
                    del x_patch

        finally:
            # ✨【重點】確保關閉大型影像的檔案控制代碼
            if img_large:
                img_large.close()
                print(f"  --- 已關閉影像檔案: {os.path.basename(f)} ---")

        # --- 處理完一張大圖 (192個區塊) ---

        # 計算這張大圖的 "平均" 指標
        avg_metrics_large = defaultdict(float)
        if not patch_metrics_list: continue

        for rv in patch_metrics_list:
            for k, v in rv.items():
                avg_metrics_large[k] += v

        print(f"  --- 大型影像 '{os.path.basename(f)}' 平均統計 ---")
        for k, v in avg_metrics_large.items():
            avg_metrics_large[k] = v / total_patches
            print(f"    Avg. {k}: {avg_metrics_large[k]:.4f}")

        # 將這張大圖的平均指標，累加到 "總指標" 中
        for k, v in avg_metrics_large.items():
            metrics[k] += v

    # --- 處理完所有大圖 ---

    if not filepaths:
        print("錯誤：未提供任何影像進行評估。")
        return metrics

    # 計算 "所有" 大圖的 "總平均" 指標
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    if save:
        print(f"\n處理完成。已儲存所有 .bin 和 .png 檔案至: {output_dir}")

    return metrics


# ==============================================================================
# (setup_args 和 main 函式保持不變)
# ==============================================================================

def setup_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path (包含 4096x3072 影像的資料夾)")
    parent_parser.add_argument("-a", "--architecture", type=str, required=True,
                               help="model architecture (e.g., simple_conv_student)")
    parent_parser.add_argument("-c", "--entropy-coder", choices=compressai.available_entropy_coders(),
                               default=compressai.available_entropy_coders()[0],
                               help="entropy coder (default: %(default)s)")
    parent_parser.add_argument("--cuda", action="store_true", help="enable CUDA")
    parent_parser.add_argument("--half", action="store_true", help="convert model to half floating point (fp16)")
    parent_parser.add_argument("--entropy-estimation", action="store_true",
                               help="use evaluated entropy estimation (no entropy coding)")
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parent_parser.add_argument("--save", action="store_true", help="儲存切割後的 .bin 和解壓縮的 .png 檔案")
    parent_parser.add_argument("-o", "--output_dir", type=str, default="reconstructed_patches",
                               help="儲存 192 個區塊檔案的資料夾 (default: %(default)s)")

    parser = argparse.ArgumentParser(description="Evaluate a model by tiling large images.", add_help=True)
    subparsers = parser.add_subparsers(help="model source", dest="source")

    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument("-m", "--metric", type=str, choices=["mse", "ms-ssim"], default="mse",
                                   help="metric trained against (default: %(default)s)")
    pretrained_parser.add_argument("-q", "--quality", dest="qualities", nargs="+", type=int, default=(1,))

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument("-p", "--path", dest="paths", type=str, nargs="*", required=True,
                                   help="checkpoint path")
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)
    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    # 這裡的 filepaths 會包含你 4096x3072 的 .tif 影像
    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print(f"Error: 在 '{args.dataset}' 資料夾中找不到任何影像。", file=sys.stderr)
        raise SystemExit(1)

    print(f"在 {args.dataset} 中找到 {len(filepaths)} 張影像。")

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"切割後的 .bin 和 .png 檔案將儲存至: {args.output_dir}")

    compressai.set_entropy_coder(args.entropy_coder)

    if args.source == "pretrained":
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    elif args.source == "checkpoint":
        runs = args.paths
        opts = (args.architecture,)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()

        model = load_func(*opts, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            print("模型已移至 CUDA")
        else:
            model = model.to("cpu")
            print("模型已移至 CPU")

        if args.half and not args.cuda:
            print("警告: --half 參數通常與 --cuda 一起使用。在 CPU 上使用 FP16 可能會更慢或不受支援。")

        if args.source == "checkpoint":
            print("正在更新模型 (update CDFs)...")
            model.update(force=True)
            print("模型更新完畢。")

        # 呼叫我們修改過的 eval_model
        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.save, args.output_dir)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = ("entropy estimation" if args.entropy_estimation else args.entropy_coder)
    output = {"name": args.architecture, "description": f"Inference ({description})", "results": results}

    print("\n--- 總體平均指標 (所有大型影像的平均) ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])