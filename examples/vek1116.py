import argparse
import json
import math
import os
import sys
import time
import subprocess  # <-- 1. 為了呼叫 tar 和 md5sum
import shutil  # <-- 2. 為了刪除暫存資料夾
from collections import defaultdict, OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

# --- 假設 compressai 已安裝在環境中 ---
try:
    import compressai
    from compressai.ans import BufferedRansEncoder, RansDecoder
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
    from compressai.models.utils import conv, update_registered_buffers
    from compressai.zoo import image_models as pretrained_models
    from compressai.zoo import load_state_dict
except ImportError:
    print("\n錯誤：找不到 'compressai' 函式庫。")
    sys.exit(1)

# ==============================================================================
# ( 1. 你的模型定義 )
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
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y_hat": y_hat, "z_hat": z_hat
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
# 輔助函式
# ==============================================================================
architectures = {
    "simple_conv_student": SimpleConvStudentModel
}
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
# ✨✨✨ 推論 (Inference) 函式 (修正版：移除 Padding) ✨✨✨
# ==============================================================================
# ==============================================================================
# ✨✨✨ 推論 (Inference) 函式 (【最終修正版】) ✨✨✨
# ==============================================================================
@torch.no_grad()
def inference(model, x, save=False, output_dir=None, base_filename=None, patch_index=None):
    x = x.unsqueeze(0)  # [1, 3, 256, 256]

    # --- 【修正】移除 Padding，因為 256 是 64 的倍數 ---
    x_padded = x
    # --- 【修正結束】---

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    # --- 【修正】移除 Un-padding ---
    # out_dec["x_hat"] = F.pad(out_dec["x_hat"], ...)
    # --- 【修正結束】---

    num_pixels = x.size(0) * x.size(2) * x.size(3)

    # --- ✨✨✨【BPP 修正】使用 new, correct logic to calculate bpp ✨✨✨ ---
    bpp = sum(len(s) for s_list in out_enc["strings"] for s in s_list) * 8.0 / num_pixels

    if save and output_dir is not None and base_filename is not None:

        # --- ✨✨✨【NameError 修正】在儲存前，必須先定義變數 ✨✨✨ ---
        reconstructed_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze().cpu())
        output_filepath = os.path.join(output_dir, f"{base_filename}_reconstructed.png")

        # 1. 儲存 .png (現在可以正常運作了)
        reconstructed_img.save(output_filepath)

        # 2. 儲存 .bin (使用我們新的多字串邏輯)
        bin_filepath = os.path.join(output_dir, f"{base_filename}.bin")
        y_strings_list = out_enc["strings"][0]
        z_strings_list = out_enc["strings"][1]
        shape = out_enc["shape"]

        try:
            with open(bin_filepath, "wb") as f:
                # 1. 寫入 Shape
                f.write(shape[0].to_bytes(2, 'little', signed=False))
                f.write(shape[1].to_bytes(2, 'little', signed=False))

                # 2. 寫入 Z 字串 (通常只有1個)
                f.write(len(z_strings_list).to_bytes(2, 'little', signed=False))  # 寫入 Z 列表的長度
                for s in z_strings_list:
                    f.write(len(s).to_bytes(4, 'little', signed=False))  # 寫入這個字串的長度
                    f.write(s)  # 寫入字串本身

                # 3. 寫入 Y 字串 (可能有很多個)
                f.write(len(y_strings_list).to_bytes(2, 'little', signed=False))  # 寫入 Y 列表的長度
                for s in y_strings_list:
                    f.write(len(s).to_bytes(4, 'little', signed=False))  # 寫入這個字串的長度
                    f.write(s)  # 寫入字串本身
        except Exception as e:
            print(f"Error saving bitstream: {e}")

    # 計算評估指標
    psnr_val = psnr(x, out_dec["x_hat"])
    ms_ssim_val = ms_ssim(x, out_dec["x_hat"], data_range=1.0).item()

    return {"psnr": psnr_val, "ms-ssim": ms_ssim_val,
            "bpp": bpp, "encoding_time": enc_time, "decoding_time": dec_time}


# ... (eval_model, setup_args, main 函式保持不變，它們都包含了 tar/md5sum 邏輯) ...
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
            name = k[7:]
        else:
            name = k
        clean_state_dict[name] = v
    if is_dataparallel:
        print("偵測到 'module.' 前綴，已自動移除。")
    if arch not in architectures:
        print(f"錯誤: 架構 '{arch}' 未在 'architectures' 字典中註冊。")
        sys.exit(1)
    return architectures[arch].from_state_dict(clean_state_dict).eval()


def eval_model(model, filepaths, entropy_estimation=False, half=False, save=False, output_dir=None, use_tar=False,
               keep_bins=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    to_tensor_transform = transforms.ToTensor()
    PATCH_SIZE = 256
    EXPECTED_W = 4096
    EXPECTED_H = 3072
    for i, f in enumerate(filepaths):
        print(f"\n--- 正在開啟大型影像 {i + 1}/{len(filepaths)}: {os.path.basename(f)} ---")
        img_large = None
        try:
            img_large = Image.open(f).convert("RGB")
        except Exception as e:
            print(f"  錯誤：無法讀取影像 {f}。錯誤訊息: {e}。跳過此檔案。")
            if img_large: img_large.close()
            continue
        img_w, img_h = img_large.size
        if img_w != EXPECTED_W or img_h != EXPECTED_H:
            print(f"  警告：影像 '{os.path.basename(f)}' 的大小不是 {EXPECTED_W}x{EXPECTED_H} (而是 {img_w}x{img_h})。")
        num_patches_x = img_w // PATCH_SIZE
        num_patches_y = img_h // PATCH_SIZE
        total_patches = num_patches_y * num_patches_x
        if total_patches == 0:
            print(f"  錯誤：影像太小，無法切割成 {PATCH_SIZE}x{PATCH_SIZE} 的區塊。跳過此影像。")
            img_large.close()
            continue
        print(f"  影像將被切割成 {num_patches_y} (高) x {num_patches_x} (寬) = {total_patches} 張 256x256 區塊。")
        base_filename_large = os.path.splitext(os.path.basename(f))[0]
        current_image_patch_dir = os.path.join(output_dir, base_filename_large)
        if save:
            os.makedirs(current_image_patch_dir, exist_ok=True)
            print(f"  區塊將儲存至: {current_image_patch_dir}")
        patch_metrics_list = []
        patch_bin_files_list = []
        patch_index = 0
        try:
            for y_idx in range(num_patches_y):
                for x_idx in range(num_patches_x):
                    print(f"    處理區塊 {patch_index + 1}/{total_patches} (Row {y_idx}, Col {x_idx})...", end="")
                    left = x_idx * PATCH_SIZE
                    upper = y_idx * PATCH_SIZE
                    right = left + PATCH_SIZE
                    lower = upper + PATCH_SIZE
                    box = (left, upper, right, lower)
                    patch_img = img_large.crop(box)
                    x_patch = to_tensor_transform(patch_img).to(device)
                    if half:
                        model = model.half()
                        x_patch = x_patch.half()
                    patch_base_filename = f"{base_filename_large}_patch_{patch_index:03d}"
                    if not entropy_estimation:
                        rv = inference(model,
                                       x_patch,
                                       save,
                                       current_image_patch_dir,
                                       patch_base_filename,
                                       patch_index)
                        if save and use_tar:
                            bin_name = f"{patch_base_filename}.bin"
                            relative_bin_path = os.path.join(base_filename_large, bin_name)
                            patch_bin_files_list.append(relative_bin_path)
                    else:
                        rv = inference_entropy_estimation(model, x_patch)
                    print(f" PSNR: {rv['psnr']:.2f} dB, BPP: {rv['bpp']:.4f}")
                    patch_metrics_list.append(rv)
                    patch_index += 1
                    del patch_img
                    del x_patch
        finally:
            if img_large:
                img_large.close()
                print(f"  --- 已關閉影像檔案: {os.path.basename(f)} ---")
        avg_metrics_large = defaultdict(float)
        if not patch_metrics_list: continue
        for rv in patch_metrics_list:
            for k, v in rv.items():
                avg_metrics_large[k] += v
        print(f"  --- 大型影像 '{os.path.basename(f)}' 平均統計 ---")
        for k, v in avg_metrics_large.items():
            avg_metrics_large[k] = v / total_patches
            print(f"    Avg. {k}: {avg_metrics_large[k]:.4f}")
        for k, v in avg_metrics_large.items():
            metrics[k] += v
        if save and use_tar and patch_bin_files_list:
            print(f"  --- Gnu tar {len(patch_bin_files_list)} .bin ... ---")
            tar_filename = f"{base_filename_large}.tar"
            tar_filepath = os.path.join(output_dir, tar_filename)
            tar_command = ["tar", "-cf", tar_filename]
            tar_command.extend(patch_bin_files_list)
            try:
                subprocess.run(tar_command, check=True, cwd=output_dir, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
                print(f"  ✅ : {tar_filepath}")
                try:
                    md5_filename = f"{tar_filename}.md5"
                    md5_filepath = os.path.join(output_dir, md5_filename)
                    md5_command = ["md5sum", tar_filename]
                    md5_result = subprocess.run(
                        md5_command, check=True, cwd=output_dir,
                        capture_output=True, text=True, encoding='utf-8'
                    )
                    md5_checksum_output = md5_result.stdout.strip()
                    with open(md5_filepath, "w", encoding='utf-8') as md5_f:
                        md5_f.write(md5_checksum_output + "\n")
                    print(f"  ✅ Checksum: {md5_filename}")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ : 'md5sum' (code {e.returncode})")
                    print(f"  : {e.stderr.decode()}")
                except FileNotFoundError:
                    print("  ❌ : 'md5sum' ")
                if not keep_bins:
                    shutil.rmtree(current_image_patch_dir)
                    print(f"  🗑️ : {current_image_patch_dir}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ : 'tar' (code {e.returncode})")
                print(f"  : {e.stderr.decode()}")
            except FileNotFoundError:
                print("  ❌ : 'tar' ")
    if not filepaths:
        print("：")
        return metrics
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    if save:
        print(f"\n: {output_dir}")
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path ( 4096x3072 )")
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
    parent_parser.add_argument("--save", action="store_true", help=" .bin .png ")
    parent_parser.add_argument(
        "--use-tar", action="store_true",
        help="[Linux/macOS] .bin .tar ( tar)"
    )
    parent_parser.add_argument(
        "--keep-bins", action="store_true",
        help=" --use-tar .bin "
    )
    parent_parser.add_argument("-o", "--output_dir", type=str, default="no_header",
                               help=" 192 (default: %(default)s)")
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
    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print(f"Error: '{args.dataset}' .", file=sys.stderr)
        raise SystemExit(1)
    print(f" {args.dataset}  {len(filepaths)} .")
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f" .bin  .png : {args.output_dir}")
        if args.use_tar:
            print(" 'tar' ( .tar ).")
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
            print(" CUDA")
        else:
            model = model.to("cpu")
            print(" CPU")
        if args.half and not args.cuda:
            print(": --half --cuda .")
        if args.source == "checkpoint":
            print(" (update CDFs)...")
            model.update(force=True)
            print(".")
        metrics = eval_model(
            model, filepaths, args.entropy_estimation, args.half, args.save,
            args.output_dir, args.use_tar, args.keep_bins
        )
        for k, v in metrics.items():
            results[k].append(v)
    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()
    description = ("entropy estimation" if args.entropy_estimation else args.entropy_coder)
    output = {"name": args.architecture, "description": f"Inference ({description})", "results": results}
    print("\n---  ( ) ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])