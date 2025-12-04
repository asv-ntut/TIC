# # Copyright (c) 2021-2022, InterDigital Communications, Inc
# # All rights reserved.
# # (版權宣告與原檔案相同)
# """
# Evaluate an end-to-end compression model on an image dataset.
# """
# import argparse
# import json
# import math
# import os
# import sys
# import time
#
# from collections import defaultdict
# from typing import List
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from PIL import Image
# from pytorch_msssim import ms_ssim
# from torchvision import transforms
#
#
#
# import compressai
#
# from compressai.zoo import image_models as pretrained_models
# from compressai.zoo import load_state_dict
# from compressai.zoo.image import model_architectures as architectures
#
#
#
#
# torch.backends.cudnn.deterministic = True
# torch.set_num_threads(1)
#
# # from torchvision.datasets.folder
# IMG_EXTENSIONS = (
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".ppm",
#     ".bmp",
#     ".pgm",
#     ".tif",
#     ".tiff",
#     ".webp",
# )
#
#
# def collect_images(rootpath: str) -> List[str]:
#     return [
#         os.path.join(rootpath, f)
#         for f in os.listdir(rootpath)
#         if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
#     ]
#
#
# def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
#     mse = F.mse_loss(a, b).item()
#     return -10 * math.log10(mse) if mse > 0 else float('inf')
#
#
# def read_image(filepath: str) -> torch.Tensor:
#     assert os.path.isfile(filepath)
#     img = Image.open(filepath).convert("RGB")
#     return transforms.ToTensor()(img)
#
#
# @torch.no_grad()
# def inference(model, x, save=False, output_dir=None, filepath=None):
#     x = x.unsqueeze(0)
#
#     h, w = x.size(2), x.size(3)
#     p = 64  # maximum 6 strides of 2
#     new_h = (h + p - 1) // p * p
#     new_w = (w + p - 1) // p * p
#     padding_left = (new_w - w) // 2
#     padding_right = new_w - w - padding_left
#     padding_top = (new_h - h) // 2
#     padding_bottom = new_h - h - padding_top
#     x_padded = F.pad(
#         x,
#         (padding_left, padding_right, padding_top, padding_bottom),
#         mode="constant",
#         value=0,
#     )
#
#     start = time.time()
#     out_enc = model.compress(x_padded)
#     enc_time = time.time() - start
#
#     start = time.time()
#     out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
#     dec_time = time.time() - start
#
#     out_dec["x_hat"] = F.pad(
#         out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
#     )
#
#     num_pixels = x.size(0) * x.size(2) * x.size(3)
#     bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
#
#     if save and output_dir is not None and filepath is not None:
#         reconstructed_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze().cpu())
#         basename = os.path.splitext(os.path.basename(filepath))[0]
#         output_filepath = os.path.join(output_dir, f"{basename}_reconstructed.png")
#         reconstructed_img.save(output_filepath)
#
#     return {
#         "psnr": psnr(x, out_dec["x_hat"]),
#         "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
#         "bpp": bpp,
#         "encoding_time": enc_time,
#         "decoding_time": dec_time,
#     }
#
#
# @torch.no_grad()
# def inference_entropy_estimation(model, x):
#     x = x.unsqueeze(0)
#
#     start = time.time()
#     out_net = model.forward(x)
#     elapsed_time = time.time() - start
#
#     num_pixels = x.size(0) * x.size(2) * x.size(3)
#     bpp = sum(
#         (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#         for likelihoods in out_net["likelihoods"].values()
#     )
#
#     return {
#         "psnr": psnr(x, out_net["x_hat"]),
#         "bpp": bpp.item(),
#         "encoding_time": elapsed_time / 2.0,  # broad estimation
#         "decoding_time": elapsed_time / 2.0,
#     }
#
#
# def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
#     return pretrained_models[model](
#         quality=quality, metric=metric, pretrained=True
#     ).eval()
#
#
# def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
#     # 載入 checkpoint，注意這裡的 'state_dict' key 要跟你儲存時的一致
#     # 你的訓練腳本儲存的 key 是 "state_dict"，所以這邊不需要改
#     checkpoint = torch.load(checkpoint_path)
#     state_dict = checkpoint.get("state_dict", checkpoint)  # 兼容兩種儲存格式
#
#     # 這裡會根據 -a 傳入的架構名稱 (例如 "simple_conv_student")
#     # 去我們修改過的 architectures 字典裡找到對應的 class，並呼叫 from_state_dict
#     return architectures[arch].from_state_dict(state_dict).eval()
#
#
# def eval_model(model, filepaths, entropy_estimation=False, half=False, save=False, output_dir=None):
#     device = next(model.parameters()).device
#     metrics = defaultdict(float)
#     # 使用 for f in tqdm(filepaths): 可以顯示進度條，更方便
#     for f in filepaths:
#         x = read_image(f).to(device)
#         if not entropy_estimation:
#             if half:
#                 model = model.half()
#                 x = x.half()
#             rv = inference(model, x, save, output_dir, f)
#         else:
#             rv = inference_entropy_estimation(model, x)
#         for k, v in rv.items():
#             metrics[k] += v
#     for k, v in metrics.items():
#         metrics[k] = v / len(filepaths)
#     return metrics
#
#
# def setup_args():
#     parent_parser = argparse.ArgumentParser(
#         add_help=False,
#     )
#
#     # Common options.
#     parent_parser.add_argument("dataset", type=str, help="dataset path")
#     parent_parser.add_argument(
#         "-a",
#         "--architecture",
#         type=str,
#         # ✨✨✨【修改點 3: 移除 `choices` 限制】✨✨✨
#         # 移除 choices=pretrained_models.keys()，
#         # 這樣我們才能傳入自訂的架構名稱 "simple_conv_student" 而不會報錯
#         # choices=pretrained_models.keys(),
#         # ✨✨✨【修改結束】✨✨✨
#         help="model architecture",
#         required=True,
#     )
#     parent_parser.add_argument(
#         "-c",
#         "--entropy-coder",
#         choices=compressai.available_entropy_coders(),
#         default=compressai.available_entropy_coders()[0],
#         help="entropy coder (default: %(default)s)",
#     )
#     parent_parser.add_argument(
#         "--cuda",
#         action="store_true",
#         help="enable CUDA",
#     )
#     parent_parser.add_argument(
#         "--half",
#         action="store_true",
#         help="convert model to half floating point (fp16)",
#     )
#     parent_parser.add_argument(
#         "--entropy-estimation",
#         action="store_true",
#         help="use evaluated entropy estimation (no entropy coding)",
#     )
#     parent_parser.add_argument(
#         "-v",
#         "--verbose",
#         action="store_true",
#         help="verbose mode",
#     )
#     parent_parser.add_argument(
#         "--save", action="store_true", help="Save reconstructed images"
#     )
#     parent_parser.add_argument(
#         "-o",
#         "--output_dir",
#         type=str,
#         default="reconstructed",
#         help="Directory to save reconstructed images (default: %(default)s)",
#     )
#
#     parser = argparse.ArgumentParser(
#         description="Evaluate a model on an image dataset.", add_help=True
#     )
#     subparsers = parser.add_subparsers(help="model source", dest="source")
#
#     pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
#     pretrained_parser.add_argument(
#         "-m",
#         "--metric",
#         type=str,
#         choices=["mse", "ms-ssim"],
#         default="mse",
#         help="metric trained against (default: %(default)s)",
#     )
#     pretrained_parser.add_argument(
#         "-q",
#         "--quality",
#         dest="qualities",
#         nargs="+",
#         type=int,
#         default=(1,),
#     )
#
#     checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
#     checkpoint_parser.add_argument(
#         "-p",
#         "--path",
#         dest="paths",
#         type=str,
#         nargs="*",
#         required=True,
#         help="checkpoint path",
#     )
#
#     return parser
#
#
# def main(argv):
#     parser = setup_args()
#     args = parser.parse_args(argv)
#
#     if not args.source:
#         print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
#         parser.print_help()
#         raise SystemExit(1)
#
#     filepaths = collect_images(args.dataset)
#     if len(filepaths) == 0:
#         print("Error: no images found in directory.", file=sys.stderr)
#         raise SystemExit(1)
#
#     if args.save:
#         os.makedirs(args.output_dir, exist_ok=True)
#
#     compressai.set_entropy_coder(args.entropy_coder)
#
#     if args.source == "pretrained":
#         runs = sorted(args.qualities)
#         opts = (args.architecture, args.metric)
#         load_func = load_pretrained
#         log_fmt = "\rEvaluating {0} | {run:d}"
#     elif args.source == "checkpoint":
#         runs = args.paths
#         opts = (args.architecture,)
#         load_func = load_checkpoint
#         log_fmt = "\rEvaluating {run:s}"
#
#     results = defaultdict(list)
#     for run in runs:
#         if args.verbose:
#             sys.stderr.write(log_fmt.format(*opts, run=run))
#             sys.stderr.flush()
#
#         model = load_func(*opts, run)
#
#         if args.cuda and torch.cuda.is_available():
#             model = model.to("cuda")
#
#         # 訓練完的模型在評估前需要呼叫 update() 來更新 CDFs
#         if args.source == "checkpoint":
#             model.update(force=True)
#
#         metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.save, args.output_dir)
#         for k, v in metrics.items():
#             results[k].append(v)
#
#     if args.verbose:
#         sys.stderr.write("\n")
#         sys.stderr.flush()
#
#     description = (
#         "entropy estimation" if args.entropy_estimation else args.entropy_coder
#     )
#     output = {
#         "name": args.architecture,
#         "description": f"Inference ({description})",
#         "results": results,
#     }
#
#     print(json.dumps(output, indent=2))
#
#
# if __name__ == "__main__":
#     main(sys.argv[1:])

import argparse
import json
import math
import os
import sys
import time

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

# [--- MODIFIED 1 ---]
import numpy as np
try:
    import rasterio
except ImportError:
    rasterio = None
    print("⚠️ 警告: Rasterio 載入失敗，無法讀取衛星 TIF 檔，但跑 PNG 沒問題。")
# [--- MODIFIED END ---]

import compressai

from compressai.zoo import image_models as pretrained_models
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as architectures

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

# [--- MODIFIED 2 ---]
# 將 TIF 相關的邏輯移到 read_image 中
TIF_EXTENSIONS = (".tif", ".tiff")


def collect_images(rootpath: str) -> List[str]:
    """Collects 8-bit image paths."""
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS and os.path.splitext(f)[-1].lower() not in TIF_EXTENSIONS
    ]


def collect_tif_images(rootpath: str) -> List[str]:
    """Collects TIF image paths."""
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in TIF_EXTENSIONS
    ]


# [--- MODIFIED END ---]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    # PSNR on [0, 1] range
    mse = F.mse_loss(a, b).item()
    if mse == 0:
        return float('inf')
    return -10 * math.log10(mse)  # -10 * log10(MSE) = 20 * log10(1.0 / sqrt(MSE))


# [--- MODIFIED 3 ---]
# 替換 read_image 函式
def read_image(filepath: str, dataset_type: str = "normal") -> torch.Tensor:
    """
    Reads an image file, processing it based on dataset_type.
    'normal': 8-bit RGB (JPG, PNG) -> [0, 1] float tensor (auto / 255.0)
    'tif': 16-bit satellite TIF -> [0, 1] float tensor (manual / 10000.0)
    """
    assert os.path.isfile(filepath)

    if dataset_type == "tif":
        # --- 16-bit TIF 邏輯 (同 Dataloader) ---
        RGB_BANDS = [3, 2, 1]  # B4, B3, B2
        CLIP_MIN = 0.0
        CLIP_MAX = 10000.0
        SCALE = 10000.0

        try:
            with rasterio.open(filepath) as src:
                raw_data = src.read().astype(np.float32)

            if np.isnan(raw_data).any():
                for i in range(raw_data.shape[0]):
                    band = raw_data[i]
                    if np.isnan(band).any():
                        band[np.isnan(band)] = np.nanmean(band)
                        raw_data[i] = band

            rgb_data = raw_data[RGB_BANDS, :, :]
            clipped_data = np.clip(rgb_data, CLIP_MIN, CLIP_MAX)
            normalized_data = clipped_data / SCALE
            return torch.from_numpy(normalized_data.copy())

        except Exception as e:
            print(f"Error reading TIF {filepath}: {e}")
            raise

    elif dataset_type == "normal":
        # --- 8-bit 影像邏輯 (transforms.ToTensor 會自動除以 255.0) ---
        img = Image.open(filepath).convert("RGB")
        return transforms.ToTensor()(img)  # 自動轉為 [0, 1] 比例

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# [--- MODIFIED END ---]


@torch.no_grad()
def inference(model, x, save=False, output_dir=None, filepath=None):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    if save and output_dir is not None and filepath is not None:
        # out_dec["x_hat"] is [0, 1] float, ToPILImage handles conversion to 8-bit PNG
        reconstructed_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze().cpu())
        basename = os.path.splitext(os.path.basename(filepath))[0]
        output_filepath = os.path.join(output_dir, f"{basename}_reconstructed.png")
        reconstructed_img.save(output_filepath)

    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),  # data_range is 1.0
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    # 1. 載入訓練好的權重
    # map_location='cpu' 確保即使沒有 GPU 也能載入
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # 2. 修正 DataParallel 的權重名稱 (如果有的話)
    # 如果訓練時用了多張顯卡，權重前面會有 "module."，需要拿掉才能用
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # 去掉 "module."
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    # 3. 確保模型架構有被匯入
    if arch not in architectures:
        # 如果你的模型不在 compressai 預設庫中，嘗試手動載入
        try:
            from compressai.models.tic import TIC
            architectures[arch] = TIC
        except ImportError:
            pass

    # 4. [修改點] 直接使用 from_state_dict 自動載入
    # 我們移除了原本那兩行 N = ... 和 M = ... 的程式碼
    # 因為 CompressAI 會自己根據 state_dict 的內容去調整模型大小
    try:
        model = architectures[arch].from_state_dict(state_dict)
    except Exception as e:
        print(f"⚠️ 自動載入失敗，嘗試使用預設參數初始化... 錯誤: {e}")
        # 如果失敗，嘗試先建立模型再載入 (Fallback)
        model = architectures[arch]()
        model.load_state_dict(state_dict)

    return model.eval()

# [--- 新增的 Helper 函式：放在 eval_model 上面 ---]
stats_dict = {}  # 用來存所有層的 min/max

def get_activation_hook(name):
    """這是一個 Hook 函式，當資料流過某一層時會自動被呼叫"""
    def hook(model, input, output):
        # output 就是該層算出來的結果 (Tensor)
        # 我們只關心 min 和 max
        current_min = output.min().item()
        current_max = output.max().item()

        if name not in stats_dict:
            # 第一次遇到這一層，直接初始化
            stats_dict[name] = {"min": current_min, "max": current_max}
        else:
            # 之後遇到，就要比較並更新全域的 min/max
            stats_dict[name]["min"] = min(stats_dict[name]["min"], current_min)
            stats_dict[name]["max"] = max(stats_dict[name]["max"], current_max)
    return hook

# [--- MODIFIED 4 ---]
# 加入 dataset_type 參數
# [--- 修改原本的 eval_model ---]
def eval_model(model, filepaths, dataset_type, entropy_estimation=False, half=False, save=False, output_dir=None):
    device = next(model.parameters()).device
    metrics = defaultdict(float)

    # ==========================================
    # [步驟 1]：註冊 Hook (在開始跑迴圈之前)
    # ==========================================
    print("正在註冊 Hooks 以統計輸出範圍...")

    # 這裡可以決定要監聽哪些層，通常我們監聽 Conv2d, Linear, PixelShuffle
    # 或是你要監聽所有層也可以
    hooks = []
    for name, module in model.named_modules():
        # 如果你是為了 Vitis AI 量化，通常最重要的是 Conv2d 和 Linear
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.PixelShuffle, nn.ConvTranspose2d)):
            # 註冊 hook，並把這層的名子傳進去
            h = module.register_forward_hook(get_activation_hook(name))
            hooks.append(h)

    # ==========================================
    # [步驟 2]：正常的推論迴圈 (不用改)
    # ==========================================
    # 只要跑這個迴圈，資料流過模型，上面的 hook 就會自動記錄數值
    print(f"開始推論 {len(filepaths)} 張圖片並統計數值...")

    for i, f in enumerate(filepaths):
        x = read_image(f, dataset_type).to(device)

        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, save, output_dir, f)
        else:
            rv = inference_entropy_estimation(model, x)

        # 顯示進度
        if (i + 1) % 10 == 0:
            print(f"已處理 {i + 1}/{len(filepaths)}...")

        for k, v in rv.items():
            metrics[k] += v

    # ==========================================
    # [步驟 3]：移除 Hooks 並印出結果
    # ==========================================
    for h in hooks:
        h.remove()  # 養成好習慣，用完移除

    print("\n" + "=" * 50)
    print("每一層的輸出範圍統計 (Min / Max)")
    print("=" * 50)

    # 排序並列印
    for layer_name in sorted(stats_dict.keys()):
        min_val = stats_dict[layer_name]['min']
        max_val = stats_dict[layer_name]['max']
        print(f"Layer: {layer_name:<40} | Range: [{min_val: .4f}, {max_val: .4f}]")

    print("=" * 50 + "\n")

    # 原本的 metrics 計算
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parent_parser.add_argument("dataset", type=str, help="dataset path (file or directory)")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )

    # [--- MODIFIED 6 ---]
    # 加入 dataset-type 參數
    parent_parser.add_argument(
        "--dataset-type",
        choices=["normal", "tif"],
        default="normal",
        help="Type of dataset: 'normal' for 8-bit RGB, 'tif' for 16-bit satellite TIFs (default: %(default)s)",
    )
    # [--- MODIFIED END ---]

    parent_parser.add_argument(
        "--save", action="store_true", help="Save reconstructed images"
    )
    parent_parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="reconstructed",
        help="Directory to save reconstructed images (default: %(default)s)",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )

    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    # [--- MODIFIED 7 ---]
    # 根據 dataset_type 決定要抓取哪些檔案
    if os.path.isdir(args.dataset):
        if args.dataset_type == "tif":
            filepaths = collect_tif_images(args.dataset)
        else:
            filepaths = collect_images(args.dataset)
    elif os.path.isfile(args.dataset):
        filepaths = [args.dataset]
    else:
        print(f"Error: dataset path {args.dataset} is not a valid file or directory.", file=sys.stderr)
        raise SystemExit(1)

    if len(filepaths) == 0:
        print(f"Error: no images found in {args.dataset} (type: {args.dataset_type}).", file=sys.stderr)
        raise SystemExit(1)
    # [--- MODIFIED END ---]

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

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

        if args.source == "checkpoint":
            model.update(force=True)

        # [--- MODIFIED 8 ---]
        # 傳入 args.dataset_type
        metrics = eval_model(
            model,
            filepaths,
            args.dataset_type,  # <-- 傳入
            args.entropy_estimation,
            args.half,
            args.save,
            args.output_dir
        )
        # [--- MODIFIED END ---]

        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.architecture,
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])