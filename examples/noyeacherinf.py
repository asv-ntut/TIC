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

# ===================================================================
# ✨ 1. 從您的獨立訓練腳本中，匯入 SimpleConv 模型 Class
#    (請確認檔名 'noteacher_cnn' 是否正確)
# ===================================================================
from noteachercnn import SimpleConv

import compressai
from compressai.zoo.image import model_architectures as architectures

# ===================================================================
# ✨ 2. 幫您的 SimpleConv 模型註冊一個新的代號 "simple_conv"
# ===================================================================
architectures["simple_conv"] = SimpleConv

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# --- 底下的輔助函式與原版 evaluate_student.py 完全相同 ---

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


@torch.no_grad()
def inference(model, x, save=False, output_dir=None, filepath=None):
    device = next(model.parameters()).device
    x = x.to(device).unsqueeze(0)
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

    if save and output_dir is not None and filepath is not None:
        reconstructed_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze().cpu())
        basename = os.path.splitext(os.path.basename(filepath))[0]
        output_filepath = os.path.join(output_dir, f"{basename}_reconstructed.png")
        reconstructed_img.save(output_filepath)

    return {"psnr": psnr(x, out_dec["x_hat"]), "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
            "bpp": bpp, "encoding_time": enc_time, "decoding_time": dec_time}


def eval_model(model, filepaths, save=False, output_dir=None):
    metrics = defaultdict(float)
    for f in filepaths:
        x = read_image(f)
        rv = inference(model, x, save, output_dir, f)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


# ===================================================================
# ✨ 3. 簡化命令列參數，只保留從 checkpoint 載入的選項
# ===================================================================
def setup_args():
    parser = argparse.ArgumentParser(description="Evaluate a custom model on an image dataset.")
    parser.add_argument("dataset", type=str, help="Path to the dataset")
    parser.add_argument("-a", "--architecture", type=str, required=True,
                        help="Model architecture name (e.g., 'simple_conv')")
    parser.add_argument("-p", "--path", dest="path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("-c", "--entropy-coder", choices=compressai.available_entropy_coders(),
                        default=compressai.available_entropy_coders()[0], help="Entropy coder (default: %(default)s)")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--save", action="store_true", help="Save reconstructed images")
    parser.add_argument("-o", "--output_dir", type=str, default="reconstructed",
                        help="Directory to save reconstructed images (default: %(default)s)")
    return parser


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)

    filepaths = collect_images(args.dataset)
    if not filepaths:
        print("Error: no images found in directory.", file=sys.stderr)
        sys.exit(1)

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)

    compressai.set_entropy_coder(args.entropy_coder)

    # 載入模型
    print(f"Loading model from checkpoint: {args.path}")
    state_dict = torch.load(args.path)
    state_dict = state_dict.get("state_dict", state_dict)
    model = architectures[args.architecture].from_state_dict(state_dict).eval()

    if args.cuda and torch.cuda.is_available():
        model = model.to("cuda")

    # 在推論前務必執行 update
    model.update(force=True)

    # 評估模型
    print("Evaluating model...")
    metrics = eval_model(model, filepaths, args.save, args.output_dir)

    # 輸出結果
    results = {k: [v] for k, v in metrics.items()}
    output = {
        "name": args.architecture,
        "description": f"Inference ({args.entropy_coder})",
        "results": results
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])