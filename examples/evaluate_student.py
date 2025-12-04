# 檔案名稱: evaluate_student.py (請將以下所有內容複製貼上)

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

# ✨ 1. 從你的 onlyconvulution.py 匯入學生模型 Class
from onlyconvulution import SimpleConvStudentModel

import compressai
from compressai.zoo import image_models as pretrained_models
from compressai.zoo import load_state_dict
from compressai.zoo.image import model_architectures as architectures

# ✨ 2. 幫你的學生模型註冊一個代號 "simple_conv_student"
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


@torch.no_grad()
def inference(model, x, save=False, output_dir=None, filepath=None):
    x = x.unsqueeze(0)
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
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels  #拿compress產生後的數據算bpp

    if save and output_dir is not None and filepath is not None:
        reconstructed_img = transforms.ToPILImage()(out_dec["x_hat"].squeeze().cpu())
        basename = os.path.splitext(os.path.basename(filepath))[0]
        output_filepath = os.path.join(output_dir, f"{basename}_reconstructed.png")
        reconstructed_img.save(output_filepath)

    return {"psnr": psnr(x, out_dec["x_hat"]), "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
            "bpp": bpp, "encoding_time": enc_time, "decoding_time": dec_time}


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)
    start = time.time()
    out_net = model.forward(x)
    elapsed_time = time.time() - start
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_net["likelihoods"].values())
    return {"psnr": psnr(x, out_net["x_hat"]), "bpp": bpp.item(), "encoding_time": elapsed_time / 2.0,
            "decoding_time": elapsed_time / 2.0}


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](quality=quality, metric=metric, pretrained=True).eval()


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    return architectures[arch].from_state_dict(state_dict).eval()


def eval_model(model, filepaths, entropy_estimation=False, half=False, save=False, output_dir=None):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in filepaths:
        x = read_image(f).to(device)
        if not entropy_estimation:
            if half:
                model = model.half()
                x = x.half()
            rv = inference(model, x, save, output_dir, f)
        else:
            rv = inference_entropy_estimation(model, x)
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument("-a", "--architecture", type=str, required=True,
                               help="model architecture")  # ✨ 3. 移除了 choices 限制
    parent_parser.add_argument("-c", "--entropy-coder", choices=compressai.available_entropy_coders(),
                               default=compressai.available_entropy_coders()[0],
                               help="entropy coder (default: %(default)s)")
    parent_parser.add_argument("--cuda", action="store_true", help="enable CUDA")
    parent_parser.add_argument("--half", action="store_true", help="convert model to half floating point (fp16)")
    parent_parser.add_argument("--entropy-estimation", action="store_true",
                               help="use evaluated entropy estimation (no entropy coding)")
    parent_parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parent_parser.add_argument("--save", action="store_true", help="Save reconstructed images")
    parent_parser.add_argument("-o", "--output_dir", type=str, default="reconstructed",
                               help="Directory to save reconstructed images (default: %(default)s)")

    parser = argparse.ArgumentParser(description="Evaluate a model on an image dataset.", add_help=True)
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
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

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

        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.save, args.output_dir)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = ("entropy estimation" if args.entropy_estimation else args.entropy_coder)
    output = {"name": args.architecture, "description": f"Inference ({description})", "results": results}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

    # python
    # evaluate_student.py
    # checkpoint
    # C:\Users\Matt\Desktop\0
    # 922 - a
    # simple_conv_student - p
    # "C:\Users\Matt\PycharmProjects\PythonProject1\TIC\distilled\2025-09-19_173153_from_tic_q3\checkpoint_best_loss.pth.tar" - -cuda - -save - o
    # reconstructed_outputs