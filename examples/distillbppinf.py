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

# 載入 compressai 相關模組
import compressai
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# --- 輔助函式 ---

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def collect_images(rootpath: str) -> List[str]:
    return [os.path.join(rootpath, f) for f in os.listdir(rootpath) if
            os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    if mse == 0:
        return float('inf')
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


# --- 學生模型定義 ---

class SimpleConvStudentModel(nn.Module):
    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = N
        self.M = M
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            conv(N, M, kernel_size=3, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=3, stride=2), nn.BatchNorm2d(N), nn.ReLU(inplace=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2),
        )
        self.h_s = nn.Sequential(
            deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            conv(N, M * 2, kernel_size=3, stride=1),
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
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def update(self, force=False):
        SCALES_MIN = 0.11
        SCALES_MAX = 256
        SCALES_LEVELS = 64
        scale_table = torch.exp(torch.linspace(math.log(SCALES_MIN), math.log(SCALES_MAX), SCALES_LEVELS))
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        self.entropy_bottleneck.update(force=force)

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict,
        )
        update_registered_buffers(
            self.entropy_bottleneck, "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"], state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.9.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net


# --- 為學生模型編寫的 compress 和 decompress 函式 ---

def compress_model(model, x):
    y = model.g_a(x)
    z = model.h_a(y)
    z_strings = model.entropy_bottleneck.compress(z)
    z_hat = model.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

    gaussian_params = model.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)

    indexes = model.gaussian_conditional.build_indexes(scales_hat)
    y_q = torch.round(y - means_hat)

    encoder = BufferedRansEncoder()
    y_q_list = y_q.int().flatten().tolist()
    indexes_list = indexes.int().flatten().tolist()

    encoder.encode_with_indexes(
        y_q_list, indexes_list,
        model.gaussian_conditional._quantized_cdf.tolist(),
        model.gaussian_conditional._cdf_length.tolist(),
        model.gaussian_conditional._offset.tolist(),
    )
    y_string = encoder.flush()
    return {"strings": [y_string, z_strings], "shape": z.size()[-2:]}


def decompress_model(model, strings, shape):
    z_hat = model.entropy_bottleneck.decompress(strings[1], shape)

    gaussian_params = model.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)

    indexes = model.gaussian_conditional.build_indexes(scales_hat)

    decoder = RansDecoder()
    decoder.set_stream(strings[0])

    y_q = decoder.decode_stream(
        indexes.int().flatten().tolist(),
        model.gaussian_conditional._quantized_cdf.tolist(),
        model.gaussian_conditional._cdf_length.tolist(),
        model.gaussian_conditional._offset.tolist(),
    )

    y_height = z_hat.size(2) * 4
    y_width = z_hat.size(3) * 4
    y_q = torch.tensor(y_q).reshape(1, model.M, y_height, y_width).to(z_hat.device).float()

    y_hat = model.gaussian_conditional._dequantize(y_q, means_hat)

    x_hat = model.g_s(y_hat).clamp_(0, 1)
    return {"x_hat": x_hat}


# --- 推論函式 ---

@torch.no_grad()
def inference(model, x):
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
    out_enc = compress_model(model, x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = decompress_model(model, out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom))

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = (len(out_enc["strings"][0]) + sum(len(s) for s in out_enc["strings"][1])) * 8.0 / num_pixels

    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
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
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in out_net["likelihoods"].values())

    return {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,
        "decoding_time": elapsed_time / 2.0,
    }


def eval_model(model, filepaths, entropy_estimation=False):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for i, f in enumerate(filepaths):
        print(f"Processing image {i + 1}/{len(filepaths)}: {os.path.basename(f)}", end='\r')
        x = read_image(f).to(device)
        if entropy_estimation:
            rv = inference_entropy_estimation(model, x)
        else:
            rv = inference(model, x)

        for k, v in rv.items():
            metrics[k] += v

    print()
    for k, v in metrics.items():
        metrics[k] /= len(filepaths)

    return metrics


def setup_args():
    parser = argparse.ArgumentParser(description="評估學生模型在影像資料集上的表現。")
    parser.add_argument("dataset", type=str, help="資料集路徑")
    parser.add_argument("-p", "--path", dest="path", type=str, required=True, help="模型權重 (.pth.tar) 的路徑")
    parser.add_argument("--cuda", action="store_true", help="啟用 CUDA")
    parser.add_argument("--entropy-estimation", action="store_true", help="使用熵估計（不進行實際編解碼）")
    return parser


def main(argv):
    args = setup_args().parse_args(argv)
    filepaths = collect_images(args.dataset)
    if not filepaths:
        print("錯誤：在指定目錄中找不到任何圖片。", file=sys.stderr)
        sys.exit(1)

    compressai.set_entropy_coder(compressai.available_entropy_coders()[0])

    print(f"從 {args.path} 載入學生模型權重...")
    checkpoint = torch.load(args.path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = SimpleConvStudentModel.from_state_dict(state_dict)
    model.eval()

    print("更新模型的 CDF 表格...")
    model.update(force=True)

    if args.cuda and torch.cuda.is_available():
        model.to("cuda")

    results = eval_model(model, filepaths, args.entropy_estimation)

    print("\n--- 評估結果 ---")
    for k, v in sorted(results.items()):
        print(f"{k:15s}: {v:.4f}")

    output = {"name": "SimpleConvStudentModel", "results": results}
    print("\n--- JSON 格式輸出 ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])