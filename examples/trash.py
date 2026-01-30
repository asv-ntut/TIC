import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 新增 rasterio
try:
    import rasterio
except ImportError:
    rasterio = None

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers

# --- Constants and Utility Functions ---
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
TIF_EXTENSIONS = {'.tif', '.tiff'}


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


try:
    import wandb
except ImportError:
    wandb = None


# --- Custom Dataset Class for 16-bit TIF ---
class TifImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train"):
        if rasterio is None:
            raise ImportError("Please install rasterio to use TifImageFolder: pip install rasterio")

        splitdir = Path(root) / split
        if not splitdir.is_dir():
            if Path(root).is_dir():
                print(f"Warning: '{split}' subdirectory not found. Scanning '{root}' directly.")
                splitdir = Path(root)
            else:
                raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file() and f.suffix.lower() in TIF_EXTENSIONS]

        if not self.samples:
            print(f"Warning (TifImageFolder): No .tif / .tiff files found in {splitdir}. Check your dataset path.")

        self.transform = transform

        # 衛星影像前處理設定
        self.RGB_BANDS = [3, 2, 1]  # B4 (Red), B3 (Green), B2 (Blue)
        self.CLIP_MIN = 0.0
        self.CLIP_MAX = 10000.0  # 裁切最大值 (16-bit)
        self.SCALE = 10000.0  # 縮放分母 (轉為比例)

    def __getitem__(self, index):
        filepath = self.samples[index]
        try:
            with rasterio.open(filepath) as src:
                raw_data = src.read().astype(np.float32)

            # [加速優化 1] 移除 NaN 檢查 (假設訓練資料是正常的)
            # 這會大幅提升讀取速度，因為不需要 CPU 掃描每個像素
            # if np.isnan(raw_data).any():
            #     for i in range(raw_data.shape[0]):
            #         band = raw_data[i]
            #         if np.isnan(band).any():
            #             band_mean = np.nanmean(band)
            #             band[np.isnan(band)] = band_mean
            #             raw_data[i] = band

            if raw_data.shape[0] > max(self.RGB_BANDS):
                rgb_data = raw_data[self.RGB_BANDS, :, :]
            else:
                rgb_data = raw_data[:3, :, :]

            clipped_data = np.clip(rgb_data, self.CLIP_MIN, self.CLIP_MAX)
            normalized_data = clipped_data / self.SCALE
            img_tensor = torch.from_numpy(normalized_data.copy())

            if self.transform:
                img_tensor = self.transform(img_tensor)

            return img_tensor

        except Exception as e:
            print(f"Error loading TIF image {filepath}: {e}")
            return torch.zeros((3, 256, 256))

    def __len__(self):
        return len(self.samples)


def deconv_pixelshuffle(in_channels, out_channels, kernel_size=5, stride=2):
    internal_channels = out_channels * (stride ** 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        nn.PixelShuffle(upscale_factor=stride)
    )


# --- Model Definition (Standard) ---
class SimpleConvModel(nn.Module):
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
        update_registered_buffers(self.entropy_bottleneck, "entropy_bottleneck",
                                  ["_quantized_cdf", "_offset", "_cdf_length"], state_dict)
        update_registered_buffers(self.gaussian_conditional, "gaussian_conditional",
                                  ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        super().load_state_dict(state_dict, strict=strict)


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        y_likelihoods = output["likelihoods"].get("y")
        z_likelihoods = output["likelihoods"].get("z")
        bpp_y = torch.tensor(0.0, device=target.device)
        bpp_z = torch.tensor(0.0, device=target.device)
        if y_likelihoods is not None and y_likelihoods.numel() > 0:
            bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
        if z_likelihoods is not None and z_likelihoods.numel() > 0:
            bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * num_pixels)

        out["bpp_loss"] = bpp_y + bpp_z
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val,
               n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count if self.count != 0 else 0


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            try:
                return getattr(self.module, key)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' object or its module has no attribute '{key}'")


# --- Setup Functions ---
def init(args):
    base_dir = f'./checkpoints/{args.name}_q{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)
    logging.info(f'Logging file is {log_dir}')


def configure_optimizers(net, args):
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=args.learning_rate)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer


# --- Training and Evaluation Functions ---
def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, use_wandb):
    model.train()
    device = next(model.parameters()).device
    loss_meter = AverageMeter()
    bpp_loss_meter = AverageMeter()
    mse_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)
        out_criterion = criterion(out_net, d)
        
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        loss_meter.update(out_criterion["loss"].item())
        bpp_loss_meter.update(out_criterion["bpp_loss"].item())
        mse_loss_meter.update(out_criterion["mse_loss"].item())
        aux_loss_meter.update(aux_loss.item())

        if i % 100 == 0:
            logging.info(
                f"Train ep {epoch + 1} | [{i * len(d)}/{len(train_dataloader.dataset)}] | "
                f"Loss: {loss_meter.val:.4f} | MSE: {mse_loss_meter.val:.5f} | Bpp: {bpp_loss_meter.val:.4f} | Aux: {aux_loss_meter.val:.2f}"
            )

        if use_wandb and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            wandb.log({
                "train/step": step,
                "train/loss": loss_meter.val,
                "train/bpp_loss": bpp_loss_meter.val,
                "train/mse_loss": mse_loss_meter.val,
                "train/aux_loss": aux_loss_meter.val,
                "train/lr": optimizer.param_groups[0]['lr']
            })

    return loss_meter.avg


def eval_epoch(epoch, dataloader, model, criterion, use_wandb):
    model.eval()
    device = next(model.parameters()).device
    loss_meter = AverageMeter()
    bpp_loss_meter = AverageMeter()
    mse_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for d in dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss_meter.update(model.aux_loss())
            bpp_loss_meter.update(out_criterion["bpp_loss"])
            loss_meter.update(out_criterion["loss"])
            mse_val = out_criterion["mse_loss"].item()
            mse_loss_meter.update(mse_val)
            if mse_val > 0:
                psnr_meter.update(10 * math.log10(1. / mse_val))

    log_prefix = "Final Test" if isinstance(epoch, str) else f"Validation epoch {epoch + 1}"
    logging.info(
        f"{log_prefix}: Loss:{loss_meter.avg:.4f} | MSE:{mse_loss_meter.avg:.6f} | PSNR:{psnr_meter.avg:.3f} | Bpp:{bpp_loss_meter.avg:.4f} | Aux:{aux_loss_meter.avg:.4f}")

    if use_wandb and not isinstance(epoch, str):
        wandb.log({
            "val/epoch": epoch + 1, 
            "val/loss": loss_meter.avg, 
            "val/mse_loss": mse_loss_meter.avg,
            "val/psnr": psnr_meter.avg, 
            "val/bpp_loss": bpp_loss_meter.avg, 
            "val/aux_loss": aux_loss_meter.avg
        })
    return loss_meter.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    path = os.path.join(base_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))


# --- Argument Parsing ---
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Standard VAE Training (No Distillation) for 16-bit TIF.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="satellite-compression", help="W&B project name.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path (folder with TIFs).")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Epochs.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3, help="Aux LR.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size.")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Crop size.")
    parser.add_argument("--clip_max_norm", type=float, default=1.0, help="Gradient clipping.")
    
    # Lambda 控制品質：值越大，畫質越好但檔案越大；值越小，壓縮率越高但畫質越差
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.01, help="RD loss lambda (control quality).")
    parser.add_argument("--quality-level", type=int, default=3, help="Quality index (for logging purpose).")
    
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Workers.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda.")
    parser.add_argument("--save", action="store_true", default=True, help="Save checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint.")
    parser.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d_%H%M%S'), help="Experiment name.")
    
    # Hidden params to define model size (N, M)
    parser.add_argument("--N", type=int, default=128, help="Number of channels in latent space.")
    parser.add_argument("--M", type=int, default=192, help="Number of channels in hyperprior.")
    
    args = parser.parse_args(argv)
    return args


# --- Main Function ---
def main(argv):
    args = parse_args(argv)
    base_dir = init(args)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    use_wandb = args.wandb
    if use_wandb:
        if wandb is None: raise ImportError("Please install wandb")
        wandb.init(project=args.wandb_project, name=args.name, config=vars(args))

    setup_logger(os.path.join(base_dir, 'train.log'))
    logging.info(f"Training run: {args.name}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info(f"Training on {device}")

    # Dataset setup
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size)])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size)])
    try:
        logging.info(f"Loading TIF dataset from {args.dataset}...")
        train_dataset = TifImageFolder(args.dataset, split="train", transform=train_transforms)
        val_dataset = TifImageFolder(args.dataset, split="val", transform=test_transforms)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)

    # [加速優化 2] 啟用 persistent_workers (當 workers > 0 時)
    # 這會讓 DataLoader 在 epoch 之間保持 workers 存活，不用重新啟動
    use_persistent = (args.num_workers > 0)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                  pin_memory=(device == "cuda"), persistent_workers=use_persistent)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=(device == "cuda"), persistent_workers=use_persistent)

    # Model Setup (No Teacher)
    net = SimpleConvModel(N=args.N, M=args.M).to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    # Resume logic
    last_epoch = 0
    if args.checkpoint:
        logging.info(f"Resuming: {args.checkpoint}")
        cp = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(cp["state_dict"], strict=False)
        if "optimizer" in cp: optimizer.load_state_dict(cp["optimizer"])
        if "aux_optimizer" in cp: aux_optimizer.load_state_dict(cp["aux_optimizer"])
        last_epoch = cp.get("epoch", -1) + 1

    best_loss = float("inf")

    try:
        for epoch in range(last_epoch, args.epochs):
            logging.info(f"====== Epoch {epoch + 1}/{args.epochs} ======")

            train_loss = train_one_epoch(net, criterion, train_dataloader, optimizer,
                                        aux_optimizer, epoch, args.clip_max_norm, use_wandb)

            val_loss = eval_epoch(epoch, val_dataloader, net, criterion, use_wandb)

            lr_scheduler.step(val_loss)

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if args.save:
                sd = net.module.state_dict() if isinstance(net, CustomDataParallel) else net.state_dict()
                save_checkpoint({"epoch": epoch, "state_dict": sd, "optimizer": optimizer.state_dict(),
                                 "aux_optimizer": aux_optimizer.state_dict()}, is_best, base_dir)

    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        if use_wandb: wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])