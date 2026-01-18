import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from collections import OrderedDict
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
from compressai.zoo import image_models
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

            if np.isnan(raw_data).any():
                for i in range(raw_data.shape[0]):
                    band = raw_data[i]
                    if np.isnan(band).any():
                        band_mean = np.nanmean(band)
                        band[np.isnan(band)] = band_mean
                        raw_data[i] = band

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


# --- Hook Functions ---
teacher_features = {}


def get_teacher_y_hat_hook(module, input, output):
    teacher_features['y'] = output[0]


def get_teacher_z_hat_hook(module, input, output):
    teacher_features['z_hat'] = output[0]


def deconv_pixelshuffle(in_channels, out_channels, kernel_size=5, stride=2):
    internal_channels = out_channels * (stride ** 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
        nn.PixelShuffle(upscale_factor=stride)
    )


# --- Model Definition ---
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

    @classmethod
    def from_state_dict(cls, state_dict):
        try:
            first_g_a_weight_key = 'g_a.0.weight'
            if first_g_a_weight_key not in state_dict:
                g_a_weight_keys = sorted([k for k in state_dict if k.startswith('g_a.') and k.endswith('.weight')])
                if not g_a_weight_keys: raise KeyError("No g_a weights found")
                first_g_a_weight_key = g_a_weight_keys[0]
            N = state_dict[first_g_a_weight_key].size(0)
        except Exception as e:
            print(f"Error inferring N: {e}. Assuming default N=128.")
            N = 128
        try:
            g_a_weight_keys = sorted([k for k in state_dict if k.startswith('g_a.') and k.endswith('.weight')])
            last_g_a_weight_key = g_a_weight_keys[-1]
            M = state_dict[last_g_a_weight_key].size(0)
        except Exception as e:
            print(f"Error inferring M: {e}. Assuming default M=192.")
            M = 192
        net = cls(N, M)
        net.load_state_dict(state_dict, strict=False)
        return net


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {};
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
    base_dir = f'./distilled/{args.name}_from_{args.teacher_model}_q{args.teacher_quality}/'
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
# ✨✨✨ 修改：傳入 loss_weights 並返回平均 Loss ✨✨✨
def train_one_epoch(student_model, teacher_model, criterion, train_dataloader, optimizer, aux_optimizer, epoch,
                    clip_max_norm, alpha, beta, gamma, use_wandb, loss_weights):
    student_model.train()
    teacher_model.eval()
    device = next(student_model.parameters()).device
    task_loss_meter = AverageMeter()
    response_loss_meter = AverageMeter()
    feature_loss_meter = AverageMeter()
    hyper_latent_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with torch.no_grad():
            _ = teacher_model(d)

        student_out = student_model(d)
        task_loss_dict = criterion(student_out, d)
        task_loss = task_loss_dict["loss"]

        # 1. 計算 Response Distillation Loss
        response_distill_loss = torch.tensor(0.0).to(device)
        if alpha > 0:
            with torch.no_grad():
                teacher_out_full = teacher_model(d)
                if isinstance(teacher_out_full, dict) and "x_hat" in teacher_out_full:
                    teacher_x_hat = teacher_out_full["x_hat"].detach()
                else:
                    teacher_x_hat = teacher_out_full.detach()
            response_distill_loss = F.mse_loss(student_out["x_hat"], teacher_x_hat)

        # 2. 計算 Feature Distillation Loss
        feature_distill_loss = torch.tensor(0.0).to(device)
        if beta > 0:
            teacher_y_hat = teacher_features.get('y')
            feature_distill_loss = F.mse_loss(student_out["y_hat"], teacher_y_hat.detach())

        # 3. 計算 Hyper-Latent Distillation Loss
        hyper_latent_distill_loss = torch.tensor(0.0).to(device)
        if gamma > 0:
            teacher_z_hat = teacher_features.get('z_hat')
            hyper_latent_distill_loss = F.mse_loss(student_out["z_hat"], teacher_z_hat.detach())

        # ✨✨✨ 修改：應用動態權重 ✨✨✨
        w_resp = loss_weights['response']
        w_feat = loss_weights['feature']
        w_hyper = loss_weights['hyper']

        total_loss = task_loss + \
                     (alpha * w_resp * response_distill_loss) + \
                     (beta * w_feat * feature_distill_loss) + \
                     (gamma * w_hyper * hyper_latent_distill_loss)

        total_loss.backward()
        if clip_max_norm > 0: torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = student_model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        task_loss_meter.update(task_loss.item())
        if alpha > 0: response_loss_meter.update(response_distill_loss.item())
        if beta > 0: feature_loss_meter.update(feature_distill_loss.item())
        if gamma > 0: hyper_latent_loss_meter.update(hyper_latent_distill_loss.item())
        aux_loss_meter.update(aux_loss.item())

        if i % 100 == 0:
            logging.info(
                f"Train ep {epoch + 1} | [{i * len(d)}/{len(train_dataloader.dataset)}] | Loss:{total_loss.item():.4f} | Task:{task_loss_meter.val:.4f} | Resp:{response_loss_meter.val:.6f} (w={w_resp:.2e}) | Feat:{feature_loss_meter.val:.6f} (w={w_feat:.2e}) | Hyper:{hyper_latent_loss_meter.val:.6f} (w={w_hyper:.2e})")

        if use_wandb and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            log_dict = {"train/step": step, "train/loss": total_loss.item(), "train/task_loss": task_loss.item(),
                        "train/bpp_loss": task_loss_dict["bpp_loss"].item(),
                        "train/mse_loss": task_loss_dict["mse_loss"].item(), "train/aux_loss": aux_loss.item(),
                        "train/lr": optimizer.param_groups[0]['lr']}
            if alpha > 0: log_dict["train/response_loss"] = response_distill_loss.item()
            if beta > 0: log_dict["train/feature_loss"] = feature_distill_loss.item()
            if gamma > 0: log_dict["train/hyper_latent_loss"] = hyper_latent_distill_loss.item()
            wandb.log(log_dict)

    # ✨✨✨ 修改：回傳平均 Loss 供 main 更新權重 ✨✨✨
    return {
        "task": torch.tensor(task_loss_meter.avg, device=device),
        "response": torch.tensor(response_loss_meter.avg, device=device),
        "feature": torch.tensor(feature_loss_meter.avg, device=device),
        "hyper": torch.tensor(hyper_latent_loss_meter.avg, device=device)
    }


def eval_epoch(epoch, dataloader, model, criterion, use_wandb):
    model.eval()
    device = next(model.parameters()).device
    loss_meter = AverageMeter();
    bpp_loss_meter = AverageMeter();
    mse_loss_meter = AverageMeter();
    aux_loss_meter = AverageMeter();
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
            if mse_val > 0: psnr_meter.update(10 * math.log10(1. / mse_val))

    log_prefix = "Final Test" if isinstance(epoch, str) else f"Validation epoch {epoch + 1}"
    logging.info(
        f"{log_prefix}: Loss:{loss_meter.avg:.4f} | MSE:{mse_loss_meter.avg:.6f} | PSNR:{psnr_meter.avg:.3f} | Bpp:{bpp_loss_meter.avg:.4f} | Aux:{aux_loss_meter.avg:.4f}")

    if use_wandb and not isinstance(epoch, str):
        wandb.log({"val/epoch": epoch + 1, "val/loss": loss_meter.avg, "val/mse_loss": mse_loss_meter.avg,
                   "val/psnr": psnr_meter.avg, "val/bpp_loss": bpp_loss_meter.avg, "val/aux_loss": aux_loss_meter.avg})
    return loss_meter.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    path = os.path.join(base_dir, filename)
    torch.save(state, path)
    if is_best: shutil.copyfile(path, os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))


# --- Argument Parsing ---
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Distillation with Dynamic Weight Balancing (TIF supported).")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="student-distill", help="W&B project name.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Response loss base weight.")
    parser.add_argument("--beta", type=float, default=1.0, help="Latent loss base weight.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Hyper-latent loss base weight.")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path (folder with TIFs).")
    parser.add_argument("--teacher-model", type=str, required=True, help="Teacher model name.")
    parser.add_argument("--teacher-quality", type=int, required=True, help="Teacher model quality.")
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Teacher checkpoint path.")
    parser.add_argument("--checkpoint", type=str, help="Student checkpoint to resume.")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Epochs.")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3, help="Aux LR.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size.")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Crop size.")
    parser.add_argument("--clip_max_norm", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.01, help="RD loss lambda.")
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Workers.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda.")
    parser.add_argument("--save", action="store_true", default=True, help="Save checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d_%H%M%S'), help="Experiment name.")
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
    logging.info(f"Distillation run: {args.name}")

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

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                  pin_memory=(device == "cuda"))
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=(device == "cuda"))

    # Teacher Model
    logging.info(f"Loading teacher model '{args.teacher_model}'...")
    teacher_net = image_models[args.teacher_model](quality=args.teacher_quality, pretrained=False).to(device)
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_net.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint, strict=False)
    teacher_net.eval()
    for param in teacher_net.parameters(): param.requires_grad = False

    # Infer N, M
    teacher_N, teacher_M = 128, 192  # Default
    try:
        sd = teacher_net.state_dict()
        if 'g_a.0.weight' in sd: teacher_N = sd['g_a.0.weight'].size(0)
        keys = sorted([k for k in sd.keys() if 'g_a' in k and 'weight' in k])
        if keys: teacher_M = sd[keys[-1]].size(0)
    except Exception:
        pass

    # Student Model
    student_net = SimpleConvStudentModel(N=teacher_N, M=teacher_M).to(device)

    # Register Hooks
    if hasattr(teacher_net, 'gaussian_conditional'):
        teacher_net.gaussian_conditional.register_forward_hook(get_teacher_y_hat_hook)
    if hasattr(teacher_net, 'entropy_bottleneck'):
        teacher_net.entropy_bottleneck.register_forward_hook(get_teacher_z_hat_hook)

    if args.cuda and torch.cuda.device_count() > 1:
        student_net = CustomDataParallel(student_net)
        teacher_net = CustomDataParallel(teacher_net)

    optimizer, aux_optimizer = configure_optimizers(student_net, args)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    # Resume logic
    last_epoch = 0
    if args.checkpoint:
        logging.info(f"Resuming: {args.checkpoint}")
        cp = torch.load(args.checkpoint, map_location=device)
        student_net.load_state_dict(cp["state_dict"], strict=False)
        if "optimizer" in cp: optimizer.load_state_dict(cp["optimizer"])
        if "aux_optimizer" in cp: aux_optimizer.load_state_dict(cp["aux_optimizer"])
        last_epoch = cp.get("epoch", -1) + 1

    # ✨✨✨ 初始化動態權重 (預設為 1.0) ✨✨✨
    loss_weights = {
        "response": torch.tensor(1.0, device=device),
        "feature": torch.tensor(1.0, device=device),
        "hyper": torch.tensor(1.0, device=device)
    }
    logging.info("Initial dynamic weights set to 1.0")

    best_loss = float("inf")
    epsilon = 1e-8  # 防止除以零

    try:
        for epoch in range(last_epoch, args.epochs):
            logging.info(f"====== Epoch {epoch + 1}/{args.epochs} ======")

            # Warmup: 前幾圈不進行特徵蒸餾 (beta=0, gamma=0)
            cur_beta = 0.0 if epoch < args.warmup_epochs else args.beta
            cur_gamma = 0.0 if epoch < args.warmup_epochs else args.gamma

            # 訓練一個 epoch，並獲取回傳的平均 loss
            avg_losses = train_one_epoch(student_net, teacher_net, criterion, train_dataloader, optimizer,
                                         aux_optimizer, epoch, args.clip_max_norm, args.alpha, cur_beta, cur_gamma,
                                         use_wandb, loss_weights)

            # ✨✨✨ 每 10 個 epoch 更新一次動態權重 ✨✨✨
            if (epoch + 1) % 10 == 0 and epoch > 0:
                logging.info(f"Updating dynamic loss weights (End of Epoch {epoch + 1})...")
                avg_task = avg_losses['task']
                avg_resp = avg_losses['response']
                avg_feat = avg_losses['feature']
                avg_hype = avg_losses['hyper']

                # 計算新權重: Target / (Current + eps)
                # 確保分母不為 0，且只有當該 Loss 有被啟用時才更新
                if avg_resp > epsilon: loss_weights['response'] = avg_task / (avg_resp + epsilon)
                if avg_feat > epsilon: loss_weights['feature'] = avg_task / (avg_feat + epsilon)
                if avg_hype > epsilon: loss_weights['hyper'] = avg_task / (avg_hype + epsilon)

                logging.info(
                    f"New Weights -> Resp: {loss_weights['response']:.2e}, Feat: {loss_weights['feature']:.2e}, Hyper: {loss_weights['hyper']:.2e}")

                if use_wandb:
                    wandb.log({
                        "weights/response": loss_weights['response'].item(),
                        "weights/feature": loss_weights['feature'].item(),
                        "weights/hyper": loss_weights['hyper'].item(),
                        "epoch": epoch + 1
                    })

            val_loss = eval_epoch(epoch, val_dataloader, student_net, criterion, use_wandb)

            if epoch >= args.warmup_epochs: lr_scheduler.step()

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if args.save:
                sd = student_net.module.state_dict() if isinstance(student_net,
                                                                   CustomDataParallel) else student_net.state_dict()
                save_checkpoint({"epoch": epoch, "state_dict": sd, "optimizer": optimizer.state_dict(),
                                 "aux_optimizer": aux_optimizer.state_dict()}, is_best, base_dir)

    except KeyboardInterrupt:
        logging.info("Interrupted.")
    finally:
        if use_wandb: wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
