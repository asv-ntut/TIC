import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# ---【修改】--- 不再需要從 zoo 載入教師模型
from compressai.zoo import image_models
from compressai.datasets import ImageFolder
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder

# --- 模型定義部分 (SimpleConvStudentModel) 完全不變，所以直接貼上 ---

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class SimpleConv(nn.Module):
    def __init__(self, N=128, M=196):
        super().__init__()
        self.N = N
        self.M = M
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            conv(N, M, kernel_size=3, stride=2),
        )
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
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

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = False
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                updated |= m.update(force=force)
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.entropy_bottleneck, "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"], state_dict)
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
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


# --- 訓練相關的 class 和函數 ---

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in
                              output["likelihoods"].values())
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        return out


class AverageMeter:
    def __init__(self):
        self.val = 0;
        self.avg = 0;
        self.sum = 0;
        self.count = 0

    def update(self, val, n=1):
        self.val = val;
        self.sum += val * n;
        self.count += n;
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    # ---【修改】--- 更改儲存目錄的名稱，不再依賴教師模型
    base_dir = f'./standalone/SimpleConvModel_lambda_{args.lmbda}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)
    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)
    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=args.learning_rate)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=args.aux_learning_rate)
    return optimizer, aux_optimizer


# ---【修改】--- 核心修改區域：train_one_epoch 函數
def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm):
    model.train()  # 將模型設為訓練模式
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # 直接計算模型輸出
        out_net = model(d)

        # 計算損失 (只有任務損失)
        out_criterion = criterion(out_net, d)
        total_loss = out_criterion["loss"]  # 總損失就是任務損失

        total_loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            # 修改 log 訊息，移除蒸餾損失的部分
            logging.info(
                f"Train epoch {epoch + 1} | [{i * len(d)}/{len(train_dataloader.dataset)}] | "
                f"Loss: {total_loss.item():.4f} | "
                f"MSE loss: {out_criterion['mse_loss'].item():.6f} | "
                f"Bpp loss: {out_criterion['bpp_loss'].item():.4f}"
            )


# eval_epoch 函數無需修改，因為它本來就只評估單一模型
def eval_epoch(epoch, dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    loss, bpp_loss, mse_loss, aux_loss, psnr = [AverageMeter() for _ in range(5)]
    with torch.no_grad():
        for d in dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            mse_val = out_criterion["mse_loss"].item()
            psnr_val = 10 * math.log10(1.0 / mse_val) if mse_val > 0 else float('inf')
            psnr.update(psnr_val)
    log_prefix = "Final Test" if isinstance(epoch, str) else f"Validation epoch {epoch}"
    logging.info(
        f"{log_prefix}: Average losses | Loss: {loss.avg:.4f} | MSE loss: {mse_loss.avg:.6f} | PSNR: {psnr.avg:.3f} | Bpp loss: {bpp_loss.avg:.4f} | Aux loss: {aux_loss.avg:.2f}")
    return loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(base_dir, filename), os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))


# ---【修改】--- 移除所有跟 teacher 和 alpha 相關的參數
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Standalone training script for SimpleConvStudentModel.")
    # -- 保留下來的參數 --
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("-e", "--epochs", default=300, type=int, help="Number of epochs.")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Dataloaders threads.")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-2, help="Bit-rate distortion parameter.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size.")
    parser.add_argument("--aux-learning-rate", default=1e-3, help="Auxiliary loss learning rate.")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256),
                        help="Size of the patches to be cropped.")
    parser.add_argument("--cuda", action="store_true", help="Use cuda.")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk.")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility.")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm.")
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str,
                        help='Result dir name.')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint to resume training.")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)
    if args.seed is not None:
        torch.manual_seed(args.seed);
        random.seed(args.seed)

    setup_logger(os.path.join(base_dir, 'train.log'))
    logging.info(f"Standalone training run: {args.name}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info(f"Training on {device}")

    # Dataset 和 Dataloader 維持不變
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    val_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                  pin_memory=(device == "cuda"))
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=(device == "cuda"))

    # ---【修改】--- 主要修改區域：移除 Teacher Model
    logging.info("Creating model: SimpleConv (Standalone)")
    net = SimpleConv().to(device)

    # ---【修改】--- 整個載入 teacher model 的區塊都刪除了

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:
        logging.info("Resuming training from checkpoint", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info(f"====== Epoch {epoch + 1}/{args.epochs} ======")

        # ---【修改】--- 更新 train_one_epoch 的呼叫方式
        train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm)

        val_loss = eval_epoch(epoch + 1, val_dataloader, net, criterion)
        lr_scheduler.step(val_loss)
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if args.save:
            save_checkpoint({"epoch": epoch, "state_dict": net.state_dict(), "loss": val_loss,
                             "optimizer": optimizer.state_dict(), "aux_optimizer": aux_optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict()}, is_best, base_dir)

    logging.info("Training finished.")
    logging.info("Loading best model from checkpoint for final testing.")
    best_checkpoint_path = os.path.join(base_dir, "checkpoint_best_loss.pth.tar")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        if isinstance(net, CustomDataParallel):
            net.module.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)
        eval_epoch("Final", test_dataloader, net, criterion)
    else:
        logging.warning("Could not find best checkpoint for final testing.")


if __name__ == "__main__":
    main(sys.argv[1:])