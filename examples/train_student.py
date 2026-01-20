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
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from compressai.datasets import ImageFolder
from compressai.models import TIC, TIC_Student
import wandb

class KnowledgeDistillationLoss(nn.Module):
    """
    根據論文 arXiv:2509.10366v1 實作的知識蒸餾損失函數。
    採用 Figure 6 中 student_5_2 的權重配置。
    """

    def __init__(self, lmbda=1e-2, w_latent=0.3, w_recon=0.3, w_rd=0.4, roi_factor=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none') 
        self.mse_mean = nn.MSELoss()            
        
        self.lmbda = lmbda
        self.roi_factor = roi_factor
        
        self.w_latent = w_latent
        self.w_recon = w_recon
        self.w_rd = w_rd

    def forward(self, student_out, teacher_out, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        # 1. Rate-Distortion Loss
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in student_out["likelihoods"].values()
        )

        mse_pixel_wise = self.mse(student_out["x_hat"], target)
        
        if self.roi_factor == 1.0:
            weighted_mse = mse_pixel_wise.mean()
        else:
            roi_map = torch.ones_like(target) # 簡化邏輯
            weighted_mse = (mse_pixel_wise * roi_map).mean()

        out["mse_loss"] = weighted_mse
        rd_loss = self.lmbda * 255**2 * weighted_mse + out["bpp_loss"]

        # 2. Knowledge Distillation Loss
        latent_loss = self.mse_mean(student_out["y_hat"], teacher_out["y_hat"])
        recon_distill_loss = self.mse_mean(student_out["x_hat"], teacher_out["x_hat"])

        # 3. Total Loss
        total_loss = (self.w_latent * latent_loss + 
                      self.w_recon * recon_distill_loss + 
                      self.w_rd * rd_loss)

        out["loss"] = total_loss
        out["rd_loss"] = rd_loss
        out["latent_loss"] = latent_loss
        out["recon_distill_loss"] = recon_distill_loss
        out["mse_loss_raw"] = self.mse_mean(student_out["x_hat"], target)

        return out

class AverageMeter:
    def __init__(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def configure_optimizers(net, args):
    parameters = {
        n for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def train_one_epoch(student, teacher, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm):
    student.train()
    teacher.eval()
    
    device = next(student.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with torch.no_grad():
            teacher_out = teacher(d)

        student_out = student(d)

        out_criterion = criterion(student_out, teacher_out, d)
        
        out_criterion["loss"].backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = student.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 1000 == 0:
            total_loss = out_criterion["loss"].item()
            mse_raw = out_criterion["mse_loss_raw"].item()
            psnr_val = 10 * math.log10(1.0 / (mse_raw + 1e-10))
            
            logging.info(
                f"Epoch [{epoch}][{i}/{len(train_dataloader)}] "
                f"Loss: {total_loss:.4f} | "
                f"RD: {out_criterion['rd_loss'].item():.4f} | "
                f"Latent: {out_criterion['latent_loss'].item():.5f} | "
                f"PSNR: {psnr_val:.2f}"
            )
            
            wandb.log({
                "train_loss": total_loss,
                "train_psnr": psnr_val,
                "train_latent_distill": out_criterion['latent_loss'].item(),
                "train_recon_distill": out_criterion['recon_distill_loss'].item()
            })

def eval_epoch(epoch, dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss_raw = AverageMeter()
    psnr_meter = AverageMeter()

    with torch.no_grad():
        for d in dataloader:
            d = d.to(device)
            out_net = model(d)
            
            N, _, H, W = d.size()
            num_pixels = N * H * W
            
            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out_net["likelihoods"].values()
            )
            
            mse = criterion.mse_mean(out_net["x_hat"], d)
            total = criterion.lmbda * 255**2 * mse + bpp

            loss.update(total.item())
            bpp_loss.update(bpp.item())
            mse_loss_raw.update(mse.item())
            
            psnr_val = 10 * math.log10(1.0 / (mse.item() + 1e-10))
            psnr_meter.update(psnr_val)

    logging.info(
        f"Val Epoch {epoch}: Loss: {loss.avg:.4f} | PSNR: {psnr_meter.avg:.2f} | Bpp: {bpp_loss.avg:.4f}"
    )
    
    wandb.log({
        "val_loss": loss.avg,
        "val_psnr": psnr_meter.avg,
        "val_bpp": bpp_loss.avg
    })

    return loss.avg

def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir + filename)
    if is_best:
        shutil.copyfile(base_dir + filename, base_dir + "checkpoint_best.pth.tar")

def parse_args(argv):
    parser = argparse.ArgumentParser(description="TIC Knowledge Distillation Training")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path")
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to pre-trained teacher checkpoint")
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Dataloaders threads")
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    
    # ★★★ 補回這行，預設為 True ★★★
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), type=str)
    
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    
    wandb.init(project="tic-distillation-student64", name=args.name, config=vars(args))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    log_dir = f'./experiments/{args.name}'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

    teacher = TIC(N=128, M=192)
    student = TIC_Student() 

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    teacher = teacher.to(device)
    student = student.to(device)

    if os.path.isfile(args.teacher_checkpoint):
        logging.info(f"Loading teacher model from {args.teacher_checkpoint}")
        checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        teacher.load_state_dict(state_dict)
    else:
        logging.error(f"Teacher checkpoint not found at {args.teacher_checkpoint}")
        sys.exit(1)

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    optimizer, aux_optimizer = configure_optimizers(student, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 90], gamma=0.1)

    criterion = KnowledgeDistillationLoss(
        lmbda=args.lmbda, 
        roi_factor=1.0, 
        w_latent=0.3, 
        w_recon=0.3, 
        w_rd=0.4
    )

    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    val_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    logging.info(f"Start training... Teacher(N=128) -> Student(N=64)")
    logging.info(f"Workers: {args.num_workers}, Batch: {args.batch_size}")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        train_one_epoch(student, teacher, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm)
        
        val_loss = eval_epoch(epoch, val_dataloader, student, criterion)
        
        lr_scheduler.step()

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": student.state_dict(),
                    "loss": val_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                },
                is_best,
                log_dir
            )

    wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])