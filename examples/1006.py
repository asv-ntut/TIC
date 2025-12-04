# # 檔案名稱: onlyconvulution.py (最終整合版 - 已修正 Hook 邏輯)
# # 基於您可運作的腳本，透過 Hook 技術實現特徵蒸餾，並整合 W&B

# import argparse
# import math
# import random
# import shutil
# import sys
# import os
# import time
# import logging
# from datetime import datetime

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from torch.utils.data import DataLoader
# from torchvision import transforms

# from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# from compressai.zoo import image_models
# from compressai.datasets import ImageFolder
# from compressai.models.utils import conv, deconv, update_registered_buffers
# from compressai.ans import BufferedRansEncoder, RansDecoder

# # 這個函式是模型 update 時需要的
# SCALES_MIN = 0.11
# SCALES_MAX = 256
# SCALES_LEVELS = 64

# def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
#     """回傳一個用於量化的尺度表 (scale table)"""
#     return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

# try:
#     import wandb
# except ImportError:
#     wandb = None

# # --- 全域變數，用於儲存 Hook 捕獲的老師特徵 ---
# teacher_features = {}


# def get_teacher_features_hook(module, input, output):
#     """這是一個 PyTorch Hook 函式。每次老師模型前向傳播時，
#     它會被自動觸發，並將指定層的輸出儲存到全域變數中。"""
#     teacher_features['y'] = output


# # --- 模型定義 ---

# class BottleneckAdapter(nn.Module):
#     def __init__(self, in_channels, out_channels, expand_ratio=2):
#         super().__init__()
#         hidden_channels = int(in_channels * expand_ratio)
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False), #pointwise
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels, #depthwise
#                       bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False) #pointwise
#         )
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         else:
#             self.shortcut = nn.Identity()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         identity = self.shortcut(x)
#         out = self.block(x)
#         out += identity
#         out = self.relu(out)
#         return out


# class SimpleConvStudentModel(nn.Module):
#     def __init__(self, N=128, M=196, teacher_channels=192):
#         super().__init__()
#         self.N = N
#         self.M = M
#         self.g_a = nn.Sequential(
#             conv(3, N, kernel_size=5, stride=2), nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             conv(N, M, kernel_size=3, stride=2),
#         )
#         self.g_s = nn.Sequential(
#             deconv(M, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             deconv(N, 3, kernel_size=5, stride=2),
#         )
#         self.h_a = nn.Sequential(
#             conv(M, N, kernel_size=3, stride=1), nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             conv(N, N, kernel_size=3, stride=2),
#         )
#         self.h_s = nn.Sequential(
#             deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
#             conv(N, M * 2, kernel_size=3, stride=1),
#         )
#         self.adapter = BottleneckAdapter(in_channels=M, out_channels=teacher_channels)
#         self.entropy_bottleneck = EntropyBottleneck(N)
#         self.gaussian_conditional = GaussianConditional(None)
#         self.apply(self._init_weights)

#     def forward(self, x):
#         y = self.g_a(x)
#         y_adapted = self.adapter(y)
#         z = self.h_a(y)
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
#         x_hat = self.g_s(y_hat)
#         return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}, "y_adapted": y_adapted}

#     def aux_loss(self):
#         return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None: nn.init.constant_(m.bias, 0)

#     def update(self, scale_table=None, force=False):
#         if scale_table is None: scale_table = get_scale_table()
#         self.gaussian_conditional.update_scale_table(scale_table, force=force)
#         return any(m.update(force=force) for m in self.modules() if isinstance(m, EntropyBottleneck))

#     def load_state_dict(self, state_dict, strict=True):
#         update_registered_buffers(self.entropy_bottleneck, "entropy_bottleneck",
#                                   ["_quantized_cdf", "_offset", "_cdf_length"], state_dict)
#         update_registered_buffers(self.gaussian_conditional, "gaussian_conditional",
#                                   ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
#         super().load_state_dict(state_dict, strict=strict)

#     @classmethod
#     def from_state_dict(cls, state_dict):
#         N = state_dict["g_a.0.weight"].size(0)
#         M = state_dict["g_a.6.weight"].size(0)
#         teacher_channels = 192 # Default value
#         if "adapter.shortcut.weight" in state_dict:
#             teacher_channels = state_dict["adapter.shortcut.weight"].size(0)
#         elif "adapter.block.4.weight" in state_dict:
#              teacher_channels = state_dict["adapter.block.4.weight"].size(0)

#         net = cls(N, M, teacher_channels=teacher_channels)
#         net.load_state_dict(state_dict, strict=False)
#         return net

#     def compress(self, x):
#         y = self.g_a(x)
#         z = self.h_a(y)
#         z_strings = self.entropy_bottleneck.compress(z)
#         z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
#         return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

#     def decompress(self, strings, shape):
#         assert isinstance(strings, list) and len(strings) == 2
#         z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
#         gaussian_params = self.h_s(z_hat)
#         scales_hat, means_hat = gaussian_params.chunk(2, 1)
#         indexes = self.gaussian_conditional.build_indexes(scales_hat)
#         y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
#         x_hat = self.g_s(y_hat).clamp_(0, 1)
#         return {"x_hat": x_hat}


# # --- 訓練腳本組件 ---

# class RateDistortionLoss(nn.Module):
#     def __init__(self, lmbda=1e-2):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda

#     def forward(self, output, target):
#         N, _, H, W = target.size()
#         out = {}
#         num_pixels = N * H * W
#         out["bpp_loss"] = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in
#                               output["likelihoods"].values())
#         out["mse_loss"] = self.mse(output["x_hat"], target)
#         out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
#         return out


# class AverageMeter:
#     def __init__(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
#     def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


# class CustomDataParallel(nn.DataParallel):
#     def __getattr__(self, key):
#         try:
#             return super().__getattr__(key)
#         except AttributeError:
#             return getattr(self.module, key)


# def init(args):
#     base_dir = f'./distilled/{args.name}_from_{args.teacher_model}_q{args.teacher_quality}/'
#     os.makedirs(base_dir, exist_ok=True)
#     return base_dir


# def setup_logger(log_dir):
#     log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
#     root_logger = logging.getLogger()
#     root_logger.setLevel(logging.INFO)
#     if root_logger.hasHandlers(): root_logger.handlers.clear()
#     log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
#     log_file_handler.setFormatter(log_formatter)
#     root_logger.addHandler(log_file_handler)
#     log_stream_handler = logging.StreamHandler(sys.stdout)
#     log_stream_handler.setFormatter(log_formatter)
#     root_logger.addHandler(log_stream_handler)
#     logging.info(f'Logging file is {log_dir}')


# def configure_optimizers(net, args):
#     parameters = {n for n, p in net.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
#     aux_parameters = {n for n, p in net.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
#     params_dict = dict(net.named_parameters())
#     optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=args.learning_rate)
#     aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=args.aux_learning_rate)
#     return optimizer, aux_optimizer


# def train_one_epoch(student_model, teacher_model, criterion, train_dataloader, optimizer, aux_optimizer, epoch,
#                     clip_max_norm, alpha, beta, use_wandb):
#     student_model.train()
#     teacher_model.eval()
#     device = next(student_model.parameters()).device
#     for i, d in enumerate(train_dataloader):
#         d = d.to(device)
#         optimizer.zero_grad()
#         aux_optimizer.zero_grad()
#         with torch.no_grad():
#             teacher_out = teacher_model(d)
#         student_out = student_model(d)
#         task_loss_dict = criterion(student_out, d)
#         task_loss = task_loss_dict["loss"]
#         response_distill_loss = F.mse_loss(student_out["x_hat"], teacher_out["x_hat"].detach())

#         teacher_y = teacher_features.get('y')
#         if beta > 0 and teacher_y is None:
#             raise RuntimeError(
#                 "Could not retrieve teacher's feature 'y' via hook. The hook might not have been attached correctly.")

#         feature_distill_loss = F.mse_loss(student_out["y_adapted"], teacher_y.detach()) if beta > 0 else torch.tensor(0.0).to(device)

#         total_loss = task_loss + alpha * response_distill_loss + beta * feature_distill_loss
#         total_loss.backward()
#         if clip_max_norm > 0: torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
#         optimizer.step()

#         aux_loss = student_model.aux_loss()
#         aux_loss.backward()
#         aux_optimizer.step()

#         if i % 100 == 0:
#             logging.info(
#                 f"Train ep {epoch + 1} | [{i * len(d)}/{len(train_dataloader.dataset)}] | Loss:{total_loss.item():.3f} | Task:{task_loss.item():.3f} | Resp:{response_distill_loss.item():.6f} | Feat:{feature_distill_loss.item():.6f}")
#         if use_wandb and i % 100 == 0:
#             step = epoch * len(train_dataloader) + i
#             mse = task_loss_dict["mse_loss"].item()
#             if mse > 0:
#                 psnr_val = 10 * math.log10(1. / mse)
#             else:
#                 psnr_val = float('inf')
#             wandb.log({"train/step": step, "train/loss": total_loss.item(), "train/task_loss": task_loss.item(),
#                        "train/response_loss": response_distill_loss.item(),
#                        "train/feature_loss": feature_distill_loss.item(),
#                        "train/bpp_loss": task_loss_dict["bpp_loss"].item(), "train/mse_loss": mse,
#                        "train/psnr": psnr_val, "train/aux_loss": aux_loss.item(),
#                        "train/lr": optimizer.param_groups[0]['lr']})


# def eval_epoch(epoch, dataloader, model, criterion, use_wandb):
#     model.eval()
#     device = next(model.parameters()).device
#     loss, bpp_loss, mse_loss, aux_loss, psnr = [AverageMeter() for _ in range(5)]
#     with torch.no_grad():
#         for d in dataloader:
#             d = d.to(device)
#             out_net = model(d)
#             out_criterion = criterion(out_net, d)
#             aux_loss.update(model.aux_loss())
#             bpp_loss.update(out_criterion["bpp_loss"])
#             loss.update(out_criterion["loss"])

#             mse_val = out_criterion["mse_loss"].item()
#             mse_loss.update(mse_val)
#             if mse_val > 0:
#                 psnr.update(10 * math.log10(1. / mse_val))

#     log_prefix = "Final Test" if isinstance(epoch, str) else f"Validation epoch {epoch + 1}"
#     logging.info(
#         f"{log_prefix}: Loss:{loss.avg:.4f} | MSE:{mse_loss.avg:.6f} | PSNR:{psnr.avg:.3f} | Bpp:{bpp_loss.avg:.4f} | Aux:{aux_loss.avg:.2f}")

#     if use_wandb and not isinstance(epoch, str):
#         wandb.log({"val/epoch": epoch + 1, "val/loss": loss.avg, "val/mse_loss": mse_loss.avg, "val/psnr": psnr.avg,
#                    "val/bpp_loss": bpp_loss.avg, "val/aux_loss": aux_loss.avg})
#     return loss.avg


# def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
#     torch.save(state, os.path.join(base_dir, filename))
#     if is_best: shutil.copyfile(os.path.join(base_dir, filename),
#                                 os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))


# def parse_args(argv):
#     parser = argparse.ArgumentParser(description="Distillation training script with Hooks and W&B.")
#     # ... (所有參數定義都維持原樣，無需修改) ...
#     parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
#     parser.add_argument("--wandb_project", type=str, default="student-distill", help="W&B project name.")
#     parser.add_argument("--alpha", type=float, default=0.5, help="Weight for response distillation loss.")
#     parser.add_argument("--beta", type=float, default=10000.0,
#                         help="Weight for feature distillation loss. Set to 0 to disable.")
#     parser.add_argument("--dynamic-beta", action="store_true", help="Enable dynamic beta scheduling.")
#     parser.add_argument("--beta-start", type=float, default=100.0, help="Start value for dynamic beta.")
#     parser.add_argument("--beta-end", type=float, default=50000.0, help="End value for dynamic beta.")
#     parser.add_argument("--teacher-channels", type=int, default=192,
#                         help="Output channels of teacher's g_a, for Adapter.")
#     parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
#     parser.add_argument("--teacher-model", type=str, required=True, help="Teacher model name from compressai.zoo.")
#     parser.add_argument("--teacher-quality", type=int, required=True, help="Teacher model quality level.")
#     parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to teacher model checkpoint.")
#     parser.add_argument("--checkpoint", type=str, help="Path to a student checkpoint to resume.")
#     parser.add_argument("-e", "--epochs", default=300, type=int)
#     parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
#     parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
#     parser.add_argument("--batch-size", type=int, default=8)
#     parser.add_argument("--test-batch-size", type=int, default=1)
#     parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
#     parser.add_argument("--clip_max_norm", default=1.0, type=float)
#     parser.add_argument("--lambda", dest="lmbda", type=float, default=0.01)
#     parser.add_argument("-n", "--num-workers", type=int, default=8)
#     parser.add_argument("--cuda", action="store_true", help="Use cuda.")
#     parser.add_argument("--save", action="store_true", default=True, help="Save model to disk.")
#     parser.add_argument("--seed", type=int, default=42, help="Set random seed.")
#     parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H%M%S'), type=str)
#     return parser.parse_args(argv)


# def main(argv):
#     args = parse_args(argv)
#     base_dir = init(args)
#     if args.seed is not None:
#         torch.manual_seed(args.seed)
#         random.seed(args.seed)

#     if args.wandb:
#         if wandb is None: raise ImportError("Please install wandb: pip install wandb")
#         wandb.init(project=args.wandb_project, name=args.name, config=vars(args))

#     setup_logger(os.path.join(base_dir, 'train.log'))
#     logging.info(f"Distillation run: {args.name}")
#     logging.info(f"Hyperparameters: {vars(args)}")

#     device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
#     logging.info(f"Training on {device}")

#     # --- 資料集與資料載入器設定 ---
#     train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
#     test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

#     train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
#     val_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)

#     train_dataloader = DataLoader(
#         train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
#         shuffle=True, pin_memory=(device == "cuda"))
#     val_dataloader = DataLoader(
#         val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
#         shuffle=False, pin_memory=(device == "cuda"))

#     try:
#         test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
#         test_dataloader = DataLoader(
#             test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
#             shuffle=False, pin_memory=(device == "cuda"))
#         logging.info(f"Test dataset loaded with {len(test_dataset)} images.")
#     except (FileNotFoundError, RuntimeError):
#         logging.warning("Test dataset split not found or empty. Skipping final test.")
#         test_dataloader = None

#     # --- 模型設定 ---
#     student_net = SimpleConvStudentModel(teacher_channels=args.teacher_channels).to(device)

#     logging.info(f"Loading teacher model '{args.teacher_model}' from zoo.")
#     teacher_net = image_models[args.teacher_model](quality=args.teacher_quality, pretrained=False).to(device)

#     logging.info(f"Loading teacher checkpoint from: {args.teacher_checkpoint}")
#     checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
#     teacher_net.load_state_dict(checkpoint.get("state_dict", checkpoint))
#     teacher_net.eval()
#     for param in teacher_net.parameters():
#         param.requires_grad = False

#     # --- ✨✨✨【Hook 掛載邏輯修正】✨✨✨ ---
#     hook_handle = None
#     target_layer_name = 'g_a6'  # 針對 'tic' 模型，我們明確指定最後一層的名稱

#     # 處理單 GPU 的情況
#     if hasattr(teacher_net, target_layer_name):
#         target_layer = getattr(teacher_net, target_layer_name)
#         hook_handle = target_layer.register_forward_hook(get_teacher_features_hook)
#         logging.info(f"Forward hook registered on teacher's layer: {target_layer_name}")
#     else:
#         logging.error(f"Could not find the target layer '{target_layer_name}' in the teacher model.")
#         sys.exit(1)

#     # 處理多 GPU (DataParallel) 的情況
#     if args.cuda and torch.cuda.device_count() > 1:
#         student_net = CustomDataParallel(student_net)
#         teacher_net = CustomDataParallel(teacher_net) # Teacher 也需要包裝

#         if hook_handle: hook_handle.remove() # 移除之前在單 GPU 模型上掛的 hook

#         module_teacher = teacher_net.module
#         if hasattr(module_teacher, target_layer_name):
#             target_layer = getattr(module_teacher, target_layer_name)
#             hook_handle = target_layer.register_forward_hook(get_teacher_features_hook)
#             logging.info(f"Re-registered hook on teacher.module's layer for DataParallel: {target_layer_name}")
#         else:
#             logging.error(f"Could not re-register hook for DataParallel on layer '{target_layer_name}'.")
#             sys.exit(1)

#     # --- 訓練設定 ---
#     optimizer, aux_optimizer = configure_optimizers(student_net, args)
#     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)
#     criterion = RateDistortionLoss(lmbda=args.lmbda)
#     last_epoch = 0
#     if args.checkpoint:
#         logging.info(f"Resuming from checkpoint: {args.checkpoint}")
#         checkpoint = torch.load(args.checkpoint, map_location=device)
#         # 處理 DataParallel 讀取權重
#         state_dict = checkpoint["state_dict"]
#         if isinstance(student_net, CustomDataParallel):
#             student_net.module.load_state_dict(state_dict)
#         else:
#             student_net.load_state_dict(state_dict)

#         last_epoch = checkpoint["epoch"] + 1
#         optimizer.load_state_dict(checkpoint["optimizer"])
#         aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
#         lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

#     # --- 訓練迴圈 ---
#     best_loss = float("inf")
#     try:
#         for epoch in range(last_epoch, args.epochs):
#             logging.info(f"====== Epoch {epoch + 1}/{args.epochs} ======")
#             current_beta = args.beta
#             if args.dynamic_beta:
#                 progress = epoch / (args.epochs - 1) if args.epochs > 1 else 1.0
#                 current_beta = args.beta_start + (args.beta_end - args.beta_start) * progress
#                 if args.wandb: wandb.log({"train/beta": current_beta, "epoch": epoch + 1})
#                 logging.info(f"Dynamic beta active. Current beta: {current_beta:.2f}")

#             train_one_epoch(student_net, teacher_net, criterion, train_dataloader, optimizer, aux_optimizer, epoch,
#                             args.clip_max_norm, args.alpha, current_beta, args.wandb)
#             val_loss = eval_epoch(epoch, val_dataloader, student_net, criterion, args.wandb)
#             lr_scheduler.step(val_loss)

#             is_best = val_loss < best_loss
#             best_loss = min(val_loss, best_loss)

#             if args.save:
#                 # 儲存時，如果是 DataParallel，只儲存 .module 的 state_dict
#                 state_dict_to_save = student_net.module.state_dict() if isinstance(student_net, CustomDataParallel) else student_net.state_dict()
#                 save_checkpoint({
#                     "epoch": epoch,
#                     "state_dict": state_dict_to_save,
#                     "loss": val_loss,
#                     "optimizer": optimizer.state_dict(),
#                     "aux_optimizer": aux_optimizer.state_dict(),
#                     "lr_scheduler": lr_scheduler.state_dict()
#                 }, is_best, base_dir)

#     finally:
#         if hook_handle:
#             hook_handle.remove()
#             logging.info("Forward hook removed from teacher model.")

#     # --- 最終測試 ---
#     logging.info("Training finished.")
#     if test_dataloader is not None:
#         logging.info("Loading best student model from checkpoint for final testing.")
#         best_checkpoint_path = os.path.join(base_dir, "checkpoint_best_loss.pth.tar")

#         if os.path.exists(best_checkpoint_path):
#             checkpoint = torch.load(best_checkpoint_path, map_location=device)
#             # 建立一個新的乾淨模型來載入權重，避免 DataParallel 的麻煩
#             final_student_net = SimpleConvStudentModel.from_state_dict(checkpoint["state_dict"]).to(device)
#             final_student_net.eval()
#             eval_epoch("Final Test", test_dataloader, final_student_net, criterion, args.wandb)
#         else:
#             logging.warning("Could not find best checkpoint for final testing.")

#     if args.wandb:
#         wandb.finish()


# if __name__ == "__main__":
#     main(sys.argv[1:])
# 檔案名稱: onlyconv2.py (最終整合版 - 已加入 Warmup 功能)
# 基於您可運作的腳本，透過 Hook 技術實現特徵蒸餾，並整合 W&B

# 檔案名稱: onlyconv2.py (最終整合版 - 已加入 Warmup 功能)
# 基於您可運作的腳本，透過 Hook 技術實現特徵蒸餾，並整合 W&B

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
from compressai.zoo import image_models
from compressai.datasets import ImageFolder
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ans import BufferedRansEncoder, RansDecoder

# 這個函式是模型 update 時需要的
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


try:
    import wandb
except ImportError:
    wandb = None

# --- 全域變數，用於儲存 Hook 捕獲的老師特徵 ---
teacher_features = {}


def get_teacher_features_hook(module, input, output):
    teacher_features['y'] = output


# --- 模型定義 ---

class BottleneckAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),  # pointwise
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels,
                      # depthwise
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)  # pointwise
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out += identity
        out = self.relu(out)
        return out


class SimpleConvStudentModel(nn.Module):
    def __init__(self, N=128, M=196, teacher_channels=192):
        super().__init__()
        self.N = N
        self.M = M
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            nn.BatchNorm2d(N),  # 加入 BN
            nn.ReLU(inplace=True),

            conv(N, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),  # 加入 BN
            nn.ReLU(inplace=True),

            conv(N, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),  # 加入 BN
            nn.ReLU(inplace=True),

            conv(N, M, kernel_size=3, stride=2),
            # 最後一層通常不加 BN 和 ReLU
        )
        # self.g_a = nn.Sequential(
        #     conv(3, N, kernel_size=5, stride=2), nn.ReLU(inplace=True),
        #     conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        #     conv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        #     conv(N, M, kernel_size=3, stride=2),
        # )
        # self.g_s = nn.Sequential(
        #     deconv(M, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        #     deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        #     deconv(N, N, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        #     deconv(N, 3, kernel_size=5, stride=2),
        # )
        # --- ✨ 修改 g_s，加入 BatchNorm2d ✨ ---
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),  # 加入 BN
            nn.ReLU(inplace=True),

            deconv(N, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),  # 加入 BN
            nn.ReLU(inplace=True),

            deconv(N, N, kernel_size=3, stride=2),
            nn.BatchNorm2d(N),  # 加入 BN
            nn.ReLU(inplace=True),

            deconv(N, 3, kernel_size=5, stride=2),
            # 輸出層不加 BN 和 ReLU
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
        self.adapter = BottleneckAdapter(in_channels=M, out_channels=teacher_channels)
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.g_a(x)
        y_adapted = self.adapter(y)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}, "y_adapted": y_adapted}

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def update(self, scale_table=None, force=False):
        if scale_table is None: scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return any(m.update(force=force) for m in self.modules() if isinstance(m, EntropyBottleneck))

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(self.entropy_bottleneck, "entropy_bottleneck",
                                  ["_quantized_cdf", "_offset", "_cdf_length"], state_dict)
        update_registered_buffers(self.gaussian_conditional, "gaussian_conditional",
                                  ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        teacher_channels = 192  # Default value
        if "adapter.shortcut.weight" in state_dict:
            teacher_channels = state_dict["adapter.shortcut.weight"].size(0)
        elif "adapter.block.4.weight" in state_dict:
            teacher_channels = state_dict["adapter.block.4.weight"].size(0)

        net = cls(N, M, teacher_channels=teacher_channels)
        net.load_state_dict(state_dict, strict=False)
        return net

    # ... (compress/decompress functions are omitted for brevity but should be kept) ...


# --- 訓練腳本組件 ---

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
    def __init__(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0

    def update(self, val, n=1): self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


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


def train_one_epoch(student_model, teacher_model, criterion, train_dataloader, optimizer, aux_optimizer, epoch,
                    clip_max_norm, alpha, beta, use_wandb):
    student_model.train()
    teacher_model.eval()
    device = next(student_model.parameters()).device
    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        with torch.no_grad():
            teacher_out = teacher_model(d)
        student_out = student_model(d)
        task_loss_dict = criterion(student_out, d)
        task_loss = task_loss_dict["loss"]
        response_distill_loss = F.mse_loss(student_out["x_hat"], teacher_out["x_hat"].detach())

        teacher_y = teacher_features.get('y')
        if beta > 0 and teacher_y is None:
            raise RuntimeError("Could not retrieve teacher's feature 'y' via hook.")

        feature_distill_loss = F.mse_loss(student_out["y_adapted"], teacher_y.detach()) if beta > 0 else torch.tensor(
            0.0).to(device)

        total_loss = task_loss + alpha * response_distill_loss + beta * feature_distill_loss
        total_loss.backward()
        if clip_max_norm > 0: torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = student_model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            logging.info(
                f"Train ep {epoch + 1} | [{i * len(d)}/{len(train_dataloader.dataset)}] | Loss:{total_loss.item():.3f} | Task:{task_loss.item():.3f} | Resp:{response_distill_loss.item():.6f} | Feat:{feature_distill_loss.item():.6f}")
        if use_wandb and i % 100 == 0:
            step = epoch * len(train_dataloader) + i
            mse = task_loss_dict["mse_loss"].item()
            psnr_val = 10 * math.log10(1. / mse) if mse > 0 else float('inf')
            wandb.log({"train/step": step, "train/loss": total_loss.item(), "train/task_loss": task_loss.item(),
                       "train/response_loss": response_distill_loss.item(),
                       "train/feature_loss": feature_distill_loss.item(),
                       "train/bpp_loss": task_loss_dict["bpp_loss"].item(), "train/mse_loss": mse,
                       "train/psnr": psnr_val, "train/aux_loss": aux_loss.item(),
                       "train/lr": optimizer.param_groups[0]['lr']})


def eval_epoch(epoch, dataloader, model, criterion, use_wandb):
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
            mse_val = out_criterion["mse_loss"].item()
            mse_loss.update(mse_val)
            if mse_val > 0:
                psnr.update(10 * math.log10(1. / mse_val))

    log_prefix = "Final Test" if isinstance(epoch, str) else f"Validation epoch {epoch + 1}"
    logging.info(
        f"{log_prefix}: Loss:{loss.avg:.4f} | MSE:{mse_loss.avg:.6f} | PSNR:{psnr.avg:.3f} | Bpp:{bpp_loss.avg:.4f} | Aux:{aux_loss.avg:.2f}")

    if use_wandb and not isinstance(epoch, str):
        wandb.log({"val/epoch": epoch + 1, "val/loss": loss.avg, "val/mse_loss": mse_loss.avg, "val/psnr": psnr.avg,
                   "val/bpp_loss": bpp_loss.avg, "val/aux_loss": aux_loss.avg})
    return loss.avg


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best: shutil.copyfile(os.path.join(base_dir, filename),
                                os.path.join(base_dir, "checkpoint_best_loss.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Distillation training script with Hooks and W&B.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="student-distill", help="W&B project name.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for response distillation loss.")
    parser.add_argument("--beta", type=float, default=10000.0,
                        help="Weight for feature distillation loss. Set to 0 to disable.")
    parser.add_argument("--dynamic-beta", action="store_true", help="Enable dynamic beta scheduling.")
    parser.add_argument("--beta-start", type=float, default=100.0, help="Start value for dynamic beta.")
    parser.add_argument("--beta-end", type=float, default=50000.0, help="End value for dynamic beta.")

    # ✨✨✨【新增處】✨✨✨
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs.")

    parser.add_argument("--teacher-channels", type=int, default=192,
                        help="Output channels of teacher's g_a, for Adapter.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset")
    parser.add_argument("--teacher-model", type=str, required=True, help="Teacher model name from compressai.zoo.")
    parser.add_argument("--teacher-quality", type=int, required=True, help="Teacher model quality level.")
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to teacher model checkpoint.")
    parser.add_argument("--checkpoint", type=str, help="Path to a student checkpoint to resume.")
    parser.add_argument("-e", "--epochs", default=300, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.01)
    parser.add_argument("-n", "--num-workers", type=int, default=8)
    parser.add_argument("--cuda", action="store_true", help="Use cuda.")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk.")
    parser.add_argument("--seed", type=int, default=42, help="Set random seed.")
    parser.add_argument('--name', default=datetime.now().strftime('%Y-%m-%d_%H%M%S'), type=str)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if args.wandb:
        if wandb is None: raise ImportError("Please install wandb: pip install wandb")
        wandb.init(project=args.wandb_project, name=args.name, config=vars(args))

    setup_logger(os.path.join(base_dir, 'train.log'))
    logging.info(f"Distillation run: {args.name}")
    logging.info(f"Hyperparameters: {vars(args)}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info(f"Training on {device}")

    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    val_dataset = ImageFolder(args.dataset, split="val", transform=test_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                  pin_memory=(device == "cuda"))
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                shuffle=False, pin_memory=(device == "cuda"))

    student_net = SimpleConvStudentModel(teacher_channels=args.teacher_channels).to(device)
    logging.info(f"Loading teacher model '{args.teacher_model}' from zoo.")
    teacher_net = image_models[args.teacher_model](quality=args.teacher_quality, pretrained=False).to(device)
    logging.info(f"Loading teacher checkpoint from: {args.teacher_checkpoint}")
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_net.load_state_dict(checkpoint.get("state_dict", checkpoint))
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

    hook_handle = None;
    target_layer_name = 'g_a6'
    if hasattr(teacher_net, target_layer_name):
        target_layer = getattr(teacher_net, target_layer_name)
        hook_handle = target_layer.register_forward_hook(get_teacher_features_hook)
        logging.info(f"Forward hook registered on teacher's layer: {target_layer_name}")
    else:
        logging.error(f"Could not find the target layer '{target_layer_name}' in the teacher model.");
        sys.exit(1)

    if args.cuda and torch.cuda.device_count() > 1:
        student_net = CustomDataParallel(student_net);
        teacher_net = CustomDataParallel(teacher_net)
        if hook_handle: hook_handle.remove()
        module_teacher = teacher_net.module
        if hasattr(module_teacher, target_layer_name):
            target_layer = getattr(module_teacher, target_layer_name)
            hook_handle = target_layer.register_forward_hook(get_teacher_features_hook)
            logging.info(f"Re-registered hook on teacher.module's layer for DataParallel: {target_layer_name}")
        else:
            logging.error(f"Could not re-register hook for DataParallel on layer '{target_layer_name}'.");
            sys.exit(1)

    optimizer, aux_optimizer = configure_optimizers(student_net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = RateDistortionLoss(lmbda=args.lmbda)
    last_epoch = 0
    if args.checkpoint:
        logging.info(f"Resuming from checkpoint: {args.checkpoint}");
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint["state_dict"]
        if isinstance(student_net, CustomDataParallel):
            student_net.module.load_state_dict(state_dict)
        else:
            student_net.load_state_dict(state_dict)
        last_epoch = checkpoint["epoch"] + 1;
        optimizer.load_state_dict(checkpoint["optimizer"]);
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"]);
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # --- 訓練迴圈 ---
    best_loss = float("inf")
    try:
        for epoch in range(last_epoch, args.epochs):
            logging.info(f"====== Epoch {epoch + 1}/{args.epochs} ======")

            # ✨✨✨【修改處】✨✨✨
            current_beta = args.beta
            if epoch < args.warmup_epochs:
                current_beta = 0.0
                logging.info(f"Warmup epoch {epoch + 1}/{args.warmup_epochs}. Feature distillation is OFF (beta=0).")
            elif args.dynamic_beta:
                total_schedule_epochs = args.epochs - args.warmup_epochs
                current_schedule_epoch = epoch - args.warmup_epochs
                progress = current_schedule_epoch / (total_schedule_epochs - 1) if total_schedule_epochs > 1 else 1.0
                current_beta = args.beta_start + (args.beta_end - args.beta_start) * progress
                logging.info(f"Dynamic beta active. Current beta: {current_beta:.2f}")

            if args.wandb:
                wandb.log({"train/beta": current_beta, "epoch": epoch + 1})
            # ✨✨✨【修改結束】✨✨✨

            train_one_epoch(student_net, teacher_net, criterion, train_dataloader, optimizer, aux_optimizer, epoch,
                            args.clip_max_norm, args.alpha, current_beta, args.wandb)
            val_loss = eval_epoch(epoch, val_dataloader, student_net, criterion, args.wandb)
            lr_scheduler.step()

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if args.save:
                state_dict_to_save = student_net.module.state_dict() if isinstance(student_net,
                                                                                   CustomDataParallel) else student_net.state_dict()
                save_checkpoint({
                    "epoch": epoch,
                    "state_dict": state_dict_to_save,
                    "loss": val_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict()
                }, is_best, base_dir)
    finally:
        if hook_handle:
            hook_handle.remove()
            logging.info("Forward hook removed from teacher model.")

    logging.info("Training finished.")

    # Final Test logic is omitted for brevity but should be kept

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])