"""
ONNX-based Satellite Image Decompression Tool

This script uses ONNX Runtime for faster inference on ARM/x86 CPUs.
Requires:
- ONNX model files: tic_decoder.onnx, tic_hyper_decoder.onnx
- fixed_cdfs.py: Pre-computed CDF tables for entropy coding (must match compression)
"""
import argparse
import os
import sys
import glob
import time
import struct
import zlib
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance

# Try to import MS-SSIM
try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False

# Import CompressAI entropy modules
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

# ==============================================================================
# 0. Load Fixed CDFs (Must match compression side)
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXAMPLES_DIR)

try:
    from fixed_cdfs import (
        FIXED_EB_CDF, FIXED_EB_OFFSET, FIXED_EB_LENGTH, FIXED_EB_MEDIANS,
        FIXED_GC_CDF, FIXED_GC_OFFSET, FIXED_GC_LENGTH, FIXED_GC_SCALE_TABLE
    )
    HAS_FIXED_CDFS = True
    print("✅ Loaded fixed_cdfs.py")
except ImportError:
    HAS_FIXED_CDFS = False
    print("⚠️ [WARNING] Cannot find fixed_cdfs.py, decoding may fail!")

# ==============================================================================
# 1. Packet Parsing
# ==============================================================================
def load_satellite_packet(bin_path):
    """
    Parse binary packet format: [Magic:3][ID:1][Row:1][Col:1][H:2][W:2][LenY:4][LenZ:4]...[CRC:4]
    """
    try:
        with open(bin_path, "rb") as f:
            data = f.read()

        if len(data) < 22:
            return None

        # CRC Check
        received_crc = struct.unpack('<I', data[-4:])[0]
        content = data[:-4]
        calc_crc = zlib.crc32(content) & 0xffffffff
        if received_crc != calc_crc:
            print(f"[Corrupt] CRC Fail: {os.path.basename(bin_path)}")
            return None

        # Parse Header
        header_size = 18
        magic, img_id, row, col, h, w, len_y, len_z = struct.unpack('<3sBBBHHII', data[:header_size])
        
        if magic != b'TIC':
            return None
        
        cursor = header_size
        y_str = data[cursor : cursor + len_y]
        cursor += len_y
        z_str = data[cursor : cursor + len_z]
        
        return {
            "row": row, "col": col, "img_id": img_id,
            "strings": [[y_str], [z_str]], 
            "shape": (h, w)
        }
    except Exception as e:
        print(f"[Read Error] {bin_path}: {e}")
        return None

# ==============================================================================
# 2. Image Metrics and Loading
# ==============================================================================
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * np.log10(mse) if mse > 0 else float('inf')

def read_original_image(filepath: str) -> torch.Tensor:
    """Read original image for PSNR calculation (handles TIFF and Standard formats)"""
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.tif', '.tiff']:
        # Try to use tifffile if available, else PIL
        try:
            import tifffile
            raw_data = tifffile.imread(filepath)
            original_dtype = raw_data.dtype
            
            # Handle dim order (H,W,C) -> (C,H,W)
            if raw_data.ndim == 3:
                if raw_data.shape[2] <= 4: # Channel last
                    raw_data = np.transpose(raw_data, (2, 0, 1))
            elif raw_data.ndim == 2:
                raw_data = np.expand_dims(raw_data, axis=0)
            
            # RGB only
            rgb_data = raw_data[:3, :, :].astype(np.float32)
            
            if original_dtype == np.uint8:
                return torch.from_numpy(np.clip(rgb_data, 0, 255) / 255.0)
            else:
                return torch.from_numpy(np.clip(rgb_data, 0, 10000) / 10000.0)
        except ImportError:
            # Fallback to PIL
            img = Image.open(filepath).convert("RGB")
            return transforms.ToTensor()(img)
    else:
        img = Image.open(filepath).convert("RGB")
        return transforms.ToTensor()(img)

# ==============================================================================
# 3. Initialize ONNX and Entropy Models
# ==============================================================================
def init_onnx_decoder(decoder_path, hyper_path, use_cuda=True, num_threads=4):
    """Initialize ONNX sessions and entropy models with fixed CDFs."""
    
    # ========== ONNX Graph Optimization (Operator Fusion) ==========
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    print(f"⚡ ONNX Optimization: ALL (Threads: {num_threads})")
    # ================================================================
    
    if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"🚀 Device: NVIDIA GPU (CUDA)")
    else:
        providers = ['CPUExecutionProvider']
        print(f"⚠️ Device: CPU")

    print(f"Loading Decoder: {decoder_path}")
    dec_sess = ort.InferenceSession(decoder_path, sess_options=sess_options, providers=providers)
    
    print(f"Loading HyperDecoder: {hyper_path}")
    hyper_sess = ort.InferenceSession(hyper_path, sess_options=sess_options, providers=providers)

    entropy_bottleneck = EntropyBottleneck(128) 
    gaussian_conditional = GaussianConditional(None)

    # Apply fixed CDF tables
    if HAS_FIXED_CDFS:
        print("✅ Loading fixed CDF tables...")
        device = torch.device("cpu")
        entropy_bottleneck._quantized_cdf.resize_(torch.tensor(FIXED_EB_CDF).shape).copy_(
            torch.tensor(FIXED_EB_CDF, device=device, dtype=torch.int32))
        entropy_bottleneck._offset.resize_(torch.tensor(FIXED_EB_OFFSET).shape).copy_(
            torch.tensor(FIXED_EB_OFFSET, device=device, dtype=torch.int32))
        entropy_bottleneck._cdf_length.resize_(torch.tensor(FIXED_EB_LENGTH).shape).copy_(
            torch.tensor(FIXED_EB_LENGTH, device=device, dtype=torch.int32))
        entropy_bottleneck.quantiles.data[:, 0, 1] = torch.tensor(FIXED_EB_MEDIANS, device=device).squeeze()

        gaussian_conditional._quantized_cdf.resize_(torch.tensor(FIXED_GC_CDF).shape).copy_(
            torch.tensor(FIXED_GC_CDF, device=device, dtype=torch.int32))
        gaussian_conditional._offset.resize_(torch.tensor(FIXED_GC_OFFSET).shape).copy_(
            torch.tensor(FIXED_GC_OFFSET, device=device, dtype=torch.int32))
        gaussian_conditional._cdf_length.resize_(torch.tensor(FIXED_GC_LENGTH).shape).copy_(
            torch.tensor(FIXED_GC_LENGTH, device=device, dtype=torch.int32))
        gaussian_conditional.scale_table = torch.tensor(FIXED_GC_SCALE_TABLE, device=device)
    else:
        print("⚠️ [WARNING] Using default CDF, decoding will likely fail!")
        entropy_bottleneck.update(force=True)
        import math
        scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))
        gaussian_conditional.update_scale_table(scale_table, force=True)

    return (dec_sess, hyper_sess), (entropy_bottleneck, gaussian_conditional)

# ==============================================================================
# 3. Single Packet Decompression
# ==============================================================================
def process_decompress_packet(sessions, entropy_models, packet_data):
    """Decompress a single packet using ONNX models and entropy decoding."""
    dec_sess, hyper_sess = sessions
    entropy_bottleneck, gaussian_conditional = entropy_models
    
    strings = packet_data["strings"]
    shape = packet_data["shape"]  # (h, w) of z

    # 1. Entropy decode Z
    z_hat = entropy_bottleneck.decompress(strings[1], shape)
    
    # 2. Hyper decoder (ONNX)
    z_hat_np = z_hat.numpy()
    hyper_out = hyper_sess.run(None, {"z_hat": z_hat_np})
    scales_np, means_np = hyper_out[0], hyper_out[1]
    
    # 3. Entropy decode Y
    scales = torch.from_numpy(scales_np)
    means = torch.from_numpy(means_np)
    indexes = gaussian_conditional.build_indexes(scales)
    y_hat = gaussian_conditional.decompress(strings[0], indexes, means=means)
    
    # 4. Main decoder (ONNX)
    y_hat_np = y_hat.numpy()
    dec_out = dec_sess.run(None, {"y_hat": y_hat_np})
    x_hat_np = dec_out[0]

    # Post-processing
    x_hat = torch.from_numpy(x_hat_np).clamp(0, 1)

    # Crop to 256x256 (remove padding)
    target_h, target_w = 256, 256
    curr_h, curr_w = x_hat.size(2), x_hat.size(3)
    if curr_h != target_h or curr_w != target_w:
        padding_left = (curr_w - target_w) // 2
        padding_top = (curr_h - target_h) // 2
        x_hat = x_hat[:, :, padding_top:padding_top + target_h, padding_left:padding_left + target_w]

    return x_hat

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ONNX Satellite Image Decompression")
    parser.add_argument("bin_dir", type=str, help="Directory containing .bin files")
    parser.add_argument("--dec", type=str, default=None, help="Decoder ONNX (Default: checks static_int8)")
    parser.add_argument("--hyper", type=str, default=None, help="HyperDecoder ONNX (Default: checks static_int8)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--original", type=str, default=None, help="Path to original image for PSNR calculation")
    parser.add_argument("--target_id", type=int, default=None, help="Filter by Image ID")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness adjustment factor (default: 1.0)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output filename (default: RECONSTRUCTED_ONNX.png)")
    args = parser.parse_args()

    # Smart Defaults for models (FP32 for best quality, no grid lines)
    dec_path = args.dec
    if dec_path is None:
        # Use FP32 Decoder by default to eliminate grid artifacts
        dec_path = "tic_decoder.onnx"
    
    hyper_path = args.hyper
    if hyper_path is None:
        # Use FP32 HyperDecoder by default (must match compression)
        hyper_path = "tic_hyper_decoder.onnx"

    # 1. Initialize System
    sessions, entropy_models = init_onnx_decoder(dec_path, hyper_path, use_cuda=not args.cpu)
    PATCH_SIZE = 256

    # Find .bin files
    bin_files = glob.glob(os.path.join(args.bin_dir, "*.bin"))
    if not bin_files:
        print(f"No .bin files found in: {args.bin_dir}")
        return

    # Pre-scan packets
    max_row, max_col = 0, 0
    valid_packets = []
    
    print("Scanning packets...")
    for f in bin_files:
        packet = load_satellite_packet(f)
        if packet is None:
            continue
        if args.target_id is not None and packet['img_id'] != args.target_id:
            continue

        max_row = max(max_row, packet['row'])
        max_col = max(max_col, packet['col'])
        valid_packets.append(packet)

    if not valid_packets:
        print("No valid packets found.")
        return

    canvas_w = (max_col + 1) * PATCH_SIZE
    canvas_h = (max_row + 1) * PATCH_SIZE
    print(f"Canvas size: {canvas_w}x{canvas_h}")
    full_recon_img = Image.new('RGB', (canvas_w, canvas_h), (0, 0, 0))

    # Decompress
    print(f"Decompressing {len(valid_packets)} patches...")
    start_time = time.time()
    
    for i, packet in enumerate(valid_packets):
        r, c = packet['row'], packet['col']
        print(f"Progress: {i+1}/{len(valid_packets)}", end='\r')
        
        try:
            x_hat = process_decompress_packet(sessions, entropy_models, packet)
            
            # Tensor to PIL
            rec_tensor = x_hat.squeeze().cpu()
            rec_patch_pil = transforms.ToPILImage()(rec_tensor)
            
            # Paste
            left = c * PATCH_SIZE
            upper = r * PATCH_SIZE
            full_recon_img.paste(rec_patch_pil, (left, upper))
            
        except Exception as e:
            print(f"\n[ERROR] Patch ({r},{c}) failed: {e}")

    total_time = time.time() - start_time
    print(f"\nDecompression complete! Time: {total_time:.2f} sec")

    # ==========================================================================
    # 5. Calculate Metrics (Before brightness adjustment)
    # ==========================================================================
    if args.original:
        print("-" * 40)
        if not os.path.exists(args.original):
            print(f"⚠️ Warning: Original image not found: {args.original}")
        else:
            try:
                gt_tensor = read_original_image(args.original)
                rec_tensor = transforms.ToTensor()(full_recon_img)

                # Align dimensions
                h_gt, w_gt = gt_tensor.shape[1], gt_tensor.shape[2]
                h_rec, w_rec = rec_tensor.shape[1], rec_tensor.shape[2]
                min_h, min_w = min(h_gt, h_rec), min(w_gt, w_rec)

                gt_tensor = gt_tensor[:, :min_h, :min_w]
                rec_tensor = rec_tensor[:, :min_h, :min_w]

                val_psnr = psnr(gt_tensor, rec_tensor)
                print(f"✅ PSNR:    {val_psnr:.4f} dB")
                
                if HAS_MSSSIM:
                    val_msssim = ms_ssim(gt_tensor.unsqueeze(0), rec_tensor.unsqueeze(0), data_range=1.0).item()
                    print(f"✅ MS-SSIM: {val_msssim:.4f}")
                
                print("-" * 40)
            except Exception as e:
                print(f"❌ Error calculating metrics: {e}")

    # Post-processing and save
    if args.brightness != 1.0:
        enhancer = ImageEnhance.Brightness(full_recon_img)
        full_recon_img = enhancer.enhance(args.brightness)
    
    output_filename = args.output if args.output else "RECONSTRUCTED_ONNX.png"
    output_path = os.path.join(args.bin_dir, output_filename)
    full_recon_img.save(output_path)
    print(f"Image saved: {output_path}")

if __name__ == "__main__":
    main()