"""
ONNX-based Satellite Image Compression Tool (Optimized with Batching)

Optimizations:
1. NumPy vectorized slicing - no Python loops for patch extraction
2. Batch processing - reduces ONNX launch overhead on ARM
3. Pure NumPy padding - avoids PyTorch tensor conversion overhead

Usage:
    python nxinfcom_batch.py image.tif --enc encoder.onnx --hyper hyper.onnx --batch 8
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
from PIL import Image

# Import CompressAI entropy modules
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

# Try to import tifffile for satellite imagery
try:
    import tifffile
except ImportError:
    tifffile = None

# ==============================================================================
# 0. Load Fixed CDFs
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EXAMPLES_DIR)

try:
    import fixed_cdfs
    print("✅ Loaded fixed_cdfs.py")
except ImportError:
    print("\n[CRITICAL ERROR] Cannot find fixed_cdfs.py!")
    sys.exit(1)

# ==============================================================================
# 1. Satellite Communication Packet Format
# ==============================================================================
def save_satellite_packet(out_enc, output_path, img_id, row, col):
    """Binary packet format: [Magic:3][ID:1][Row:1][Col:1][H:2][W:2][LenY:4][LenZ:4]...[CRC:4]"""
    y_strings = out_enc["strings"][0]
    z_strings = out_enc["strings"][1]
    shape = out_enc["shape"]

    if isinstance(y_strings[0], (bytes, bytearray)):
        y_str_payload = b''.join(y_strings)
    else:
        y_str_payload = b''.join(y_strings[0])
         
    if isinstance(z_strings[0], (bytes, bytearray)):
        z_str_payload = b''.join(z_strings)
    else:
        z_str_payload = b''.join(z_strings[0])

    magic = b'TIC'
    h, w = shape
    len_y = len(y_str_payload)
    len_z = len(z_str_payload)

    header = struct.pack('<3sBBBHHII', magic, img_id, row, col, h, w, len_y, len_z)
    packet_content = header + y_str_payload + z_str_payload
    crc = zlib.crc32(packet_content) & 0xffffffff
    footer = struct.pack('<I', crc)

    with open(output_path, "wb") as f:
        f.write(packet_content + footer)

# ==============================================================================
# 2. NumPy Vectorized Patch Extraction (Zero-copy)
# ==============================================================================
def extract_patches_vectorized(full_image, patch_size=256):
    """
    Extract all patches using NumPy reshape/transpose (no Python loops).
    
    Input: full_image (C, H, W) - NumPy array
    Output: patches (N, C, patch_size, patch_size), meta_data [(row, col), ...]
    """
    c, h, w = full_image.shape
    
    # Pad image to be divisible by patch_size
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(full_image, 
                            ((0, 0), (0, pad_h), (0, pad_w)), 
                            mode='constant', constant_values=0)
    else:
        img_padded = full_image
    
    _, h_pad, w_pad = img_padded.shape
    n_rows = h_pad // patch_size
    n_cols = w_pad // patch_size
    
    # [MAGIC] Reshape + Transpose for instant slicing
    # (C, H, W) -> (C, Rows, PatchH, Cols, PatchW)
    patches_grid = img_padded.reshape(c, n_rows, patch_size, n_cols, patch_size)
    
    # Transpose: (Rows, Cols, C, PatchH, PatchW)
    patches_grid = patches_grid.transpose(1, 3, 0, 2, 4)
    
    # Flatten: (N, C, PatchH, PatchW) where N = Rows * Cols
    patches = patches_grid.reshape(-1, c, patch_size, patch_size)
    
    # Generate metadata (this small loop is negligible)
    meta_data = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    
    return patches.astype(np.float32), meta_data, n_rows, n_cols

# ==============================================================================
# 3. NumPy Padding for ONNX (batch-aware)
# ==============================================================================
def pad_batch_numpy(batch, target_multiple=64):
    """
    Pad a batch of images to be divisible by target_multiple using pure NumPy.
    Input: batch (N, C, H, W)
    Output: padded_batch (N, C, new_H, new_W)
    """
    n, c, h, w = batch.shape
    new_h = ((h + target_multiple - 1) // target_multiple) * target_multiple
    new_w = ((w + target_multiple - 1) // target_multiple) * target_multiple
    
    if new_h == h and new_w == w:
        return batch
    
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    
    # np.pad format: ((batch,), (channel,), (top, bottom), (left, right))
    padded = np.pad(batch, 
                    ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0)
    return padded

# ==============================================================================
# 4. Batch Processing Core
# ==============================================================================
def process_batch_onnx(sessions, entropy_models, batch_x, batch_meta, 
                       save_dir, base_name, img_id, stats):
    """
    Process a batch of patches through ONNX and entropy coding.
    Note: Entropy coding still processes one patch at a time (CompressAI limitation).
    """
    enc_sess, hyper_sess = sessions
    entropy_bottleneck, gaussian_conditional = entropy_models
    
    batch_size = batch_x.shape[0]
    total_bpp = 0.0
    
    # 1. Padding (NumPy, batch-aware)
    t0 = time.time()
    batch_padded = pad_batch_numpy(batch_x, target_multiple=64)
    stats['prep'] += (time.time() - t0)
    
    # 2. Encoder inference (single batch call)
    t0 = time.time()
    enc_out = enc_sess.run(None, {"input_image": batch_padded})
    y_batch, z_batch = enc_out[0], enc_out[1]  # (N, M, H/16, W/16), (N, N, H/64, W/64)
    stats['encoder'] += (time.time() - t0)
    
    # 3. Process each patch's entropy coding (cannot batch - CompressAI limitation)
    for i in range(batch_size):
        row, col = batch_meta[i]
        
        # Extract single patch latents
        y = torch.from_numpy(y_batch[i:i+1])
        z = torch.from_numpy(z_batch[i:i+1])
        
        # Entropy coding for Z
        t0 = time.time()
        z_strings = entropy_bottleneck.compress(z)
        medians = entropy_bottleneck._get_medians().detach()
        spatial_dims = len(z.size()) - 2
        medians = entropy_bottleneck._extend_ndims(medians, spatial_dims)
        medians = medians.expand(z.size(0), *([-1] * (spatial_dims + 1)))
        z_hat = entropy_bottleneck.quantize(z, "dequantize", medians)
        stats['entropy_z'] += (time.time() - t0)
        
        # Hyper decoder
        t0 = time.time()
        hyper_out = hyper_sess.run(None, {"z_hat": z_hat.numpy()})
        scales = torch.from_numpy(hyper_out[0])
        means = torch.from_numpy(hyper_out[1])
        stats['hyper'] += (time.time() - t0)
        
        # Entropy coding for Y
        t0 = time.time()
        indexes = gaussian_conditional.build_indexes(scales)
        y_strings = gaussian_conditional.compress(y, indexes, means=means)
        stats['entropy_y'] += (time.time() - t0)
        
        # Save packet
        t0 = time.time()
        bin_path = os.path.join(save_dir, f"{base_name}_row{row}_col{col}.bin")
        out_enc = {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
        save_satellite_packet(out_enc, bin_path, img_id, row, col)
        stats['save'] += (time.time() - t0)
        
        # Calculate BPP
        patch_h, patch_w = batch_x.shape[2], batch_x.shape[3]
        num_pixels = patch_h * patch_w
        total_bits = os.path.getsize(bin_path) * 8.0
        total_bpp += total_bits / num_pixels
    
    return total_bpp

# ==============================================================================
# 5. Initialize ONNX Environment
# ==============================================================================
def init_onnx_environment(encoder_path, hyper_path, use_cuda=True):
    """Initialize ONNX sessions and entropy models with fixed CDFs."""
    if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"🚀 Device: NVIDIA GPU (CUDA)")
    else:
        providers = ['CPUExecutionProvider']
        print(f"⚠️ Device: CPU (ARM/x86)")

    print(f"Loading Encoder: {encoder_path}")
    enc_sess = ort.InferenceSession(encoder_path, providers=providers)
    
    print(f"Loading HyperDecoder: {hyper_path}")
    hyper_sess = ort.InferenceSession(hyper_path, providers=providers)

    entropy_bottleneck = EntropyBottleneck(128) 
    gaussian_conditional = GaussianConditional(None)

    # Apply fixed CDF tables
    device = torch.device("cpu")
    
    entropy_bottleneck._quantized_cdf.resize_(torch.tensor(fixed_cdfs.FIXED_EB_CDF).shape).copy_(
        torch.tensor(fixed_cdfs.FIXED_EB_CDF, device=device, dtype=torch.int32))
    entropy_bottleneck._offset.resize_(torch.tensor(fixed_cdfs.FIXED_EB_OFFSET).shape).copy_(
        torch.tensor(fixed_cdfs.FIXED_EB_OFFSET, device=device, dtype=torch.int32))
    entropy_bottleneck._cdf_length.resize_(torch.tensor(fixed_cdfs.FIXED_EB_LENGTH).shape).copy_(
        torch.tensor(fixed_cdfs.FIXED_EB_LENGTH, device=device, dtype=torch.int32))
    entropy_bottleneck.quantiles.data[:, 0, 1] = torch.tensor(fixed_cdfs.FIXED_EB_MEDIANS, device=device).squeeze()

    gaussian_conditional._quantized_cdf.resize_(torch.tensor(fixed_cdfs.FIXED_GC_CDF).shape).copy_(
        torch.tensor(fixed_cdfs.FIXED_GC_CDF, device=device, dtype=torch.int32))
    gaussian_conditional._offset.resize_(torch.tensor(fixed_cdfs.FIXED_GC_OFFSET).shape).copy_(
        torch.tensor(fixed_cdfs.FIXED_GC_OFFSET, device=device, dtype=torch.int32))
    gaussian_conditional._cdf_length.resize_(torch.tensor(fixed_cdfs.FIXED_GC_LENGTH).shape).copy_(
        torch.tensor(fixed_cdfs.FIXED_GC_LENGTH, device=device, dtype=torch.int32))
    gaussian_conditional.scale_table = torch.tensor(fixed_cdfs.FIXED_GC_SCALE_TABLE, device=device)
    
    return (enc_sess, hyper_sess), (entropy_bottleneck, gaussian_conditional)

# ==============================================================================
# 6. Main Compression Loop (Batched)
# ==============================================================================
def compress_single_image(sessions, entropy_models, input_path, output_dir, img_id, 
                          batch_size=8, patch_size=256):
    """Process a single image with batched ONNX inference."""
    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]
    save_dir = os.path.join(output_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nOutput folder: {save_dir}")

    stats = {
        'load_io': 0.0, 'extract': 0.0, 'prep': 0.0, 
        'encoder': 0.0, 'entropy_z': 0.0, 'hyper': 0.0, 
        'entropy_y': 0.0, 'save': 0.0
    }
    
    # 1. Load image
    t0 = time.time()
    if tifffile and os.path.splitext(input_path)[-1].lower() in ['.tif', '.tiff']:
        full_image = tifffile.imread(input_path)
        original_dtype = full_image.dtype
        if full_image.ndim == 3 and full_image.shape[2] <= 4:
            full_image = np.transpose(full_image, (2, 0, 1))
        elif full_image.ndim == 2:
            full_image = np.expand_dims(full_image, axis=0)
    else:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            full_image = np.transpose(np.array(img), (2, 0, 1))
            original_dtype = np.uint8
    
    # Normalize
    full_image = full_image.astype(np.float32)
    if np.isnan(full_image).any():
        full_image = np.nan_to_num(full_image)
    full_image = full_image[:3]  # RGB only
    
    if original_dtype == np.uint8:
        full_image = np.clip(full_image, 0, 255) / 255.0
    else:
        full_image = np.clip(full_image, 0, 10000) / 10000.0
    
    stats['load_io'] = time.time() - t0
    
    # 2. Extract all patches (NumPy vectorized - no loops!)
    t0 = time.time()
    patches, meta_data, n_rows, n_cols = extract_patches_vectorized(full_image, patch_size)
    total_patches = len(patches)
    stats['extract'] = time.time() - t0
    
    img_h, img_w = full_image.shape[1], full_image.shape[2]
    print(f"Image: {img_w}x{img_h} -> Grid: {n_rows}x{n_cols} (Batch size: {batch_size})")
    
    # 3. Process in batches
    total_bpp = 0.0
    processed = 0
    
    for i in range(0, total_patches, batch_size):
        batch_end = min(i + batch_size, total_patches)
        batch_x = patches[i:batch_end]
        batch_meta = meta_data[i:batch_end]
        
        bpp = process_batch_onnx(sessions, entropy_models, batch_x, batch_meta,
                                 save_dir, base_name, img_id, stats)
        total_bpp += bpp
        processed += len(batch_x)
        
        print(f"Progress: {processed}/{total_patches}", end='\r')
    
    avg_bpp = total_bpp / total_patches if total_patches > 0 else 0
    
    # Performance report
    print("\n" + "=" * 55)
    print(f"📊 Performance Report ({filename}) - Batch Size: {batch_size}")
    print("-" * 55)
    sum_tracked = sum(stats.values())
    for key, val in stats.items():
        percentage = (val / sum_tracked * 100) if sum_tracked > 0 else 0
        print(f"{key:<15} | {val:.4f}s    | {percentage:.1f}%")
    print("-" * 55)
    print(f"Total time: {sum_tracked:.4f} sec")
    print(f"Average BPP: {avg_bpp:.4f}")
    print("=" * 55 + "\n")
    return avg_bpp, sum_tracked

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ONNX Satellite Compression (Batched)")
    parser.add_argument("input_path", type=str, nargs='+', help="Input image(s)")
    parser.add_argument("-o", "--output_dir", type=str, default="output_onnx", help="Output directory")
    parser.add_argument("--enc", type=str, default="tic_encoder.onnx", help="Encoder ONNX")
    parser.add_argument("--hyper", type=str, default="tic_hyper_decoder.onnx", help="HyperDecoder ONNX")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--id", type=int, default=1, help="Image ID (0-255)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    sessions, entropy_models = init_onnx_environment(args.enc, args.hyper, use_cuda=not args.cpu)

    image_files = []
    for path in args.input_path:
        if os.path.isfile(path):
            image_files.append(path)
        elif os.path.isdir(path):
            for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
                image_files.extend(glob.glob(os.path.join(path, ext)))
    
    if not image_files:
        print("No image files found.")
        return

    print(f"=== Processing {len(image_files)} image(s) with batch size {args.batch} ===")
    for img_path in image_files:
        try:
            compress_single_image(sessions, entropy_models, img_path, 
                                  args.output_dir, args.id, batch_size=args.batch)
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
