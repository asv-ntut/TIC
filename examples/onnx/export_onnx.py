"""
ONNX Export Tool for TIC Model

This script exports the TIC model to three ONNX files:
1. tic_encoder.onnx      - Image encoder (g_a + h_a)
2. tic_hyper_decoder.onnx - Hyperprior decoder (h_s)
3. tic_decoder.onnx      - Image decoder (g_s)

Usage:
    python export_onnx.py -p /path/to/checkpoint.pth.tar [-o output_dir]
"""
import argparse
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

# Add parent directories to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.insert(0, ROOT_DIR)

from compressai.models.tic import TIC, TIC_Student


# ==============================================================================
# ONNX Wrapper Modules
# ==============================================================================

class NetEncoder(nn.Module):
    """Part 1: Analysis transform (Image -> y, z)"""
    def __init__(self, tic_model):
        super().__init__()
        self.g_a = tic_model.g_a
        self.h_a = tic_model.h_a

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        return y, z


class NetHyperDecoder(nn.Module):
    """Part 2: Hyperprior decoder (z_hat -> scales, means)"""
    def __init__(self, tic_model):
        super().__init__()
        self.h_s = tic_model.h_s

    def forward(self, z_hat):
        gaussian_params = self.h_s(z_hat)
        # Use slice instead of chunk/split for better ONNX compatibility
        half_channel = gaussian_params.size(1) // 2
        scales_hat = gaussian_params[:, :half_channel, :, :]
        means_hat = gaussian_params[:, half_channel:, :, :]
        return scales_hat, means_hat


class NetDecoder(nn.Module):
    """Part 3: Synthesis transform (y_hat -> Reconstructed Image)"""
    def __init__(self, tic_model):
        super().__init__()
        self.g_s = tic_model.g_s

    def forward(self, y_hat):
        x_hat = self.g_s(y_hat)
        return x_hat


# ==============================================================================
# Main Export Logic
# ==============================================================================

def load_checkpoint(checkpoint_path, device='cpu'):
    """Load TIC or TIC_Student model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle CompressAI's "state_dict" wrapper
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove "module." prefix if present (from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Infer N and M from weights
    N = new_state_dict.get('g_a.0.weight', torch.zeros(128, 3, 5, 5)).size(0)
    M = 192  # Default
    try:
        keys = sorted([k for k in new_state_dict.keys() if 'g_a' in k and 'weight' in k])
        if keys:
            M = new_state_dict[keys[-1]].size(0)
    except:
        pass
    
    print(f"Model parameters: N={N}, M={M}")
    
    # Choose model class based on N
    if N == 64:
        print("Detected TIC_Student model (N=64)")
        model = TIC_Student(N=N, M=M)
    else:
        print("Detected TIC teacher model")
        model = TIC(N=N, M=M)
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    return model


def export_onnx(checkpoint_path, output_dir, input_size=256, opset_version=17):
    """Export TIC model to ONNX format."""
    device = torch.device("cpu")
    
    # Load model
    full_model = load_checkpoint(checkpoint_path, device)
    
    # Get model parameters
    N = full_model.entropy_bottleneck.channels
    M = full_model.g_a[-1].out_channels if hasattr(full_model.g_a[-1], 'out_channels') else 192
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare dummy inputs
    # x: input image (batch, 3, H, W)
    # y: latent representation, downsampled 16x
    # z: hyperprior latent, downsampled 64x from image (4x from y)
    dummy_image = torch.randn(1, 3, input_size, input_size)
    y_size = input_size // 16
    z_size = input_size // 64
    dummy_y = torch.randn(1, M, y_size, y_size)
    dummy_z = torch.randn(1, N, z_size, z_size)
    
    print(f"\nDummy input shapes:")
    print(f"  Image: {dummy_image.shape}")
    print(f"  y: {dummy_y.shape}")
    print(f"  z: {dummy_z.shape}")
    
    # Export 1: Encoder
    encoder = NetEncoder(full_model)
    encoder_path = os.path.join(output_dir, "tic_encoder.onnx")
    print(f"\nExporting Encoder -> {encoder_path}")
    torch.onnx.export(
        encoder,
        dummy_image,
        encoder_path,
        opset_version=opset_version,
        input_names=["input_image"],
        output_names=["y", "z"],
        dynamic_axes={
            "input_image": {0: "batch", 2: "height", 3: "width"},
            "y": {0: "batch", 2: "h_y", 3: "w_y"},
            "z": {0: "batch", 2: "h_z", 3: "w_z"}
        }
    )
    print("✅ Encoder exported successfully")
    
    # Export 2: Hyper Decoder
    hyper_decoder = NetHyperDecoder(full_model)
    hyper_path = os.path.join(output_dir, "tic_hyper_decoder.onnx")
    print(f"\nExporting HyperDecoder -> {hyper_path}")
    torch.onnx.export(
        hyper_decoder,
        dummy_z,
        hyper_path,
        opset_version=opset_version,
        input_names=["z_hat"],
        output_names=["scales", "means"],
        dynamic_axes={
            "z_hat": {0: "batch", 2: "h_z", 3: "w_z"},
            "scales": {0: "batch", 2: "h_y", 3: "w_y"},
            "means": {0: "batch", 2: "h_y", 3: "w_y"}
        }
    )
    print("✅ HyperDecoder exported successfully")
    
    # Export 3: Main Decoder
    decoder = NetDecoder(full_model)
    decoder_path = os.path.join(output_dir, "tic_decoder.onnx")
    print(f"\nExporting Decoder -> {decoder_path}")
    torch.onnx.export(
        decoder,
        dummy_y,
        decoder_path,
        opset_version=opset_version,
        input_names=["y_hat"],
        output_names=["x_hat"],
        dynamic_axes={
            "y_hat": {0: "batch", 2: "h_y", 3: "w_y"},
            "x_hat": {0: "batch", 2: "height", 3: "width"}
        }
    )
    print("✅ Decoder exported successfully")
    
    # Summary
    print("\n" + "=" * 50)
    print("Export Complete!")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    for f in ["tic_encoder.onnx", "tic_hyper_decoder.onnx", "tic_decoder.onnx"]:
        fpath = os.path.join(output_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    print("\nNext steps:")
    print("1. Generate fixed_cdfs.py using dump_cdfs.py with the same checkpoint")
    print("2. Use nxinfcom.py to compress images")
    print("3. Use nxinfdec.py to decompress images")


def main():
    parser = argparse.ArgumentParser(description="Export TIC model to ONNX format")
    parser.add_argument("-p", "--checkpoint", type=str, required=True,
                        help="Path to .pth.tar checkpoint file")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Output directory for ONNX files (default: current directory)")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Input image size for tracing (default: 256)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    export_onnx(args.checkpoint, args.output_dir, args.input_size, args.opset)


if __name__ == "__main__":
    main()
