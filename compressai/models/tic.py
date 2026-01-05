import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.ans import BufferedRansEncoder, RansDecoder
from .utils import conv, update_registered_buffers

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    """
    Transposed convolution for upsampling.
    Uses Conv2d + PixelShuffle for Vitis AI compatibility.
    """
    internal_channels = out_channels * (stride ** 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            internal_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ),
        nn.PixelShuffle(upscale_factor=stride)
    )


class TIC(nn.Module):
    """
    Pure CNN Image Compression Model based on the hyperprior architecture.
    
    Architecture (matching reference diagram):
    - g_a: conv → GDN → conv → GDN → conv → GDN → conv
    - g_s: deconv → IGDN → deconv → IGDN → deconv → IGDN → deconv
    - h_a: abs → conv → ReLU → conv → ReLU → conv
    - h_s: deconv → ReLU → deconv → ReLU → conv

    Args:
        N (int): Number of channels (default: 128)
        M (int): Number of channels in the latent space (default: 192)
    """

    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = N
        self.M = M

        # ============================================
        # g_a: Analysis Transform (Encoder)
        # input image → y (latent representation)
        # ============================================
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),      # conv Mx5x5/2↓
        )

        # ============================================
        # g_s: Synthesis Transform (Decoder)
        # ŷ → reconstruction x̂
        # ============================================
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            GDN(N, inverse=True),                      # IGDN
            deconv(N, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            GDN(N, inverse=True),                      # IGDN
            deconv(N, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            GDN(N, inverse=True),                      # IGDN
            deconv(N, 3, kernel_size=5, stride=2),    # deconv 3x5x5/2↑
        )

        # ============================================
        # h_a: Hyper Analysis Transform
        # y → z (hyper latent)
        # ============================================
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),      # conv Nx3x3/1
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),      # conv Nx5x5/2↓
        )

        # ============================================
        # h_s: Hyper Synthesis Transform
        # ẑ → σ (scale parameters)
        # ============================================
        self.h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            nn.ReLU(inplace=True),
            deconv(N, N, kernel_size=5, stride=2),    # deconv Nx5x5/2↑
            nn.ReLU(inplace=True),
            conv(N, M, kernel_size=3, stride=1),      # conv Mx3x3/1
        )

        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        # Encoder: x → y
        y = self.g_a(x)
        
        # Hyper encoder: |y| → z
        z = self.h_a(torch.abs(y))
        
        # Entropy bottleneck for z
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        # Hyper decoder: ẑ → scales
        scales_hat = self.h_s(z_hat)
        
        # Quantize y
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        
        # Decoder: ŷ → x̂
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck module(s)."""
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values."""
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # Infer N and M from state dict
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes)
        
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        
        return {"x_hat": x_hat}