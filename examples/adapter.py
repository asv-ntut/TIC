import torch
import torch.nn as nn

class BottleneckAdapter(nn.Module):
    """
    一個基於 Inverted Residual Block 的高效 Adapter
    """
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()
        # 隱藏層的通道數
        hidden_channels = int(in_channels * expand_ratio)

        self.block = nn.Sequential(
            # 1. 擴張層 (Pointwise Conv): 使用 1x1 Conv 增加通道數
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

            # 2. 處理層 (Depthwise Conv): 使用 3x3 Depthwise Conv 提取空間特徵
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels, bias=False),
            nn.ReLU(inplace=True),

            # 3. 壓縮層 (Pointwise Conv): 使用 1x1 Conv 降回目標通道數
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # 如果輸入和輸出通道數不同，需要一個 1x1 Conv 來匹配維度以便相加
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # 殘差連接
        return self.shortcut(x) + self.block(x)