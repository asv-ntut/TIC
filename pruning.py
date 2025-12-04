# 檔案名稱: prune_tic.py (請確認使用此版本)

import torch
import torch.nn as nn
import torch_pruning as tp
from typing import Sequence

try:
    from compressai.models.tic import TIC
    from compressai.layers.layers import WindowAttention, CausalAttentionModule
    from compressai.entropy_models import EntropyBottleneck
    from torch_pruning.ops import _OutputOp
except ImportError as e:
    print(f"匯入錯誤: {e}")
    exit()


class EntropyBottleneckPruner(tp.BasePruningFunc):
    def get_out_channels(self, layer): return layer.channels

    def get_in_channels(self, layer): return layer.channels

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: return layer

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: return layer


class CausalAttentionPruner(tp.BasePruningFunc):
    def get_out_channels(self, layer): return layer.out_dim

    def get_in_channels(self, layer): return layer.dim

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: return layer

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module: return layer


class OutputPruner(tp.BasePruningFunc):
    def get_out_channels(self, layer):
        return None

    def get_in_channels(self, layer):
        if len(layer.shape) > 1:
            return layer.shape[1]
        else:
            return None

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        return layer

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        return layer


def main():
    print("正在建立 TIC 模型...")
    model = TIC(N=128, M=192)
    example_inputs = torch.randn(1, 3, 256, 256)

    print("=" * 50)
    print("模型剪枝前:")
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"  - MACs: {base_macs / 1e9:.2f} G")
    print(f"  - Parameters: {base_params / 1e6:.2f} M")
    print("=" * 50)

    importance = tp.importance.MagnitudeImportance(p=2, group_reduction=None, normalizer='max')
    ignored_layers = [
        model.g_a0, model.g_a2, model.g_a4, model.g_a6, model.h_a0, model.h_a2, model.h_a4,
        model.h_s0, model.h_s2, model.h_s4, model.g_s0, model.g_s2, model.g_s4, model.g_s6,
        model.entropy_bottleneck, model.context_prediction, model.entropy_parameters
    ]
    num_heads = {}
    for m in model.modules():
        if isinstance(m, WindowAttention):
            num_heads[m.qkv] = m.num_heads

    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs, importance=importance, pruning_ratio=0.3,
        global_pruning=True, num_heads=num_heads, ignored_layers=ignored_layers,
        root_module_types=[nn.Linear, nn.LayerNorm], output_transform=lambda out: out['x_hat'].sum(),
        customized_pruners={
            EntropyBottleneck: EntropyBottleneckPruner(),
            CausalAttentionModule: CausalAttentionPruner(),
            _OutputOp: OutputPruner(),
        },
    )

    print("\n正在執行剪枝...")
    pruner.step()
    print("剪枝完成。")

    print("正在更新剪枝後模型的 Attention scale 參數...")
    for m in model.modules():
        if isinstance(m, WindowAttention):
            new_dim = m.qkv.in_features
            m.dim = new_dim
            if m.num_heads > 0 and new_dim % m.num_heads == 0:
                head_dim = new_dim // m.num_heads
                m.scale = head_dim ** -0.5
    print("更新完成。")

    print("\n" + "=" * 50)
    print("模型剪枝後:")
    try:
        with torch.no_grad():
            model.eval()
            _ = model(example_inputs)
        print("\n[SUCCESS] 剪枝後模型前向傳播測試成功！")
    except Exception as e:
        print(f"\n[ERROR] 剪枝後模型前向傳播失敗: {e}")

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"  - MACs: {pruned_macs / 1e9:.2f} G")
    print(f"  - Parameters: {pruned_params / 1e6:.2f} M")
    print("=" * 50)

    print("\n剪枝成效分析:")
    print(
        f"  - MACs: {base_macs / 1e9:.2f} G -> {pruned_macs / 1e9:.2f} G (減少了 {(base_macs - pruned_macs) / base_macs:.2%})")
    print(
        f"  - Parameters: {base_macs / 1e6:.2f} M -> {pruned_params / 1e6:.2f} M (減少了 {(base_params - pruned_params) / base_params:.2%})")
    print("\n")


if __name__ == "__main__":
    main()