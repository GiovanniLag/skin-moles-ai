from __future__ import annotations
import torch.nn as nn

from modules import BasicBlockSE, ConvBNAct, GeM

class ResNetSEBackbone(nn.Module):
    """
    A ResNet-like backbone with Squeeze-and-Excitation blocks.
    """
    def __init__(self, in_channels, widths, layers, act, se_ratio, num_stages):
        super().__init__()
        assert len(widths) == len(layers) == num_stages
        pass


    def _make_stage(self, in_ch, out_ch, n_blocks, stride, act, se_ratio):
        blocks = []
        blocks.append(BasicBlockSE(in_ch, out_ch, stride=stride, se_ratio=se_ratio, act=act))
        for _ in range(1, n_blocks):
            blocks.append(BasicBlockSE(out_ch, out_ch, stride=1, se_ratio=se_ratio, act=act))
        return nn.Sequential(*blocks)