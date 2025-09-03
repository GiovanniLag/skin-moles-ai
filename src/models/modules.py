from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'silu':
        return nn.SiLU(inplace=True)
    if name == 'gelu':
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act='silu', bn=True):
        """Initialize ConvBNAct. Uses Convolution with specified parameters, followed by optional BatchNorm and activation.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        k : int, optional
            Kernel size. Default is 3.
        s : int, optional
            Stride. Default is 1.
        p : int or None, optional
            Padding. If None, it will be set to k // 2. Default is None.
        groups : int, optional
            Number of groups for grouped convolution. Default is 1.
        act : {'relu', 'silu', 'gelu'}, optional
            Activation function. Default is 'silu'.
        bn : bool, optional
            Whether to use BatchNorm. Default is True.

        Raises
        ------
        ValueError
            If an unknown activation name is provided.
        """
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=not bn)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, ch, r=16):
        """Initialize SEBlock.
        Parameters
        ----------
        ch : int
            Number of input channels.
        r : int, optional
            Reduction ratio. Default is 16.
        """
        super().__init__()
        hidden = max(8, ch // r)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1) # Global average pooling
        w = self.act(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class BasicBlockSE(nn.Module):
    """ResNet basic block (2x3x3) with optional downsample and SE."""
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, se_ratio=0.25, act='silu'):
        """Initialize BasicBlockSE.

        If out_ch != in_ch or stride != 1, a downsample layer is added to the identity connection.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        stride : int, optional
            Stride for the first convolution. Default is 1.
        se_ratio : float, optional
            Squeeze-and-Excitation ratio. If <=0, SE is not used. Default is 0.25.
        act : {'relu', 'silu', 'gelu'}, optional
            Activation function. Default is 'silu'.

        Raises
        ------
        ValueError
            If an unknown activation name is provided.
        """
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, k=3, s=stride, act=act)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, r=int(1/se_ratio)) if se_ratio>0 else nn.Identity()
        self.act = _act(act)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out

class GeM(nn.Module):
    """Generalized Mean Pooling with a learnable, positive p.
    
    To avoid NaN/Inf values during training (especially with mixed precision),
    we reparameterise p via softplus to keep it >0 and perform the pow/exp
    operations in float32.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        # use a raw parameter and apply softplus inside forward to guarantee positivity
        self.raw_p = nn.Parameter(torch.ones(1) * p, requires_grad=trainable)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype         # save the incoming dtype (could be float16 in mixed precision)
        x = x.float()           # do calculations in float32 to improve stability
        p = F.softplus(self.raw_p) + self.eps  # ensure p > eps
        # clamp x to avoid zeros, then perform generalized mean pooling
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p) # Does global avg pooling
        x = x.to(dtype)         # cast back to original dtype
        return x.view(x.size(0), -1)
