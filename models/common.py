# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from torch.cuda.amp import autocast
import torch.nn.functional as F
# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# Modify Conv class to optionally include ECA (for Step 3: add to specific Conv layers via YAML args)
class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, eca=False):  # Added eca flag
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.eca = ECA(c2) if eca else nn.Identity()  # Add ECA if flag is True

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.eca(x)  # Apply ECA

    def forward_fuse(self, x):
        x = self.act(self.conv(x))
        return self.eca(x)


class DWConv(Conv):
    """Implements a depth-wise convolution layer with optional activation for efficient spatial filtering."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """A depth-wise transpose convolutional layer for upsampling in neural networks, particularly in YOLOv5 models."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

# New CustomLayerNorm to handle AMP dtype conversion internally (fixes Float/Half mismatch)
class CustomLayerNorm(nn.Module):
    """Custom LayerNorm that ensures float32 params and computation for AMP compatibility."""

    def __init__(self, normalized_shape):
        super().__init__()
        # Explicitly set norm parameters to float32 to match computation
        self.norm = nn.LayerNorm(normalized_shape, dtype=torch.float32)

    def forward(self, input):
        original_dtype = input.dtype
        # Convert input to float32 for computation
        input_float = input.to(torch.float32)
        # Ensure norm parameters are float32 (defensive)
        self.norm.to(dtype=torch.float32)
        out = self.norm(input_float)
        return out.to(original_dtype)

# Modified TransformerLayer for stability (re-add LayerNorm to prevent NaN, with initialization)
class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.c = c
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.scale = self.head_dim ** -0.5

        # 减少参数量，使用单一线性层替代多个
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.proj = nn.Linear(c, c, bias=False)

        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)

        # 使用更轻量的FFN
        self.mlp = nn.Sequential(
            nn.Linear(c, c * 2, bias=False),
            nn.SiLU(),
            nn.Linear(c * 2, c, bias=False)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        # 保存输入用于残差连接
        residual = x

        # 第一个LayerNorm
        x = self.norm1(x)

        # QKV投影 - 更高效的实现
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 高效的注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        # 第一个残差连接
        x = x + residual

        # 第二个LayerNorm和FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)

        # 第二个残差连接
        x = x + residual

        return x

# Modified TransformerBlock for stability (add LayerNorm and better initialization)
class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.c2 = c2

        # 如果输入通道数不等于输出通道数，添加一个卷积层
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, c2))  # 固定大小的位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(c2, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(c2)

    def forward(self, x):
        # 应用卷积层（如果需要）
        if self.conv is not None:
            x = self.conv(x)

        # 保存形状
        b, c, h, w = x.shape

        # 重塑为序列
        x = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # 添加位置编码（裁剪或插值到正确的大小）
        if h * w <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :h * w, :]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=h * w,
                mode='linear'
            ).permute(0, 2, 1)

        x = x + pos_embed

        # 应用Transformer层
        for layer in self.layers:
            x = layer(x)

        # 应用最终的LayerNorm
        x = self.norm(x)

        # 重塑回原始形状
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        return x


class Bottleneck(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP bottleneck layer for feature extraction with cross-stage partial connections and optional shortcuts."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    """Implements a cross convolution layer with downsampling, expansion, and optional shortcut."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """Extends the C3 module with cross-convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

# Modified C3TR class for stability (add LayerNorm and better initialization)
class C3TR(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Process through the standard C3 forward path
        # But ensure proper type handling
        original_dtype = x.dtype

        # Process through convolutions
        a = self.cv1(x)
        b = self.m(a)  # TransformerBlock handles type conversion internally
        c = self.cv2(x)

        # Concatenate and final convolution
        return self.cv3(torch.cat((b, c), 1)).to(dtype=original_dtype)


class C3SPP(C3):
    """Extends the C3 module with an SPP layer for enhanced spatial feature extraction and customizable channels."""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    """Implements a C3 module with Ghost Bottlenecks for efficient feature extraction in YOLOv5."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    """Implements Spatial Pyramid Pooling (SPP) for feature extraction, ref: https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input channels), c2 (output channels), k (kernel sizes)."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv5 models."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    """Focuses spatial information into channel space using slicing and convolution for efficient feature extraction."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate width-height info into channel space with configurable convolution
        parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Processes input through Focus mechanism, reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Implements Ghost Convolution for efficient feature extraction, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups, and activation; halves out channels
        for efficiency.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Performs forward pass, concatenating outputs of two convolutions on input `x`: shape (B,C,H,W)."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Efficient bottleneck layer using Ghost Convolutions, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    """Contracts spatial dimensions into channel dimensions for efficient processing in neural networks."""

    def __init__(self, gain=2):
        """Initializes a layer to contract spatial dimensions (width-height) into channels, e.g., input shape
        (1,64,80,80) to (1,256,40,40).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor to expand channel dimensions by contracting spatial dimensions, yielding output shape
        `(b, c*s*s, h//s, w//s)`.
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    """Expands spatial dimensions by redistributing channels, e.g., from (1,64,80,80) to (1,16,160,160)."""

    def __init__(self, gain=2):
        """
        Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional gain
        factor.

        Example: x(1,64,80,80) to x(1,16,160,160).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor x to expand spatial dimensions by redistributing channels, requiring C / gain^2 ==
        0.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    """Concatenates tensors along a specified dimension for efficient tensor manipulation in neural networks."""

    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)

# Add ECA attention class (from ECA-Net paper, simple 1D conv for channel attention)
class ECA(nn.Module):
    """改进的ECA模块，通道数自适应"""

    def __init__(self, c1=None, gamma=2, b=1):
        super().__init__()
        self.gamma = gamma
        self.b = b

        if c1 is None:
            # 如果没有指定通道数，在forward中自动获取
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = None
        else:
            # 如果指定了通道数，预先计算卷积核大小
            t = int(abs(math.log(c1, 2) + self.b) / self.gamma)
            k = t if t % 2 else t + 1
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        if self.conv is None:
            # 动态确定卷积核大小
            c = x.shape[1]
            t = int(abs(math.log(c, 2) + self.b) / self.gamma)
            k = t if t % 2 else t + 1
            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False).to(x.device)

        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SimpleECA(nn.Module):
    """简化版ECA模块 - 更稳定的实现"""
    def __init__(self, c1, k_size=3):  # c1是输入通道数
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 特征聚合
        y = self.avg_pool(x)

        # 应用1D卷积
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # 应用通道注意力
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# =======================================================================================================================
class EnhancedECA(nn.Module):
    """增强版ECA - 专为小目标检测优化"""

    def __init__(self, c1, gamma=2, b=1):
        super().__init__()
        # 自适应卷积核大小计算
        k = int(abs((math.log(c1, 2) + b) / gamma))
        k = k if k % 2 else k + 1

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D卷积
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 增加最大池化分支增强小目标响应
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_max = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        # 特征融合权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        # 平均池化分支
        y_avg = self.avg_pool(x)
        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_avg = self.sigmoid(y_avg)

        # 最大池化分支
        y_max = self.max_pool(x)
        y_max = self.conv_max(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.sigmoid(y_max)

        # 自适应融合
        y = self.alpha * y_avg + (1 - self.alpha) * y_max

        return x * y.expand_as(x)


class EnhancedSimpleECA(nn.Module):
    """基于test3成功的SimpleECA，针对UAV数据集特性优化"""

    def __init__(self, c1, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 增加最大池化分支

        # 自适应卷积核大小
        if c1 <= 256:
            k_size = 3  # 小通道数用小核
        else:
            k_size = 5  # 大通道数用大核

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 融合权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.6)  # 偏向平均池化

    def forward(self, x):
        # 平均池化分支
        y_avg = self.avg_pool(x)
        y_avg = y_avg.squeeze(-1).transpose(-1, -2)
        y_avg = self.conv(y_avg)
        y_avg = y_avg.transpose(-1, -2).unsqueeze(-1)

        # 最大池化分支
        y_max = self.max_pool(x)
        y_max = y_max.squeeze(-1).transpose(-1, -2)
        y_max = self.conv(y_max)
        y_max = y_max.transpose(-1, -2).unsqueeze(-1)

        # 自适应融合
        y = self.alpha * y_avg + (1 - self.alpha) * y_max
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class DetectMultiBackend(nn.Module):
    """YOLOv5 MultiBackend class for inference on various backends including PyTorch, ONNX, TensorRT, and more."""

    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes DetectMultiBackend with support for various inference backends, including PyTorch and ONNX."""
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlpackage
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_profile_shape(name, 0)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wraps a TensorFlow GraphDef for inference, returning a pruned function."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                """Generates a sorted list of graph outputs excluding NoOp nodes and inputs, formatted as '<name>:0'."""
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
        # PaddlePaddle
        elif paddle:
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle>=3.0.0")
            import paddle.inference as pdi

            w = Path(w)
            if w.is_dir():
                model_file = next(w.rglob("*.json"), None)
                params_file = next(w.rglob("*.pdiparams"), None)
            elif w.suffix == ".pdiparams":
                model_file = w.with_name("model.json")
                params_file = w
            else:
                raise ValueError(f"Invalid model path {w}. Provide model directory or a .pdiparams file.")

            if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
                raise FileNotFoundError(f"Model files not found in {w}. Both .json and .pdiparams files are required.")

            config = pdi.Config(str(model_file), str(params_file))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()

        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """Performs YOLOv5 inference on input images with options for augmentation and visualization."""
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            if len(y) == 2 and len(y[1].shape) != 4:
                y = list(reversed(y))
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Performs a single inference warmup to initialize model weights, accepting an `imgsz` tuple for image size."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines model type from file path or URL, supporting various export formats.

        Example: path='path/to/model.onnx' -> type=onnx
        """
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """Loads metadata from a YAML file, returning strides and names if the file exists, otherwise `None`."""
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    """AutoShape class for robust YOLOv5 inference with preprocessing, NMS, and support for various input formats."""

    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        """Initializes YOLOv5 model for inference, setting up attributes and preparing model for evaluation."""
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        """
        Applies to(), cpu(), cuda(), half() etc.

        to model tensors excluding parameters or registered buffers.
        """
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        """
        Performs inference on inputs with optional augment & profiling.

        Supports various formats including file, URI, OpenCV, PIL, numpy, torch.
        """
        # For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f"image{i}"  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
                files.append(Path(f).with_suffix(".jpg").name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(
                    y if self.dmb else y[0],
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic,
                    self.multi_label,
                    max_det=self.max_det,
                )  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    """Manages YOLOv5 detection results with methods for visualization, saving, cropping, and exporting detections."""

    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        """Initializes the YOLOv5 Detections class with image info, predictions, filenames, timing and normalization."""
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        """Executes model predictions, displaying and/or saving outputs with optional crops and labels."""
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(", ")
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))
                    im = annotator.im
            else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display

                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays detection results with optional labels.

        Usage: show(labels=True)
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves detection results with optional labels to a specified directory.

        Usage: save(labels=True, save_dir='runs/detect/exp', exist_ok=False)
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detection results, optionally saves them to a directory.

        Args: save (bool), save_dir (str), exist_ok (bool).
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """Renders detection results with optional labels on images; args: labels (bool) indicating label inclusion."""
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Returns detections as pandas DataFrames for various box formats (xyxy, xyxyn, xywh, xywhn).

        Example: print(results.pandas().xyxy[0]).
        """
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        """
        Converts a Detections object into a list of individual detection results for iteration.

        Example: for result in results.tolist():
        """
        r = range(self.n)  # iterable
        return [
            Detections(
                [self.ims[i]],
                [self.pred[i]],
                [self.files[i]],
                self.times,
                self.names,
                self.s,
            )
            for i in r
        ]

    def print(self):
        """Logs the string representation of the current object's state via the LOGGER."""
        LOGGER.info(self.__str__())

    def __len__(self):
        """Returns the number of results stored, overrides the default len(results)."""
        return self.n

    def __str__(self):
        """Returns a string representation of the model's results, suitable for printing, overrides default
        print(results).
        """
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """Returns a string representation of the YOLOv5 object, including its class and formatted results."""
        return f"YOLOv5 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    """YOLOv5 mask Proto module for segmentation models, performing convolutions and upsampling on input tensors."""

    def __init__(self, c1, c_=256, c2=32):
        """Initializes YOLOv5 Proto module for segmentation with input, proto, and mask channels configuration."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass using convolutional layers and upsampling on input tensor `x`."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    """YOLOv5 classification head with convolution, pooling, and dropout layers for channel transformation."""

    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """Initializes YOLOv5 classification head with convolution, pooling, and dropout layers for input to output
        channel transformation.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Processes input through conv, pool, drop, and linear layers; supports list concatenation input."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

# ADD CBAM,BiFPN
class ChannelAttentionModule(nn.Module):
    def __init__(self, c_in, reduction=16): # c_in 是输入通道数
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c_in // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c_in, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c_in)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out


class CBAM(nn.Module):
    """改进的CBAM模块 - 同时支持channel_gate和channel_attention属性"""

    def __init__(self, c1, reduction=16):
        super().__init__()
        # 通道注意力
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // reduction, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(c1 // reduction, c1, 1, 1, 0),
        )
        # 添加别名，使channel_attention指向channel_gate
        self.channel_attention = self.channel_gate

        # 空间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_gate(x)
        channel_att = torch.sigmoid(channel_att)
        x = x * channel_att

        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_gate(spatial_input)
        x = x * spatial_att

        return x


class BiFPN_Add2(nn.Module):
    def __init__(self, c2):
        super().__init__()
        # 学习权重
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 确保输出通道数正确
        self.conv1 = Conv(c2, c2, k=1)  # 第一个输入的调整卷积
        self.conv2 = Conv(c2, c2, k=1)  # 第二个输入的调整卷积
        self.act = nn.SiLU()

    def forward(self, inputs):
        # 确保输入是列表
        assert isinstance(inputs, list) and len(inputs) == 2

        # 调整通道数
        x1 = self.conv1(inputs[0])
        x2 = self.conv2(inputs[1])

        # 加权融合
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.act(weight[0] * x1 + weight[1] * x2)

# 三个特征图add操作
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))


class AdaptiveAttention(nn.Module):
    """自适应注意力模块 - 使用GroupNorm避免BatchNorm在1x1特征图上的问题"""

    def __init__(self, c1):
        super().__init__()
        # 通道注意力 - 使用自定义卷积避免BatchNorm问题
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, kernel_size=1),
            nn.GroupNorm(1, c1 // 16),  # 替换BatchNorm
            nn.SiLU(),
            nn.Conv2d(c1 // 16, c1, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 注意力权重平衡器
        self.balance = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        # 通道注意力
        ca_out = self.ca(x) * x

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_out = self.sa(sa_input) * x

        # 自适应融合通道和空间注意力
        w = F.softmax(self.balance, dim=0)
        out = w[0] * ca_out + w[1] * sa_out

        return out


class MultiScaleFeatureEnhancer(nn.Module):
    """优化的多尺度特征增强模块 - 使用GroupNorm避免BatchNorm问题"""

    def __init__(self, c1):
        super().__init__()
        c_ = c1 // 2  # 减少通道数以提高效率

        # 小目标特征增强 - 小卷积核和密集连接
        self.small_branch = nn.Sequential(
            Conv(c1, c_, k=1),
            Conv(c_, c_ // 2, k=3, g=c_ // 2),  # 深度可分离卷积
            Conv(c_ // 2, c_ // 2, k=3, g=c_ // 4),  # 进一步细化特征
            Conv(c_ // 2, c_, k=1)  # 恢复通道数
        )

        # 中等目标特征增强
        self.medium_branch = nn.Sequential(
            Conv(c1, c_, k=1),
            Conv(c_, c_, k=3),
            Conv(c_, c_, k=1)
        )

        # 特征融合 - 使用注意力机制
        self.fusion = nn.Sequential(
            Conv(c_ * 2, c1, k=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 小目标特征
        small_feat = self.small_branch(x)

        # 中等目标特征
        medium_feat = self.medium_branch(x)

        # 特征融合
        fused_attention = self.fusion(torch.cat([small_feat, medium_feat], dim=1))

        # 应用注意力并添加残差连接
        return x * fused_attention + x


class BackgroundContextModule(nn.Module):
    """背景上下文模块 - 增强对复杂背景中目标的检测能力"""

    def __init__(self, c1):
        super().__init__()

        # 全局上下文 - 使用GroupNorm替代BatchNorm避免单样本问题
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, kernel_size=1),
            nn.GroupNorm(1, c1 // 16),  # 使用GroupNorm替代BatchNorm
            nn.SiLU(),
            nn.Conv2d(c1 // 16, c1, kernel_size=1),
            nn.Sigmoid()
        )

        # 局部上下文 - 多尺度感受野
        self.local_branch1 = Conv(c1, c1 // 2, k=3, d=1)  # 标准卷积
        self.local_branch2 = Conv(c1, c1 // 2, k=3, d=3)  # 空洞卷积

        # 特征融合
        self.fusion = Conv(c1, c1, k=1)

    def forward(self, x):
        # 全局上下文
        global_feat = self.global_context(x) * x

        # 局部上下文
        local_feat1 = self.local_branch1(x)
        local_feat2 = self.local_branch2(x)
        local_feat = torch.cat([local_feat1, local_feat2], dim=1)
        local_feat = self.fusion(local_feat)

        # 融合全局和局部特征
        return global_feat + local_feat


class EdgeEnhancementModule(nn.Module):
    """边缘增强模块 - 增强无人机目标的边缘特征"""

    def __init__(self, c1):
        super().__init__()
        # 边缘检测卷积
        self.edge_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),
            nn.BatchNorm2d(c1),
        )

        # 特征融合
        self.fusion = Conv(c1 * 2, c1, k=1)

    def forward(self, x):
        # 提取边缘特征
        edge_feat = self.edge_conv(x)

        # 与原始特征融合
        return self.fusion(torch.cat([x, edge_feat], dim=1))


class DroneFeatureEnhancer(nn.Module):
    """专为无人机目标检测设计的特征增强模块"""

    def __init__(self, c1):
        super().__init__()
        # 边缘特征增强 - 使用深度可分离卷积减少参数
        self.edge_branch = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),  # 深度卷积
            nn.Conv2d(c1, c1, kernel_size=1),  # 逐点卷积
            nn.SiLU()
        )

        # 空间注意力 - 轻量级实现
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),  # 减小卷积核大小
            nn.Sigmoid()
        )

        # 特征融合 - 简单残差连接
        self.fusion = Conv(c1, c1, k=1)

        # 添加少量dropout防止过拟合
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        # 边缘特征增强
        edge_feat = self.edge_branch(x)

        # 空间注意力
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_feat, max_feat], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)

        # 应用空间注意力到边缘特征
        enhanced = edge_feat * spatial_weight

        # 融合并添加少量dropout
        output = self.fusion(enhanced) + x
        return self.dropout(output)


class MultiscaleDroneEnhancer(nn.Module):
    """改进的多尺度无人机特征增强器 - 同时处理小目标和复杂背景"""

    def __init__(self, c1):
        super().__init__()
        # 小目标分支 - 使用深度可分离卷积和小卷积核
        self.small_branch = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),  # 深度卷积
            nn.Conv2d(c1, c1, kernel_size=1),  # 逐点卷积
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),  # 再次深度卷积增强边缘
            nn.Conv2d(c1, c1, kernel_size=1),  # 逐点卷积
            nn.SiLU()
        )

        # 边缘增强分支 - 使用Sobel算子思想
        self.edge_branch = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.GroupNorm(32, c1),  # 使用GroupNorm提高训练稳定性
            nn.SiLU()
        )

        # 注意力机制 - 轻量级实现
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 * 2, c1 // 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c1 // 8, c1, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = Conv(c1 * 2, c1, k=1)

        # 添加少量dropout防止过拟合
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 小目标特征增强
        small_feat = self.small_branch(x)

        # 边缘特征增强
        edge_feat = self.edge_branch(x)

        # 特征融合
        combined = torch.cat([small_feat, edge_feat], dim=1)
        attention = self.attention(combined)

        # 应用注意力
        enhanced = torch.cat([small_feat * attention, edge_feat * attention], dim=1)
        output = self.fusion(enhanced) + x

        return self.dropout(output)


class AdaptiveSpatialAttention(nn.Module):
    """改进的空间注意力模块 - 专注于中等尺寸目标"""

    def __init__(self, c1):
        super().__init__()
        # 多尺度空间特征提取
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1)  # 小感受野
        self.conv2 = nn.Conv2d(2, 1, kernel_size=5, padding=2)  # 中感受野
        self.conv3 = nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 大感受野

        # 特征融合
        self.fusion = nn.Conv2d(3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 通道注意力 - 轻量级实现
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c1 // 16, c1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 生成空间特征图
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_feat, max_feat], dim=1)

        # 多尺度空间注意力
        attn1 = self.conv1(spatial_input)
        attn2 = self.conv2(spatial_input)
        attn3 = self.conv3(spatial_input)

        # 融合多尺度注意力
        spatial_attn = self.sigmoid(self.fusion(torch.cat([attn1, attn2, attn3], dim=1)))

        # 通道注意力
        channel_attn = self.ca(x)

        # 应用注意力
        return x * spatial_attn * channel_attn


class ImprovedDroneEnhancer(nn.Module):
    """改进的无人机特征增强器 - 专注于边缘和细节增强"""

    def __init__(self, c1):
        super().__init__()
        # 特征提取分支 - 使用深度可分离卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),  # 深度卷积
            nn.Conv2d(c1, c1, kernel_size=1),  # 逐点卷积
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )

        # 边缘增强分支 - 使用不同膨胀率的卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=2, dilation=2),  # 膨胀卷积
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )

        # 通道注意力 - 使用SE风格的注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 8, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c1 // 8, c1, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = Conv(c1 * 2, c1, k=1)

    def forward(self, x):
        # 特征提取
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)

        # 特征融合
        fused = torch.cat([feat1, feat2], dim=1)
        fused = self.fusion(fused)

        # 应用通道注意力
        att = self.se(fused)

        # 残差连接
        return x + fused * att


class EnhancedBiFPN(nn.Module):
    """增强型BiFPN模块 - 更高效的特征融合"""

    def __init__(self, c1):
        super().__init__()
        # 学习权重
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

        # 特征变换
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),  # 深度卷积
            nn.Conv2d(c1, c1, kernel_size=1),  # 逐点卷积
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )

        # 空间注意力
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 特征变换
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)

        # 加权融合
        w = F.softmax(self.w, dim=0)
        fused = w[0] * feat1 + w[1] * feat2

        # 空间注意力
        avg_out = torch.mean(fused, dim=1, keepdim=True)
        max_out, _ = torch.max(fused, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        att = self.sa(spatial)

        # 应用注意力并添加残差连接
        return x + fused * att


class MixedReceptiveFieldBlock(nn.Module):
    """混合感受野模块 - 同时处理不同尺度的目标"""
    def __init__(self, c1):
        super().__init__()
        c_ = c1 // 2

        # 小感受野分支 - 适合小目标
        self.branch1 = nn.Sequential(
            Conv(c1, c_, k=1),
            Conv(c_, c_, k=3, p=1)
        )

        # 中感受野分支 - 适合中等目标
        self.branch2 = nn.Sequential(
            Conv(c1, c_, k=1),
            Conv(c_, c_, k=3, p=3, d=3)  # 膨胀卷积增大感受野
        )

        # 特征融合
        self.fusion = Conv(c_ * 2, c1, k=1)

        # 通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c1 // 16, c1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多分支特征提取
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)

        # 特征融合
        fused = self.fusion(torch.cat([feat1, feat2], dim=1))

        # 应用通道注意力
        att = self.ca(fused)

        # 残差连接
        return x + fused * att


class BoundaryAwareModule(nn.Module):
    """边界感知模块 - 增强无人机轮廓检测"""
    def __init__(self, c1):
        super().__init__()
        # Sobel算子风格的边缘检测
        self.edge_conv_x = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)
        self.edge_conv_y = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1, bias=False)

        # 初始化为Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        # 扩展为适合通道数的卷积核
        sobel_x = sobel_x.reshape(1, 1, 3, 3).repeat(c1, 1, 1, 1)
        sobel_y = sobel_y.reshape(1, 1, 3, 3).repeat(c1, 1, 1, 1)

        # 设置卷积权重
        self.edge_conv_x.weight = nn.Parameter(sobel_x)
        self.edge_conv_y.weight = nn.Parameter(sobel_y)

        # 特征融合
        self.fusion = Conv(c1 * 3, c1, k=1)

    def forward(self, x):
        # 边缘特征提取
        edge_x = self.edge_conv_x(x)
        edge_y = self.edge_conv_y(x)

        # 特征融合
        return self.fusion(torch.cat([x, edge_x, edge_y], dim=1))

#==
class MultiScaleFeaturePyramid(nn.Module):
    """多尺度特征金字塔 - 简化稳定版本"""

    def __init__(self, c1):
        super().__init__()
        # 多尺度特征提取 - 都使用卷积操作，避免池化问题
        self.scale1 = Conv(c1, c1 // 4, k=1)     # 1x1卷积
        self.scale2 = Conv(c1, c1 // 4, k=3, p=1)  # 3x3卷积
        self.scale3 = Conv(c1, c1 // 4, k=5, p=2)  # 5x5卷积
        self.scale4 = Conv(c1, c1 // 4, k=7, p=3)  # 7x7卷积

        # 特征融合和注意力
        self.fusion = Conv(c1, c1, k=1)
        self.attention = nn.Sequential(
            Conv(c1, c1 // 16, k=1),
            nn.SiLU(),
            Conv(c1 // 16, c1, k=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        feat4 = self.scale4(x)

        # 特征融合
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        fused = self.fusion(fused)

        # 应用注意力
        att = self.attention(fused)

        return x + fused * att


class AdaptiveReceptiveFieldConv(nn.Module):
    """自适应感受野卷积 - 基于Deformable Convolution思想的轻量化实现"""

    def __init__(self, c1):
        super().__init__()
        # 主要特征提取分支
        self.main_conv = Conv(c1, c1, k=3, p=1)

        # 自适应权重生成
        self.offset_conv = nn.Sequential(
            Conv(c1, c1 // 4, k=1),
            Conv(c1 // 4, c1 // 4, k=3, p=1),
            nn.Conv2d(c1 // 4, 9, kernel_size=1)  # 3x3卷积核的9个权重
        )

        # 特征融合
        self.fusion = Conv(c1 * 2, c1, k=1)

    def forward(self, x):
        # 主特征
        main_feat = self.main_conv(x)

        # 生成自适应权重
        weights = torch.softmax(self.offset_conv(x), dim=1)  # [B, 9, H, W]

        # 应用自适应权重到邻域特征
        b, c, h, w = x.shape
        x_unfold = F.unfold(x, kernel_size=3, padding=1)  # [B, C*9, H*W]
        x_unfold = x_unfold.view(b, c, 9, h, w)

        # 加权聚合
        weights = weights.unsqueeze(1)  # [B, 1, 9, H, W]
        adaptive_feat = torch.sum(x_unfold * weights, dim=2)  # [B, C, H, W]

        return self.fusion(torch.cat([main_feat, adaptive_feat], dim=1))


class ContextGuidedDetectionHead(nn.Module):
    """上下文引导检测头 - 修复版本"""

    def __init__(self, c1):
        super().__init__()
        # 局部上下文分支
        self.local_branch = nn.Sequential(
            Conv(c1, c1 // 2, k=3, p=1),
            Conv(c1 // 2, c1 // 2, k=3, p=1)
        )

        # 全局上下文分支 - 修复全局池化问题
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # 改为2x2避免1x1
            Conv(c1, c1 // 4, k=1),
            nn.SiLU(),
            Conv(c1 // 4, c1 // 2, k=2),  # 2x2卷积处理2x2特征
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU()
        )

        # 上下文融合注意力
        self.context_att = nn.Sequential(
            Conv(c1, c1 // 8, k=1),
            nn.SiLU(),
            Conv(c1 // 8, c1, k=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = Conv(c1, c1, k=1)

    def forward(self, x):
        h, w = x.shape[-2:]

        # 局部上下文
        local_feat = self.local_branch(x)

        # 全局上下文
        global_feat = self.global_branch(x)  # [B, c1//2, 1, 1]
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)

        # 上下文融合
        context_feat = torch.cat([local_feat, global_feat], dim=1)
        context_att = self.context_att(context_feat)

        enhanced_feat = x * context_att
        return self.fusion(enhanced_feat + x)


class BackgroundSuppressionModule(nn.Module):
    """背景抑制模块 - 修复版本"""

    def __init__(self, c1):
        super().__init__()
        # 前景特征增强器
        self.fg_enhancer = nn.Sequential(
            Conv(c1, c1 // 2, k=1),
            Conv(c1 // 2, c1 // 2, k=3, p=1),
            Conv(c1 // 2, c1, k=1)
        )

        # 背景特征抑制器 - 保持4x4池化，这个大小相对安全
        self.bg_suppressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            Conv(c1, c1 // 4, k=1),
            nn.SiLU(),
            Conv(c1 // 4, c1, k=4),  # 4x4卷积处理4x4特征
            nn.Sigmoid()
        )

        # 对比注意力
        self.contrast_att = nn.Sequential(
            Conv(c1 * 2, c1 // 4, k=1),
            nn.SiLU(),
            Conv(c1 // 4, c1, k=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, w = x.shape[-2:]

        # 前景特征增强
        fg_feat = self.fg_enhancer(x)

        # 背景特征抑制
        bg_weight = self.bg_suppressor(x)  # [B, c1, 1, 1]
        bg_weight = F.interpolate(bg_weight, size=(h, w), mode='bilinear', align_corners=False)

        # 对比学习
        contrast_input = torch.cat([fg_feat, x * (1 - bg_weight)], dim=1)
        contrast_weight = self.contrast_att(contrast_input)

        return x * contrast_weight + fg_feat * (1 - contrast_weight)


class SmallTargetPreservation(nn.Module):
    """小目标保护模块 - 防止小目标特征在下采样中丢失"""

    def __init__(self, c1, c2):
        super().__init__()
        # 小目标特征提取
        self.small_feat_extract = nn.Sequential(
            Conv(c1, c1, k=1),
            Conv(c1, c1, k=3, p=1, g=c1),  # 深度卷积保持细节
            Conv(c1, c2, k=1)
        )

        # 注意力权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, 1),
            nn.SiLU(),
            nn.Conv2d(c1 // 16, c1, 1),
            nn.Sigmoid()
        )

        # 下采样
        self.downsample = Conv(c1, c2, k=3, s=2)

    def forward(self, x):
        # 提取小目标特征
        small_feat = self.small_feat_extract(x)

        # 生成注意力权重
        att_weight = self.attention(x)

        # 增强小目标特征
        enhanced_x = x * (1 + att_weight)

        # 下采样
        downsampled = self.downsample(enhanced_x)

        # 上采样小目标特征并融合
        small_feat_up = F.interpolate(small_feat, size=downsampled.shape[-2:], mode='bilinear', align_corners=False)

        return downsampled + small_feat_up


class EnhancedBiFPNBlock(nn.Module):
    """增强型BiFPN块 - 稳定版本"""

    def __init__(self, c1, c2):
        super().__init__()
        # 可学习权重
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

        # 预定义的通道转换层（支持常见的通道数）
        self.conv_64_to_c2 = Conv(64, c2, k=1) if c2 != 64 else nn.Identity()
        self.conv_128_to_c2 = Conv(128, c2, k=1) if c2 != 128 else nn.Identity()
        self.conv_256_to_c2 = Conv(256, c2, k=1) if c2 != 256 else nn.Identity()
        self.conv_512_to_c2 = Conv(512, c2, k=1) if c2 != 512 else nn.Identity()
        self.conv_1024_to_c2 = Conv(1024, c2, k=1) if c2 != 1024 else nn.Identity()

        # 注意力增强
        self.attention = nn.Sequential(
            Conv(c2, c2 // 4, k=1),
            nn.SiLU(),
            Conv(c2 // 4, c2, k=1),
            nn.Sigmoid()
        )

    def _convert_channels(self, x, target_channels):
        """根据输入通道数选择合适的转换层"""
        in_channels = x.shape[1]

        if in_channels == target_channels:
            return x
        elif in_channels == 64:
            return self.conv_64_to_c2(x)
        elif in_channels == 128:
            return self.conv_128_to_c2(x)
        elif in_channels == 256:
            return self.conv_256_to_c2(x)
        elif in_channels == 512:
            return self.conv_512_to_c2(x)
        elif in_channels == 1024:
            return self.conv_1024_to_c2(x)
        else:
            # 动态创建转换层作为后备方案
            conv_layer = Conv(in_channels, target_channels, k=1).to(x.device)
            return conv_layer(x)

    def forward(self, inputs):
        # 处理单个输入和多个输入的情况
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        target_channels = self.attention[0].conv.out_channels * 4  # 从attention层推断目标通道数

        # 转换所有输入到目标通道数
        processed_inputs = [self._convert_channels(x, target_channels) for x in inputs]

        if len(processed_inputs) == 1:
            # 单输入情况
            fused = processed_inputs[0]
        elif len(processed_inputs) == 2:
            # 双路径融合
            x1, x2 = processed_inputs
            w = F.softmax(self.w1, dim=0)
            fused = w[0] * x1 + w[1] * x2
        else:
            # 多路径融合
            w_size = min(len(processed_inputs), len(self.w2))
            w = F.softmax(self.w2[:w_size], dim=0)
            fused = sum(w[i] * processed_inputs[i] for i in range(w_size))

        # 应用注意力
        att = self.attention(fused)
        return fused * att


class P2FeatureExtractor(nn.Module):
    """P2特征提取器 - 确保正确的输出通道"""

    def __init__(self, c1, c2):
        super().__init__()
        # 浅层特征保护
        self.feat_protect = nn.Sequential(
            Conv(c1, c2, k=1),
            Conv(c2, c2, k=3, p=1, g=c2),  # 深度卷积保持空间信息
            Conv(c2, c2, k=1)
        )

        # 细节增强
        self.detail_enhance = nn.Sequential(
            Conv(c2, c2 // 2, k=1),
            Conv(c2 // 2, c2 // 2, k=5, p=2),  # 大卷积核捕获更多上下文
            Conv(c2 // 2, c2, k=1)
        )

    def forward(self, x):
        feat = self.feat_protect(x)
        enhanced = self.detail_enhance(feat)
        return feat + enhanced


class LightweightAttention(nn.Module):
    """轻量级注意力 - 基于test2成功经验"""

    def __init__(self, c1):
        super().__init__()
        # 简化的通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(c1 // 16, c1, 1, bias=False),
            nn.Sigmoid()
        )

        # 简化的空间注意力
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca_weight = self.ca(x)
        x_ca = x * ca_weight

        # 空间注意力
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.sa(sa_input)

        return x_ca * sa_weight


class SmallTargetEnhancer(nn.Module):
    """简化的小目标增强器 - 避免过度复杂"""

    def __init__(self, c1):
        super().__init__()
        # 轻量级边缘增强
        self.edge_conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),  # 深度卷积
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(c1)
        )

        # 残差连接权重
        self.alpha = nn.Parameter(torch.ones(1) * 0.2)  # 可学习的残差权重

    def forward(self, x):
        edge_feat = self.edge_conv(x)
        return x + self.alpha * edge_feat

# =======================================================================================================
class LightweightCBAM(nn.Module):
    """轻量级CBAM - 保持性能的同时减少参数"""

    def __init__(self, c1, ratio=8):
        super().__init__()
        # 通道注意力 - 减少中间层通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // ratio, c1, 1, bias=False)
        )

        # 空间注意力 - 使用更小的卷积核
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.conv1(torch.cat([avg_out, max_out], dim=1))
        spatial_att = self.sigmoid(spatial_att)

        return x * spatial_att


class SmallTargetBooster(nn.Module):
    """小目标增强器 - 专门处理模糊轮廓的极小无人机"""

    def __init__(self, c1):
        super().__init__()
        # 边缘检测分支 - 增强模糊轮廓
        self.edge_branch = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1, bias=False),  # 深度卷积
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(c1)
        )

        # 细节保护分支 - 保护清晰细节
        self.detail_branch = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU(),
            nn.Conv2d(c1 // 2, c1 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 2),
            nn.SiLU(),
            nn.Conv2d(c1 // 2, c1, 1, bias=False),
            nn.BatchNorm2d(c1)
        )

        # 自适应权重
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        edge_feat = self.edge_branch(x)
        detail_feat = self.detail_branch(x)

        # 生成自适应权重
        weights = self.weight_gen(x)  # [B, 2, 1, 1]
        w1, w2 = weights[:, 0:1], weights[:, 1:2]

        # 加权融合
        enhanced = w1 * edge_feat + w2 * detail_feat
        return x + enhanced * 0.3  # 残差连接，控制增强强度


class ContextAwareAttention(nn.Module):
    """上下文感知注意力 - 处理复杂背景"""

    def __init__(self, c1):
        super().__init__()
        # 局部上下文分支
        self.local_branch = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 1),
            nn.BatchNorm2d(c1 // 4),
            nn.SiLU(),
            nn.Conv2d(c1 // 4, c1 // 4, 3, padding=1),
            nn.BatchNorm2d(c1 // 4),
            nn.SiLU(),
            nn.Conv2d(c1 // 4, c1, 1),
            nn.Sigmoid()
        )

        # 全局上下文分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),  # 避免1x1尺寸
            nn.Conv2d(c1, c1 // 4, 1),
            nn.BatchNorm2d(c1 // 4),
            nn.SiLU(),
            nn.ConvTranspose2d(c1 // 4, c1, 4, stride=1),  # 反卷积上采样
            nn.Sigmoid()
        )

        # 平衡权重
        self.balance = nn.Parameter(torch.tensor([0.7, 0.3]))  # 偏向局部上下文

    def forward(self, x):
        h, w = x.shape[2:]

        # 局部上下文注意力
        local_att = self.local_branch(x)

        # 全局上下文注意力
        global_att = self.global_branch(x)
        global_att = F.interpolate(global_att, size=(h, w), mode='bilinear', align_corners=False)

        # 平衡融合
        weights = F.softmax(self.balance, dim=0)
        combined_att = weights[0] * local_att + weights[1] * global_att

        return x * combined_att

# =========================================================================================================
class FocalModulation(nn.Module):
    """焦点调制模块 - 专注于小目标特征增强"""

    def __init__(self, c1, focal_level=2, focal_factor=2):
        super().__init__()
        self.focal_level = focal_level
        self.focal_factor = focal_factor

        # 层次化焦点卷积
        self.focal_layers = nn.ModuleList()
        for i in range(focal_level):
            kernel_size = focal_factor * i + focal_factor + 1
            self.focal_layers.append(
                nn.Conv2d(c1, c1, kernel_size, padding=kernel_size // 2, groups=c1)
            )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, 1),
            nn.GELU(),
            nn.Conv2d(c1 // 4, c1, 1),
            nn.Sigmoid()
        )

        # 调制器
        self.modulator = nn.Conv2d(c1, c1, 1)

    def forward(self, x):
        # 多层次焦点特征提取
        focal_feats = []
        for layer in self.focal_layers:
            x = layer(x)
            focal_feats.append(x)

        # 特征聚合
        focal_feat = sum(focal_feats)

        # 门控调制
        gate = self.gate(focal_feat)
        modulated = self.modulator(focal_feat * gate)

        return modulated + x  # 残差连接


class CoordinateAttention(nn.Module):
    """坐标注意力 - 保持位置信息的轻量级注意力"""

    def __init__(self, c1, reduction=32):
        super().__init__()

        # 水平和垂直全局平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 共享的MLP
        temp = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, temp, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(temp)
        self.act = nn.ReLU()

        # 分别处理x和y方向
        self.conv_h = nn.Conv2d(temp, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(temp, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()

        # 坐标信息嵌入
        x_h = self.pool_h(x)  # [N, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [N, C, W, 1]

        # 连接并处理
        y = torch.cat([x_h, x_w], dim=2)  # [N, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离处理
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [N, C, 1, W]

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # [N, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [N, C, 1, W]

        return x * a_h * a_w


class SimpleFeatureEnhancer(nn.Module):
    """简化的特征增强器 - 基于test3成功经验"""

    def __init__(self, c1):
        super().__init__()
        # 轻量级特征提取
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1 // 4),  # 分组卷积
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 1),  # 逐点卷积
            nn.BatchNorm2d(c1)
        )

        # 自适应权重
        self.weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        enhanced = self.enhance_conv(x)
        return x + self.weight * enhanced


class OptimizedMultiScaleEnhancer(nn.Module):
    """基于test2成功经验的优化版本"""

    def __init__(self, c1):
        super().__init__()
        c_ = c1 // 2

        # 小目标特征增强 - 保持test2的设计理念
        self.small_enhancer = nn.Sequential(
            Conv(c1, c_, 1),  # 使用Conv类，参数顺序为 (c_in, c_out, kernel_size)
            Conv(c_, c_ // 2, 3, g=c_ // 2),  # 深度可分离卷积
            Conv(c_ // 2, c_, 1),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        # 边缘特征增强 - 针对UAV边缘模糊特性
        self.edge_enhancer = nn.Sequential(
            Conv(c1, c_, 1),
            nn.Conv2d(c_, c_, kernel_size=3, padding=1, groups=c_, bias=False),  # 修复：使用kernel_size
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            Conv(c_, c_, 1)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            Conv(c_ * 2, c1, 1),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

        # 残差权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self, x):
        small_feat = self.small_enhancer(x)
        edge_feat = self.edge_enhancer(x)

        fused = torch.cat([small_feat, edge_feat], dim=1)
        attention = self.fusion(fused)

        return x + self.residual_weight * (x * attention)


class LightweightAdaptiveAttention(nn.Module):
    """基于test2 AdaptiveAttention的轻量级版本"""

    def __init__(self, c1):
        super().__init__()
        # 简化的通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(c1 // 16, 4), kernel_size=1),  # 确保最小通道数
            nn.SiLU(),
            nn.Conv2d(max(c1 // 16, 4), c1, kernel_size=1),
            nn.Sigmoid()
        )

        # 简化的空间注意力
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),  # 修复：使用kernel_size
            nn.Sigmoid()
        )

        # 自适应权重
        self.balance = nn.Parameter(torch.tensor([0.6, 0.4]))  # 偏向通道注意力

    def forward(self, x):
        # 通道注意力
        ca_out = self.ca(x) * x

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_out = self.sa(sa_input) * x

        # 自适应融合
        w = F.softmax(self.balance, dim=0)
        return w[0] * ca_out + w[1] * sa_out

