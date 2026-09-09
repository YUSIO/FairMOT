"""PyTorch 2 compatible DCNv2 interface used by the FairMOT DLA decoder.

The upstream project imports its PyTorch-1.7 CUDA extension as ``dcn_v2``.
This wrapper preserves the module API with torchvision's compiled deformable
convolution operator so the reference network can run on the recorded remote
environment without silently replacing deformable convolution by plain Conv2d.
"""

import math

import torch
from torch import nn
from torchvision.ops import deform_conv2d


class DCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, deformable_groups=1, bias=True):
        super(DCN, self).__init__()
        if isinstance(kernel_size, tuple):
            kernel_h, kernel_w = kernel_size
        else:
            kernel_h = kernel_w = kernel_size
        if isinstance(stride, tuple):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride
        if isinstance(padding, tuple):
            padding_h, padding_w = padding
        else:
            padding_h = padding_w = padding
        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation_w = dilation

        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.deformable_groups = deformable_groups
        self.kernel_size = (kernel_h, kernel_w)
        kernel_elements = kernel_h * kernel_w
        self.conv_offset_mask = nn.Conv2d(
            in_channels, deformable_groups * 3 * kernel_elements,
            kernel_size=(kernel_h, kernel_w), stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=True)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_h, kernel_w))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.conv_offset_mask.weight, 0)
        nn.init.constant_(self.conv_offset_mask.bias, 0)

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        offset_channels = 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        offset = offset_mask[:, :offset_channels]
        mask = torch.sigmoid(offset_mask[:, offset_channels:])
        return deform_conv2d(
            x, offset, self.weight, self.bias, self.stride, self.padding,
            self.dilation, mask)
