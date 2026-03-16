import torch.nn as nn

"""
Some Utility blocks for ViT-VQGAN.

ConvNeXt blocks are based on:
Liu, Zhuang, et al. "A convnet for the 2020s."
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
"""


class ConvNextDownsampleBig(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.group_norm = nn.GroupNorm(c_in, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=8, stride=4, padding=0)

    def forward(self, x):
        return self.conv1(self.group_norm(x))


class ConvNextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=7 // 2, groups=channels)  # 'Depthwise' conv
        self.group_norm = nn.GroupNorm(channels, channels)  # Should be equivalent to layernorm

        # Transformer-style non-linearity
        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=1, stride=1, padding=0)
        self.activation = nn.GELU()
        self.conv3 = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.group_norm(y)
        y = self.conv2(y)
        y = self.activation(y)
        y = self.conv3(y)
        return x + y


class ConvNextDownsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.group_norm = nn.GroupNorm(c_in, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv1(self.group_norm(x))
