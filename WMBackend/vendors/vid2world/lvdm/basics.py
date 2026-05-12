# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!

import torch.nn as nn
# from utils.utils import instantiate_from_config
import torch.nn.functional as F
import torch

class CausalConv1d(torch.nn.Module):
    '''
    Implementation of a 1D causal convolution.
    This implementation takes a dummy padding parameter for the sake of compatibility with conv_nd.
    However, this dummy padding is not used in the causal convolution.
    This should be alright as the Unet's padding are all aligned with the kernel size.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding= None):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.conv1d(x)
        if self.padding > 0:    
            x = x[:, :, : -self.padding]
        return x

class CausalConv2d(torch.nn.Module):
    '''
    Implementation of a 2D causal convolution.
    The causal conv is applied to the -2 axis, while the -1 axis is non-causal.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding= None):
        super(CausalConv2d, self).__init__()
        self.padding1 = (kernel_size - 1) * dilation
        self.padding2 = (kernel_size - 1) * dilation // 2
        self.conv2d = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self.padding1, self.padding2),
            dilation=dilation,
        )

    def forward(self, x):
        x = self.conv2d(x)
        if self.padding1 > 0:
            x = x[:, :, : -self.padding1, :]
        return x
    
class CausalConv3d(torch.nn.Module):
    '''
    Implementation of a 3D causal convolution.
    The causal conv is applied to the -3 axis, while the -2, -1 axis is non-causal.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding= None):
        super(CausalConv3d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size) == 3, "kernel_size must be a tuple of length 3"
        self.padding1 = (kernel_size[0] - 1) * dilation
        self.padding2 = (kernel_size[1] - 1) * dilation // 2
        self.padding3 = (kernel_size[2] - 1) * dilation // 2
        self.conv3d = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self.padding1, self.padding2, self.padding3),
            dilation=dilation,
        )

    @property
    def weight(self):
        return self.conv3d.weight

    @property
    def bias(self):
        return self.conv3d.bias

    def forward(self, x):
        x = self.conv3d(x)
        if self.padding1 > 0:
            x = x[:, :, : -self.padding1, :, :]
        return x


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def conv_nd(dims, *args, causal=False, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if causal:
        if dims == 1:
            return CausalConv1d(*args, **kwargs)
        elif dims == 2:
            return CausalConv2d(*args, **kwargs)
        elif dims == 3:
            return CausalConv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")
    else:
        if dims == 1:
            return nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            return nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            return nn.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def nonlinearity(type='silu'):
    if type == 'silu':
        return nn.SiLU()
    elif type == 'leaky_relu':
        return nn.LeakyReLU()


class GroupNormSpecific(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels, num_groups=32):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNormSpecific(num_groups, channels)


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}