import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import math

class GradientPenalty:
    def __init__(self, batch_size, gp_lambda, device):
        self.batch_size = batch_size
        self.gp_lambda = gp_lambda
        self.device = device

    def __call__(self, discriminator, real_data, fake_data, progress):
        alpha = torch.rand(self.batch_size, 1, 1, 1, requires_grad=True, device=self.device)
        interpolates = (1 - alpha) * real_data + alpha * fake_data
        d_interpolates = discriminator(interpolates, progress.alpha, progress.stage)

        gradients = grad(outputs=d_interpolates,
                         inputs=interpolates,
                         grad_outputs=torch.ones(d_interpolates.size(), device=self.device),
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]
        gradients = gradients.view(self.batch_size, -1)
        gradient_penalty = self.gp_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    
class Progress:
    def __init__(self, max_stage, max_epoch, max_step):
        self.alpha = 0
        self.stage = 0
        self.max_stage = max_stage
        self.max_epoch = max_epoch
        self.max_step = max_step

    def progress(self, current_stage, current_epoch, current_step):
        self.stage = current_stage
        p = (current_epoch * self.max_step + current_step) / (self.max_epoch * self.max_step)
        self.alpha = p if 0 < current_stage < self.max_stage else 1

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.scale = math.sqrt(2.0) / math.sqrt(in_channels)

    def forward(self, x):
        out = F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return x


class MiniBatch(nn.Module):
    def __init__(self):
        super(MiniBatch, self).__init__()
        self.offset = 1e-8

    def forward(self, x):
        stddev = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=True))**2, dim=0, keepdim=True) + self.offset)
        inject_shape = list(x.size())[:]
        inject_shape[1] = 1
        inject = torch.mean(stddev, dim=1, keepdim=True)
        inject = inject.expand(inject_shape)
        return torch.cat((x, inject), dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act, norm):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if act == "lrelu":
            self.act = nn.LeakyReLU(0.2)
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = None

        if norm:
            self.norm = PixelNorm()
        else:
            self.norm = None

    def forward(self, x):
        out = self.conv(x)
        if self.act is not None:
            out = self.act(out)
        if self.norm is not None:
            out = self.norm(out)

        return out


class Generator(nn.Module):
    def __init__(self, max_stage=8, base_channels=16, image_channels=3):
        super(Generator, self).__init__()
        self.max_stage = max_stage

        self.toRGBs = nn.ModuleList()
        for i in reversed(range(self.max_stage + 1)):
            in_channels = min(base_channels * 2 ** i, 512)
            self.toRGBs.append(ConvBlock(in_channels, image_channels, 1, 1, 0, "tanh", False))

        self.blocks = nn.ModuleList()
        self.blocks.append(self.first_conv_block(base_channels * 2 ** self.max_stage, base_channels * 2 ** self.max_stage))
        for i in reversed(range(self.max_stage)):
            in_channels = min(base_channels * 2 ** (i + 1), 512)
            out_channels = min(base_channels * 2 ** i, 512)
            self.blocks.append(self.conv_block(in_channels, out_channels, 3, 1, 1))

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, inputs, alpha, stage):
        stage = min(stage, self.max_stage)

        x = inputs
        for i in range(0, stage):
            x = self.blocks[i](x)
            x = self.upsample(x)

        identity = x
        x = self.blocks[stage](x)
        x = self.toRGBs[stage](x)

        if alpha % 1 != 0:
            identity = self.toRGBs[stage - 1](identity)
            x = alpha * x + (1 - alpha) * identity

        return x

    def first_conv_block(self, in_channels, out_channels):
        layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, 4, 1, 3, "lrelu", True),
            ConvBlock(out_channels, out_channels, 3, 1, 1, "lrelu", True))
        return layers

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding, "lrelu", True),
            ConvBlock(out_channels, out_channels, kernel_size, stride, padding, "lrelu", True))
        return layers


class Discriminator(nn.Module):
    def __init__(self, max_stage=8, base_channels=16, image_channels=3):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage

        self.fromRGBs = nn.ModuleList()
        for i in reversed(range(self.max_stage + 1)):
            out_channels = min(base_channels * 2 ** i, 512)
            self.fromRGBs.append(ConvBlock(image_channels, out_channels, 1, 1, 0, "lrelu", False))

        self.blocks = nn.ModuleList()
        self.blocks.append(self.first_conv_block(base_channels * 2 ** self.max_stage, 1))
        for i in reversed(range(self.max_stage)):
            in_channels = min(base_channels * 2 ** i, 512)
            out_channels = min(base_channels * 2 ** (i + 1), 512)
            self.blocks.append(self.conv_block(in_channels, out_channels, 3, 1, 1))

        self.downsample = nn.AvgPool2d(2, 2)
        self.minibatch = MiniBatch()

    def forward(self, inputs, alpha, stage):
        stage = min(stage, self.max_stage)

        x = self.fromRGBs[stage](inputs)

        for i in range(stage, 0, -1):
            x = self.blocks[i](x)
            x = self.downsample(x)
            if i == stage and alpha % 1 != 0:
                identity = self.downsample(inputs)
                identity = self.fromRGBs[stage - 1](identity)
                x = alpha * x + (1 - alpha) * identity

        x = self.minibatch(x)
        x = self.blocks[0](x)

        return x.squeeze()

    def first_conv_block(self, in_channels, out_channels):
        layers = nn.Sequential(
            ConvBlock(in_channels + 1, in_channels, 3, 1, 1, "lrelu", False),
            ConvBlock(in_channels, in_channels, 4, 1, 0, "lrelu", False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        return layers

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layers = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size, stride, padding, "lrelu", False),
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding, "lrelu", False))
        return layers