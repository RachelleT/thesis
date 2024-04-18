import torch
import torch.nn as nn
from torchvision.utils import save_image

# Generator Block
class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(GeneratorBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, base_channels=64, max_resolution=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.max_resolution = max_resolution

        self.init_resolution = 4
        self.init_channels = base_channels * 8

        self.init_block = GeneratorBlock(latent_dim, self.init_channels, kernel_size=4, stride=1, padding=0)

        self.blocks = nn.ModuleList()
        in_channels = self.init_channels
        for resolution in range(self.init_resolution * 2, max_resolution + 1, 2):
            out_channels = min(base_channels * (self.max_resolution // resolution), 512)
            self.blocks.append(GeneratorBlock(in_channels, out_channels))
            in_channels = out_channels

        self.final_block = nn.ConvTranspose2d(in_channels, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x, resolution):
        assert resolution <= self.max_resolution and resolution >= self.init_resolution and resolution % 2 == 0, "Invalid resolution"
        resolution_index = int(torch.log2(torch.tensor(resolution // self.init_resolution)))

        x = self.init_block(x)
        for i in range(resolution_index):
            x = self.blocks[i](x)
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.final_block(x)
        return torch.tanh(x)

# Discriminator Block
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, base_channels=64, max_resolution=256):
        super(Discriminator, self).__init__()
        self.base_channels = base_channels
        self.max_resolution = max_resolution

        self.init_resolution = 4
        self.init_channels = base_channels

        self.init_block = DiscriminatorBlock(1, self.init_channels, kernel_size=4, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        in_channels = self.init_channels
        for resolution in range(self.init_resolution * 2, max_resolution + 1, 2):
            out_channels = min(base_channels * (self.max_resolution // resolution), 512)
            self.blocks.append(DiscriminatorBlock(in_channels, out_channels))
            in_channels = out_channels

        self.final_block = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x, resolution):
        assert resolution <= self.max_resolution and resolution >= self.init_resolution and resolution % 2 == 0, "Invalid resolution"
        resolution_index = int(torch.log2(torch.tensor(resolution // self.init_resolution)))

        x = self.init_block(x)
        for i in range(resolution_index):
            x = self.blocks[i](x)
            x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.final_block(x)
        return torch.sigmoid(x)