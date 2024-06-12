import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class UpSample3D(nn.Module):
    def __init__(self):
        super(UpSample3D, self).__init__()
        self.upsample = partial(self._interpolate, mode='nearest')

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)

class DualConv3D(nn.Module):
    """(conv => [BN] => LReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.dual_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.dual_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=9):
        super(UNet3D, self).__init__()
        filters = [16, 32, 64, 128, 256]
        
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.up = UpSample3D()

        self.conv0_0 = DualConv3D(in_ch, filters[0])
        self.conv1_0 = DualConv3D(filters[0], filters[1])
        self.conv2_0 = DualConv3D(filters[1], filters[2])
        self.conv3_0 = DualConv3D(filters[2], filters[3])
        self.conv4_0 = DualConv3D(filters[3], filters[4])

        self.conv0_1 = DualConv3D(filters[0] + filters[1], filters[0])
        self.conv1_1 = DualConv3D(filters[1] + filters[2], filters[1])
        self.conv2_1 = DualConv3D(filters[2] + filters[3], filters[2])
        self.conv3_1 = DualConv3D(filters[3] + filters[4], filters[3])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x3_0, x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x2_0, x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x1_0, x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x0_0, x1_1)], 1))
        
        output0_4 = self.final(x0_1)
        return output0_4

class UNetPlus3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=9):
        super(UNetPlus3D, self).__init__()
        filters = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.up = UpSample3D()

        self.conv0_0 = DualConv3D(in_ch, filters[0])
        self.conv1_0 = DualConv3D(filters[0], filters[1])
        self.conv2_0 = DualConv3D(filters[1], filters[2])
        self.conv3_0 = DualConv3D(filters[2], filters[3])
        self.conv4_0 = DualConv3D(filters[3], filters[4])

        self.conv0_1 = DualConv3D(filters[0] + filters[1], filters[0])
        self.conv1_1 = DualConv3D(filters[1] + filters[2], filters[1])
        self.conv2_1 = DualConv3D(filters[2] + filters[3], filters[2])
        self.conv3_1 = DualConv3D(filters[3] + filters[4], filters[3])

        self.conv0_2 = DualConv3D(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = DualConv3D(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = DualConv3D(filters[2]*2 + filters[3], filters[2])

        self.conv0_3 = DualConv3D(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = DualConv3D(filters[1]*3 + filters[2], filters[1])
        self.conv0_4 = DualConv3D(filters[0]*4 + filters[1], filters[0])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x0_0, x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x1_0, x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x0_1, x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x2_0, x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x1_1, x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x0_2, x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x3_0, x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x2_1, x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x1_2, x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x0_3, x1_3)], 1))
        
        output0_4 = self.final(x0_4)
        return output0_4

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.dual_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.dual_conv(inputs)
        return outputs

class GatingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(GatingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=0),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs

class IOConcatenation(nn.Module):
    def __init__(self):
        super(IOConcatenation, self).__init__()

    def forward(self, inputs, down_outputs, g):
        g = F.interpolate(g, size=down_outputs.shape[2:], mode='trilinear', align_corners=True)
        outputs = torch.cat([inputs, down_outputs, g], dim=1)
        return outputs

class ModifiedUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=9):
        super(ModifiedUNet3D, self).__init__()

        filters = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = ConvBlock(in_ch, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        self.gate1 = GatingBlock(filters[0], filters[0], kernel_size=1)
        self.gate2 = GatingBlock(filters[1], filters[1], kernel_size=1)
        self.gate3 = GatingBlock(filters[2], filters[2], kernel_size=1)
        self.gate4 = GatingBlock(filters[3], filters[3], kernel_size=1)

        self.io_concat1 = IOConcatenation()
        self.io_concat2 = IOConcatenation()
        self.io_concat3 = IOConcatenation()
        self.io_concat4 = IOConcatenation()

        self.conv0_1 = ConvBlock(filters[0]*2, filters[0])
        self.conv1_1 = ConvBlock(filters[1]*2, filters[1])
        self.conv2_1 = ConvBlock(filters[2]*2, filters[2])
        self.conv3_1 = ConvBlock(filters[3]*2, filters[3])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Gating signals
        g1 = self.gate1(x0_0)
        g2 = self.gate2(x1_0)
        g3 = self.gate3(x2_0)
        g4 = self.gate4(x3_0)
        
        # Decoder
        x3_1 = self.conv3_1(self.io_concat4(self.up(x4_0), x3_0, g4))
        x2_1 = self.conv2_1(self.io_concat3(self.up(x3_1), x2_0, g3))
        x1_1 = self.conv1_1(self.io_concat2(self.up(x2_1), x1_0, g2))
        x0_1 = self.conv0_1(self.io_concat1(self.up(x1_1), x0_0, g1))
        
        output = self.final(x0_1)
        return output
