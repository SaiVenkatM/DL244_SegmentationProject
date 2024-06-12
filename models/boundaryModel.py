import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import torch

class upsample_3d(nn.Module):
    def __init__(self):
        super(upsample_3d, self).__init__()
        self.upsample = partial(self._interpolate, mode='nearest')

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class DoubleConv_3d(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class unet3d_mtl_tsd(nn.Module):
    def __init__(self, in_ch=1, out_ch=9):
        super(unet3d_mtl_tsd, self).__init__()

        filters = [16, 32, 64, 128, 256]

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.Up = upsample_3d()

        self.conv0_0 = DoubleConv_3d(in_ch, filters[0])
        self.conv1_0 = DoubleConv_3d(filters[0], filters[1])
        self.conv2_0 = DoubleConv_3d(filters[1], filters[2])
        self.conv3_0 = DoubleConv_3d(filters[2], filters[3])
        self.conv4_0 = DoubleConv_3d(filters[3], filters[4])

        self.conv0_1 = DoubleConv_3d(filters[0] + filters[1], filters[0])
        self.conv1_1 = DoubleConv_3d(filters[1] + filters[2], filters[1])
        self.conv2_1 = DoubleConv_3d(filters[2] + filters[3], filters[2])
        self.conv3_1 = DoubleConv_3d(filters[3] + filters[4], filters[3])

        self.conv0_1e = DoubleConv_3d(filters[0] + filters[1], filters[0])
        self.conv1_1e = DoubleConv_3d(filters[1] + filters[2], filters[1])
        self.conv2_1e = DoubleConv_3d(filters[2] + filters[3], filters[2])
        self.conv3_1e = DoubleConv_3d(filters[3] + filters[4], filters[3])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)
        self.final_edge = nn.Conv3d(filters[0], 1, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        #### decoder for organ prediction ###
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x3_0, x4_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x2_0, x3_1)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x1_0, x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x0_0, x1_1)], 1))

        #### decoder for edge prediction ###
        x3_1e = self.conv3_1e(torch.cat([x3_0, self.Up(x3_0, x4_0)], 1))
        x2_1e = self.conv2_1e(torch.cat([x2_0, self.Up(x2_0, x3_1e)], 1))
        x1_1e = self.conv1_1e(torch.cat([x1_0, self.Up(x1_0, x2_1e)], 1))
        x0_1e = self.conv0_1e(torch.cat([x0_0, self.Up(x0_0, x1_1e)], 1))

        output0_4 = self.final(x0_1)
        output0_4_e = self.final_edge(x0_1e)
        return output0_4, output0_4_e


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.double_conv(inputs)
        return outputs


class gating_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(gating_block, self).__init__()

        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size, padding=0),
                                  nn.BatchNorm3d(out_ch),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, inputs):
        outputs = self.conv(inputs)
        return outputs


class io_concatenation(nn.Module):
    def __init__(self):
        super(io_concatenation, self).__init__()

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        return y


class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class attention_unet_3d(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(attention_unet_3d, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.conv1 = conv_block(in_ch, filters[0])
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = conv_block(filters[0], filters[1])
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = conv_block(filters[1], filters[2])
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = conv_block(filters[2], filters[3])
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = conv_block(filters[3], filters[4])

        self.up6 = nn.ConvTranspose3d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att6 = attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.conv6 = conv_block(filters[4], filters[3])

        self.up7 = nn.ConvTranspose3d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att7 = attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.conv7 = conv_block(filters[3], filters[2])

        self.up8 = nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att8 = attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.conv8 = conv_block(filters[2], filters[1])

        self.up9 = nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att9 = attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.conv9 = conv_block(filters[1], filters[0])

        self.final = nn.Conv3d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        att6 = self.att6(g=up_6, x=c4)
        up_6 = torch.cat((att6, up_6), dim=1)
        c6 = self.conv6(up_6)

        up_7 = self.up7(c6)
        att7 = self.att7(g=up_7, x=c3)
        up_7 = torch.cat((att7, up_7), dim=1)
        c7 = self.conv7(up_7)

        up_8 = self.up8(c7)
        att8 = self.att8(g=up_8, x=c2)
        up_8 = torch.cat((att8, up_8), dim=1)
        c8 = self.conv8(up_8)

        up_9 = self.up9(c8)
        att9 = self.att9(g=up_9, x=c1)
        up_9 = torch.cat((att9, up_9), dim=1)
        c9 = self.conv9(up_9)

        output = self.final(c9)

        return output
