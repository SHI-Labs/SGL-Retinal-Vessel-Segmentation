import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from model import common


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        hg = hg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1
        wg = wg.type(torch.cuda.FloatTensor).repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1
        guidemap = guidemap.permute(0,2,3,1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)

        coeff = F.grid_sample(bilateral_grid, guidemap_guide)

        return coeff.squeeze(2)

class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)

class GridNet(nn.Module):
    def __init__(self, inc=3):
        super(GridNet, self).__init__()
        self.conv1 = self.conv_block(3,32)
        self.conv2 = self.conv_block(32,64)
        self.conv3 = self.conv_block(64,128)
        self.conv4 = self.conv_block(128,128*2)
        self.conv5 = self.conv_block(128*2,128*4)
        self.pool = torch.nn.MaxPool2d(2)
        self.upconv1 = self.upconv(64,32)
        self.upconv2 = self.upconv(128,64)
        self.upconv3 = self.upconv(128*2,128)
        self.upconv4 = self.upconv(128*4,128*2)
        self.conv6 = self.conv_block(128*4,128*2)
        self.conv7 = self.conv_block(128*2,128)
        self.conv8 = self.conv_block(128,64)
        self.conv9 = self.conv_block(64,32)
        self.conv11 = self.conv_block(35,12)
        self.last_act = nn.PReLU()

    def conv_block(self, channel_in, channel_out):
        if channel_in==3:
            return nn.Sequential(
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )
        else:
            return nn.Sequential(
                nn.PReLU(),
                nn.BatchNorm2d(channel_in),
                nn.Conv2d(channel_in, channel_out, 3, 1, 1),
                nn.PReLU(),
                nn.BatchNorm2d(channel_out),
                nn.Conv2d(channel_out, channel_out, 3, 1, 1),
            )

    def upconv(self, channel_in, channel_out):
        return nn.ConvTranspose2d(channel_in,channel_out,kernel_size=2,stride=2)

    def forward(self, x):
        x = x / 255.
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)
        x4 = self.conv4(x4)
        x5 = self.pool(x4)
        x5 = self.conv5(x5)
        u4 = self.upconv4(x5)
        u4 = torch.cat([u4, x4], 1)
        u4 = self.conv6(u4)
        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, x3], 1)
        u3 = self.conv7(u3)
        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, x2], 1)
        u2 = self.conv8(u2)
        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, x1], 1)
        u1 = self.conv9(u1)
        #u1 = self.last_act(u1)
        u1 = torch.cat([u1, x], 1)
        #pred = self.conv10(u1) + x
        #out_pred = F.sigmoid(self.conv11(u1))
        out_grid = self.conv11(u1)
        return out_grid

class conv_block(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU(inplace=True), is_BN=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size, padding=padding, stride=stride, bias=use_bias),
                nn.BatchNorm2d(outc),
                activation
            )

    def forward(self, input):
        return self.conv(input)

class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        self.grid = GridNet()
        self.trans = Transform()
    def forward(self, x):
        coeff = self.grid(x)
        colored = self.trans(coeff, x)

        return colored
