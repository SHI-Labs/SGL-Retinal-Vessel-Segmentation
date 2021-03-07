import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from model import common
from model.benhf import EnhanceNet

def make_model(args, parent=False):
    return MainNet()

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
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
        self.conv11 = self.conv_block(35,1)
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
        u1 = torch.cat([u1, x], 1)
        out_pred = F.sigmoid(self.conv11(u1))
        return out_pred

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.s1 = EnhanceNet()
        self.s2 = SegNet()
    def forward(self, x):
        x1 = self.s1(x)
        out = self.s2(x1)
        return x1, out
