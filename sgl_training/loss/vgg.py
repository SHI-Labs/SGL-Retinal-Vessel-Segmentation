from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features #all the modules of vgg
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0: #lower level features
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('11') >= 0: #higher level features
            self.vgg = nn.Sequential(*modules[:4])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        #sr = torch.cat([sr,sr,sr], 1)
        #hr = torch.cat([hr,hr,hr], 1)    
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
