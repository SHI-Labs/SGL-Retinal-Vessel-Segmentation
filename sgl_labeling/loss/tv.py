import torch
import torch.nn as nn
from torch.autograd import Variable

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.abs((x[:,:,1:,:]-x[:,:,:h_x-1,:])).sum()
        w_tv = torch.abs((x[:,:,:,1:]-x[:,:,:,:w_x-1])).sum()
        return self.TVLoss_weight*(h_tv/count_h+w_tv/count_w)/batch_size
