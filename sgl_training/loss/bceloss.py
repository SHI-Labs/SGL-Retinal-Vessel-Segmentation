import torch
import torch.nn as nn
from torch.autograd import Variable as V

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):

        y_ce_true = y_true.squeeze(dim=1).long()


        a = self.ce_loss(y_pred, y_ce_true)

        return a


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b



def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))


class pdice_loss(nn.Module):
    def __init__(self, batch=True):
        super(pdice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred, p):
        smooth = 0.0  # may change
        if self.batch:
            pmap = p.clone()
            pmap[pmap>=0.8] = 1
            pmap[pmap<0.8] = 0
            y_true_th = y_true * pmap
            y_pred_th = y_pred * pmap
            i = torch.sum(y_true_th)
            j = torch.sum(y_pred_th)
            intersection = torch.sum(y_true_th * y_pred_th)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred, pmap):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred, pmap)
        return loss

    def forward(self, y_pred, y_true, pmap):
        b = self.soft_dice_loss(y_true, y_pred, pmap)
        return b

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def forward(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        #return a
        return b

class base_bce_loss(nn.Module):
    def __init__(self):
        super(base_bce_loss, self).__init__()
    def forward(self, y_pred, y_true, dp):
        B, C, W, H = y_pred.size()
        bce = - y_true*torch.log(y_pred+1e-14) - (1 - y_true) * torch.log(1 - y_pred + 1e-14)
        bce = torch.sum(bce) / (B*C*W*H)
        return bce

class p_bce_loss(nn.Module):
    def __init__(self):
        super(p_bce_loss, self).__init__()
    def forward(self, y_pred, y_true, dp):
        B, C, W, H = y_pred.size()
        bce = - y_true*torch.log(y_pred+1e-14) - 0.1 * (1 - y_true) * torch.log(1 - y_pred + 1e-14)
        bce = torch.sum(bce) / (B*C*W*H)
        return bce

class df_bce_loss(nn.Module):
    def __init__(self):
        super(p_bce_loss, self).__init__()
    def forward(self, y_pred, y_true, dp):
        B, C, W, H = y_pred.size()
        bce = - y_true*torch.log(y_pred+1e-14) - 0.1 * (1 - y_true) * torch.log(1 - y_pred + 1e-14)
        bce = torch.sum(bce) / (B*C*W*H)
        return bce

class sym_bce_loss(nn.Module):
    def __init__(self):
        super(sym_bce_loss, self).__init__()
        self.bceloss = base_bce_loss()

    def forward(self, y_pred, y_true, dp):
        B, C, W, H = y_pred.size()
        bce = self.bceloss(y_pred, y_true, dp)
        rbce = - y_pred*torch.log(y_true+1e-14) - (1 - y_pred) * torch.log(1 - y_true + 1e-14)
        rbce = torch.sum(rbce) / (B*C*W*H)
        out = bce + rbce
        return out

class sg_bce_loss(nn.Module):
    def __init__(self):
        super(sg_bce_loss, self).__init__()
        self.bceloss = base_bce_loss()

    def forward(self, y_pred, y_true, dp):
        B, C, W, H = y_pred.size()
        #bce = self.bceloss(y_pred, y_true, dp)
        #factor = 1 / (1 + torch.exp(-5*((1 - y_pred)-0.5)))
        rbce = - y_true*torch.log(y_pred+1e-14) - (1 - y_true) * factor * torch.log(1 - y_pred + 1e-14)
        rbce = torch.sum(rbce) / (B*C*W*H)
        out = rbce
        return out

class penalty_bce_loss(nn.Module):
    def __init__(self):
        super(penalty_bce_loss, self).__init__()
    def forward(self, y_pred, y_true, pmap):
        B, C, W, H = y_pred.size()
        bce = - y_true*torch.log(y_pred+1e-14) - (1 - y_true) * torch.log(1 - y_pred + 1e-14)
        bce = bce * pmap
        bce = torch.sum(bce) / (B*C*W*H)
        return bce

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i,:, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduction=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction:
            return torch.mean(F_loss)
        else:
            return F_loss

