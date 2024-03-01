"""
Code from:
https://github.com/NikolasEnt/Lyft-Perception-Challenge
"""
import torch
from torch import nn


def fb_loss(preds, trues, beta):
    smooth = 1e-4
    beta2 = beta*beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    TP = (preds * trues).sum(2)
    FP = (preds * (1-trues)).sum(2)
    FN = ((1-preds) * trues).sum(2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = Fb.sum() / (weights.sum() + smooth)
    return torch.clamp(score, 0., 1.)


class FBLoss(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        return 1 - fb_loss(output, target, self.beta)


class FbBceLoss(nn.Module):
    def __init__(self, fb_weight=0.5, fb_beta=1, bce_weight=0.5):
        super().__init__()
        self.fb_weight = fb_weight
        self.bce_weight = bce_weight

        self.fb_loss = FBLoss(beta=fb_beta)
        self.bce_loss = nn.BCELoss()

    def forward(self, output, target):
        # output has shape (B, C, H, W)
        # target has shape (B, C - 1, H, W)

        # extract last channel from output for each batch and concatenate
        thresh_binary = output[:, -1:, :, :]
        output = output[:, :-1, :, :]

        fb = self.fb_loss(output, target) * self.fb_weight
        bce = self.bce_loss(output, target) * self.bce_weight
        dice = dice_loss(thresh_binary, target[:, 0, :, :])
        return fb + bce + dice


def dice_loss(pred, m):
    eps = 1e-6
    intersection = torch.sum(pred * m)
    union = torch.sum(pred * m) + eps
    loss = 1 - 2.0 * intersection / union
    return loss
