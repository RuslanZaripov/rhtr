import torch
from torch import nn
import torch.nn.functional as F


def wbce(output, target, pos_weight=None, epsilon=10 ** -15):
    output = output.clamp(epsilon, 1 - epsilon)

    if pos_weight is not None:
        assert pos_weight.numel() == 1, "Expected only one weight value."

        loss = pos_weight * (target * torch.log(output)) + \
               (1 - pos_weight) * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    result = torch.neg(torch.mean(loss))

    return result


def fb_loss(preds, trues, beta):
    smooth = 1e-4
    beta2 = beta * beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    TP = (preds * trues).sum(2)
    FP = (preds * (1 - trues)).sum(2)
    FN = ((1 - preds) * trues).sum(2)
    Fb = ((1 + beta2) * TP + smooth) / ((1 + beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = Fb.sum() / (weights.sum() + smooth)
    return torch.clamp(score, 0., 1.)


class FBLoss(torch.nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, output, target):
        return 1 - fb_loss(output, target, self.beta)


def dice_loss(inputs, targets, smooth=1):
    # flatten label and prediction tensors
    inputs = inputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        inputs = torch.cat((output['binary'], output['lines']), 1)
        targets = torch.cat((target['binary'], target['lines']), 1)

        dice = dice_loss(inputs, targets, smooth=1)

        return 1 - dice


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.focal_loss = FocalLoss()

    def forward(self, output, target, stage=None, tb_writer=None, tb_x=None):
        total_class_loss = 0.0
        for class_idx in range(output['watershed'].shape[1]):
            watershed_class_output = output['watershed'][:, class_idx]
            watershed_class_target = target['watershed'][:, class_idx]

            dice_loss_class = dice_loss(watershed_class_output, watershed_class_target)
            focal_loss_class = self.focal_loss(watershed_class_output, watershed_class_target)

            class_loss = (dice_loss_class + focal_loss_class) / 2

            total_class_loss += class_loss

        total_class_loss /= output['watershed'].shape[1]

        dice_loss_binary = dice_loss(output['binary'], target['binary'])
        focal_loss_binary = self.focal_loss(output['binary'], target['binary'])

        binary_loss = (dice_loss_binary + focal_loss_binary) / 2

        dice_loss_lines = dice_loss(output['lines'], target['lines'])
        focal_loss_lines = self.focal_loss(output['lines'], target['lines'])

        lines_loss = (dice_loss_lines + focal_loss_lines) / 2

        dice_loss_border = dice_loss(output['border_mask'], target['border_mask'])
        focal_loss_border = self.focal_loss(output['border_mask'], target['border_mask'])

        border_loss = (dice_loss_border + focal_loss_border) / 2

        total_loss = (binary_loss + 2 * lines_loss + border_loss + total_class_loss) / 5

        return total_loss
