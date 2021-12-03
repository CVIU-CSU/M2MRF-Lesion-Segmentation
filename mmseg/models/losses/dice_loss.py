# REF: https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES


def _make_one_hot(label, num_classes):
    """
    :param label: [N, *], values in [0,num_classes)
    :return: [N, C, *]
    """
    label = label.unsqueeze(1)
    shape = list(label.shape)
    shape[1] = num_classes

    result = torch.zeros(shape, device=label.device)
    result.scatter_(1, label, 1)

    return result


def binary_dice_loss(pred, label, smooth=1e-5):
    """
    :param pred: [N, *]: here should be scores in [0,1]
    :param label: [N, *]
    :param power: 1 for abs, 2 for square
    :return: [N]
    """

    pred = pred.contiguous().view(pred.shape[0], -1).float()
    label = label.contiguous().view(label.shape[0], -1).float()

    num = 2 * torch.sum(torch.mul(pred, label), dim=1) + smooth
    den = torch.sum(pred, dim=1) + torch.sum(label, dim=1) + smooth

    loss = 1 - num / den
    return loss


def dice_loss(pred_raw,
              label_raw,
              weight=None,
              class_weight=None,
              reduction='mean',
              avg_factor=None,
              ignore_class=-1,
              smooth=1e-5):
    """
    :param pred:  [N, C, *]scores without softmax
    :param label: [N, *]
    :return: reduction([N])
    """
    pred = pred_raw.clone()
    label = label_raw.clone()
    num_classes = pred.shape[1]
    if class_weight is not None:
        class_weight = class_weight.float()

    if pred.shape != label.shape:
        label = _make_one_hot(label, num_classes)

    pred = F.softmax(pred, dim=1)

    loss = 0.
    for i in range(num_classes):
        if i != ignore_class:
            dice_loss = binary_dice_loss(pred[:, i], label[:, i], smooth)

            if class_weight is not None:
                dice_loss *= class_weight[i]
            loss += dice_loss

    if ignore_class != -1:
        num_classes -= 1

    loss = loss / num_classes
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_class=-1,
                 smooth=1e-5
                 ):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.cls_criterion = dice_loss
        self.ignore_class = ignore_class
        self.smooth = smooth

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
            assert class_weight.shape[0] == label.shape[1], \
                'Expect weight shape [{}], get[{}]'.format(label.shape[1], class_weight.shape[0])
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_class=self.ignore_class,
            smooth=self.smooth
        )
        return loss_cls
