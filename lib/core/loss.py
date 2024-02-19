import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from itertools import combinations, product


class ParamLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ParamLoss, self).__init__()        
        if type == 'l1': self.criterion = nn.L1Loss(reduction='sum')
        elif type == 'l2': self.criterion = nn.MSELoss()

    def forward(self, param_out, param_gt, valid=None):        
        return self.criterion(param_out, param_gt)


class CLSLoss(nn.Module):
    def __init__(self):
        super(CLSLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pred, target):
        loss = self.ce_loss(pred, target.float())
        return loss


def get_loss():
    loss = {}
    loss['hand_sdf'] = ParamLoss(type='l1')
    loss['obj_sdf'] = ParamLoss(type='l1')
    loss['volume_joint'] = ParamLoss(type='l2')
    loss['obj_corners'] = ParamLoss(type='l2')
    loss['hand_cls'] = CLSLoss()
    return loss