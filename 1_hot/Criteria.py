# -*- coding: utf-8 -*-
"""
Created on Tuesday Jun 25 13:34:42 2018

@author: lux32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)