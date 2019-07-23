# -*- coding: utf-8 -*-
"""
Created on Tuesday Jun 25 13:34:42 2018

@author: lux32
"""

from torch.autograd import Variable
import torch.nn as nn
import torch

class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        # padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (1, kSize), stride=stride, bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output

class P(nn.Module):
    def __init__(self, kSize, stride=2):
        super().__init__()
        # padding = int((kSize - 1) / 2)
        self.pool = nn.MaxPool2d((1, kSize), stride=stride)

    def forward(self, input):
        output = self.pool(input)
        return output


class shallow_net(nn.Module):
    def __init__(self):
        super().__init__()   # 20 * 1 * 13
        self.level1 = C(20, 512, 2)  # 1 * 12 * 512
        self.level1_0 = P(2)   # 1 * 6  * 512
        self.level2 = C(512, 512, 3)  # 1 * 4 * 512
        self.level3_0 = nn.Linear(1 * 4 * 512, 400)
        self.level3_d = nn.Dropout(p=0.5)
        self.level3_t = nn.Tanh()
        self.level3_1= nn.Linear(400,1)

    def forward(self, input):
        output = self.level1(input)
        output = self.level1_0(output)
        output = self.level2(output)
        output = output.view(-1, 1 * 4 * 512)
        output = self.level3_t(output)
        output = self.level3_d(output)
        output = self.level3_0(output)
        output = self.level3_t(output)
        output = self.level3_d(output)
        output = self.level3_1(output)
        return output


class LSTM_block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,  dropout=0.2, bidirectional=True)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).cuda()) #.cuda()
        c0 = Variable(torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).cuda()) #.cuda()
        output, (h_n, c_n) = self.lstm(input, (h0, c0))
        return output


class LSTM_net(nn.Module):
    def __init__(self):
        super().__init__()   # 20 * 1 * 13
        self.level1 = C(20, 512, 2)  # 512 * 1 * 12
        self.level1_0 = P(2)   # 512 * 1 * 6
        self.level2 = C(512, 512, 3)  # 1 * 4 * 512
        self.level3 = LSTM_block(4, 4, 2)
        self.level4_0 = nn.Linear(1 * 8 * 512, 400)
        self.level4_d = nn.Dropout(p=0.5)
        self.level4_t = nn.Tanh()
        self.level4_1= nn.Linear(400,1)

    def forward(self, input):
        output = self.level1(input)
        output = self.level1_0(output)
        output = self.level2(output)
        output = torch.squeeze(output, 2)
        output = self.level3(output)
        output = output.contiguous().view(output.size(0), -1)
        output = self.level4_t(output)
        output = self.level4_d(output)
        output = self.level4_0(output)
        output = self.level4_t(output)
        output = self.level4_d(output)
        output = self.level4_1(output)
        return output


class GRU_block(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,  dropout=0.2, bidirectional=True)

    def forward(self, input):
        h0 = Variable(torch.zeros(self.num_layers * 2, input.size(0), self.hidden_size).cuda()) #.cuda()
        output, h_n = self.gru(input, h0)
        return output

class GRU_net(nn.Module):
    def __init__(self):
        super().__init__()   # 20 * 1 * 13
        self.level1 = C(20, 512, 2)  # 512 * 1 * 12
        self.level1_0 = P(2)   # 512 * 1 * 6
        self.level2 = C(512, 512, 3)  # 1 * 4 * 512
        self.level3 = GRU_block(4, 4, 2)
        self.level4_0 = nn.Linear(1 * 8 * 512, 400)
        self.level4_d = nn.Dropout(p=0.5)
        self.level4_t = nn.Tanh()
        self.level4_1= nn.Linear(400,1)

    def forward(self, input):
        output = self.level1(input)
        output = self.level1_0(output)
        output = self.level2(output)
        output = torch.squeeze(output, 2)
        output = self.level3(output)
        output = output.contiguous().view(output.size(0), -1)
        output = self.level4_t(output)
        output = self.level4_d(output)
        output = self.level4_0(output)
        output = self.level4_t(output)
        output = self.level4_d(output)
        output = self.level4_1(output)
        return output


"""x = torch.Tensor(torch.randn(9, 20, 1, 13))
model = GRU_net()
y = model.forward(x)
print(y.size())
print(y)
"""


