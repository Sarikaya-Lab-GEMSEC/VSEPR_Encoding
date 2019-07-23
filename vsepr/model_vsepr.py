import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CBR(nn.Module):  #same size
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)
        self.drop_out = nn.Dropout(p=0.4)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        output = self.drop_out(output)
        return output

class DownSampler_1d(nn.Module):
    def __init__(self, nIn, nOut, kSize=3):
        super().__init__()
        self.conv = Con_1d(nIn, nOut, kSize)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class Con_1d(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=(2,1)):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=(0, padding), bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output

class CR_1d(nn.Module):
    def __init__(self, nIn, nOut, stride=2, kSize=3):
        super().__init__()
        self.conv = Con_1d(nIn, nOut, kSize, stride=stride)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.act(output)
        return output

class max_pool_1d(nn.Module):
    def __init__(self, kSize=(2, 1)):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kSize, stride=(2, 1))

    def forward(self, input):
        output = self.max_pool(input)
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

# class Con_stack(nn.Module):
# #     def __init__(self, nIn, nOut):
# #         super().__init__()

class vanilla(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.level0_c1 = CR_1d(5, 64)
        self.level0_c2 = CR_1d(64, 128)
        self.level0_c3 = CR_1d(128, 128)
        self.level0_c4 = CR_1d(128, 128)
        self.level0_pool = max_pool_1d()
        self.level0_dp = nn.Dropout(p=0.25)

        self.level1_c1 = CR_1d(128, 256)
        self.level1_c2 = CR_1d(256, 256)
        self.level1_c3 = CR_1d(256, 256)
        self.level1_c4 = CR_1d(256, 512)
        self.level1_pool = max_pool_1d()
        self.level1_dp = nn.Dropout(p=0.25)

        self.level2_c1 = CR_1d(512, 512)
        self.level2_c2 = CR_1d(512, 512)
        self.level2_c3 = CR_1d(512, 512)
        self.level2_c4 = CR_1d(512, 256)
        self.level2_pool = max_pool_1d()
        self.level2_dp = nn.Dropout(p=0.25)

        self.level3_c1 = CR_1d(256, 128)
        self.level3_c2 = CR_1d(128, 128)
        self.level3_c3 = CR_1d(128, 64)
        self.level3_c4 = CR_1d(64, 16)
        self.level3_pool = max_pool_1d()
        self.level3_dp = nn.Dropout(p=0.25)

        self.level4_gru = GRU_block(18, 8, 4)

        self.level5_0 = nn.Linear(768, 512)
        self.level5_1= nn.Linear(512, 256)
        self.level5_2= nn.Linear(256, 1)
        self.level5_dp = nn.Dropout(p=0.5)
        self.level5_act = nn.ReLU(True)

    def forward(self, input):
        output = self.level0_c1(input)
        output = self.level0_c2(output)
        output = self.level0_c3(output)
        output = self.level0_c4(output)
        output = self.level0_pool(output)
        output = self.level0_dp(output)

        output = self.level1_c1(output)
        output = self.level1_c2(output)
        output = self.level1_c3(output)
        output = self.level1_c4(output)
        output = self.level1_pool(output)
        output = self.level1_dp(output)

        output = self.level2_c1(output)
        output = self.level2_c2(output)
        output = self.level2_c3(output)
        output = self.level2_c4(output)
        output = self.level2_pool(output)
        output = self.level2_dp(output)

        output = self.level3_c1(output)
        output = self.level3_c2(output)
        output = self.level3_c3(output)
        output = self.level3_c4(output)
        output = self.level3_pool(output)
        output = self.level3_dp(output) # 1 * 16 * 18 * 18

        output = output.view(output.size()[0], 16 * 3, 18)
        output = self.level4_gru(output)
        output = output.contiguous().view(output.size(0), -1)

        output = self.level5_0(output)
        output = self.level5_act(output)
        output = self.level5_dp(output)

        output = self.level5_1(output)
        output = self.level5_act(output)
        output = self.level5_dp(output)

        output = self.level5_2(output)
        output = self.level5_act(output)
        output = self.level5_dp(output)

        return output


class CDilated(nn.Module): #same size
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class BR(nn.Module): #same size
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output

class CB(nn.Module):  #same size
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class DilatedParllelResidualBlockB1(nn.Module):  # with k=4
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        k = 4
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        self.c1 = Con_1d(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        output = self.act(output)
        return output


class ResNetC1(nn.Module):
    def __init__(self,):
        super().__init__()
        self.level1 = CBR(5, 16, 3, 1)
        self.level1_0 = CBR(5, 64, 3, 1)

        self.level2 = DownSampler_1d(16, 64)
        self.level2_0 = DilatedParllelResidualBlockB1(64, 64)
        self.level2_1 = DilatedParllelResidualBlockB1(64, 64)

        self.br_2 = BR(128)

        self.level3_0 = DownSampler_1d(128, 64)
        self.level3_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        self.level3_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)

        self.level4_0 = DownSampler_1d(192, 128)
        self.level4_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)

        self.br_4 = BR(192)
        self.br_con_4 = BR(256)

        self.level5_0 = DownSampler_1d(256, 64)
        self.level5_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        self.level5_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)

        self.br_con_5 = BR(128)

        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input1): #(1, 5, 174, 18)
        # input1 = self.cmlrn(input)
        output0 = self.level1(input1) #(1, 5, 174, 18)
        output1_0 = self.level2(output0) #(1, 64, 86, 18)
        output1 = self.level2_0(output1_0) #(1, 64, 86, 18)
        output1 = self.level2_1(output1) #(1, 64, 86, 18)

        output1 = self.br_2(torch.cat([output1_0, output1], 1)) #(1, 128, 86, 18)

        output2_0 = self.level3_0(output1) #(1, 64, 42, 18)
        output2 = self.level3_1(output2_0) #(1, 64, 42, 18)
        output2 = self.level3_2(output2) #(1, 64, 42, 18)

        output2 = self.br_2(torch.cat([output2_0, output2], 1)) #(1, 128, 42, 18)

        output3 = self.level4_1(output2) #(1, 128, 42, 18)
        output3 = self.level4_2(output3) #(1, 128, 42, 18)

        output3 = self.br_4(torch.cat([output2_0, output3], 1)) #(1, 192, 42, 18)

        l5_0 = self.level4_0(output3)
        l5_1 = self.level4_1(l5_0)
        l5_2 = self.level4_2(l5_1)
        l5_con = self.br_con_4(torch.cat([l5_0, l5_2], 1)) #(1, 256, 20, 18)

        l6_0 = self.level5_0(l5_con)
        l6_1 = self.level5_1(l6_0)
        l6_2 = self.level5_2(l6_1)
        l6_con = self.br_con_5(torch.cat([l6_0, l6_2], 1)) #(1, 256, 9, 18)

        glbAvg = self.global_Avg(l6_con)
        flatten = glbAvg.view(glbAvg.size(0), -1)
        fc1 = self.fc1(flatten)
        output = self.fc2(fc1)

        return output



class ResNetC1_0(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.level1 = CBR(5, 16, 3, 1)
        self.level1_0 = CBR(5, 64, 3, 1)

        self.level2 = DownSampler_1d(16, 64)
        self.level2_0 = DilatedParllelResidualBlockB1(64, 64)
        self.level2_1 = DilatedParllelResidualBlockB1(64, 64)

        self.br_2 = BR(128)

        self.level3_0 = DownSampler_1d(128, 64)
        self.level3_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        self.level3_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)

        self.level4_0 = DownSampler_1d(192, 128)
        self.level4_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)

        self.br_4 = BR(192)
        self.br_con_4 = BR(256)

        self.level5_0 = DownSampler_1d(256, 64)
        self.level5_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        self.level5_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)

        self.br_con_5 = BR(128)

        self.global_Avg = nn.AdaptiveAvgPool2d((1, 18))

        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 1)
        self.fc_dp = nn.Dropout(p=0.1)
        self.fc_act = nn.ReLU(True)

    def forward(self, input1):  # (1, 5, 174, 18)
        # input1 = self.cmlrn(input)
        output0 = self.level1(input1)  # (1, 5, 174, 18)
        output1_0 = self.level2(output0)  # (1, 64, 86, 18)
        output1 = self.level2_0(output1_0)  # (1, 64, 86, 18)
        output1 = self.level2_1(output1)  # (1, 64, 86, 18)

        output1 = self.br_2(torch.cat([output1_0, output1], 1))  # (1, 128, 86, 18)

        output2_0 = self.level3_0(output1)  # (1, 64, 42, 18)
        output2 = self.level3_1(output2_0)  # (1, 64, 42, 18)
        output2 = self.level3_2(output2)  # (1, 64, 42, 18)

        output2 = self.br_2(torch.cat([output2_0, output2], 1))  # (1, 128, 42, 18)

        output3 = self.level4_1(output2)  # (1, 128, 42, 18)
        output3 = self.level4_2(output3)  # (1, 128, 42, 18)

        output3 = self.br_4(torch.cat([output2_0, output3], 1))  # (1, 192, 42, 18)

        l5_0 = self.level4_0(output3)
        l5_1 = self.level4_1(l5_0)
        l5_2 = self.level4_2(l5_1)
        l5_con = self.br_con_4(torch.cat([l5_0, l5_2], 1))  # (1, 256, 20, 18)

        l6_0 = self.level5_0(l5_con)
        l6_1 = self.level5_1(l6_0)
        l6_2 = self.level5_2(l6_1)
        #l6_con = self.br_con_5(torch.cat([l6_0, l6_2], 1))  # (1, 128, 9, 18)

        glbAvg = self.global_Avg(l6_2)
        flatten = glbAvg.view(glbAvg.size(0), -1)
        fc1 = self.fc1(flatten)
        fc1 = self.fc_dp(self.fc_act(fc1))
        fc2 = self.fc2(fc1)
        output = self.fc_dp(self.fc_act(fc2))
        return output

# net1 = ResNetC1_0()
# a = torch.Tensor(1, 5, 174, 18)
# out1 = net1(a)
# print(out1.size())


class PSPDec(nn.Module):
    def __init__(self, nIn, nOut, downSize, upSize):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(downSize),
            nn.Conv2d(nIn, nOut, 1, bias=False),
            nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(nOut),
            nn.Upsample(size=upSize, mode='bilinear')
        )

    def forward(self, x):
        return self.features(x)

class pre_train(nn.Module):
    def __init__(self,):
        super().__init__()
        self.resNet = ResNetC1()
        self.up_1 = PSPDec(128, 128, 128, (20, 18))
        self.up_2 = PSPDec(128, 192, 128, (42, 18))
        self.up_3 = PSPDec(192, 128, 192, (42, 18))
        self.up_4 = PSPDec(128, 128, 128, (42, 18))
        self.up_5 = PSPDec(128, 64, 128, (86, 18))
        self.up_6 = PSPDec(64, 5, 64, (174, 18))
        self.up_7 = PSPDec(5, 5, 5, (174, 18))

    def forward(self, input1):
        output0 = self.resNet.level1(input1)  # (1, 5, 174, 18)
        output1_0 = self.resNet.level2(output0)  # (1, 64, 86, 18)
        output1 = self.resNet.level2_0(output1_0)  # (1, 64, 86, 18)
        output1 = self.resNet.level2_1(output1)  # (1, 64, 86, 18)

        output1 = self.resNet.br_2(torch.cat([output1_0, output1], 1))  # (1, 128, 86, 18)

        output2_0 = self.resNet.level3_0(output1)  # (1, 64, 42, 18)
        output2 = self.resNet.level3_1(output2_0)  # (1, 64, 42, 18)
        output2 = self.resNet.level3_2(output2)  # (1, 64, 42, 18)

        output2 = self.resNet.br_2(torch.cat([output2_0, output2], 1))  # (1, 128, 42, 18)

        output3 = self.resNet.level4_1(output2)  # (1, 128, 42, 18)
        output3 = self.resNet.level4_2(output3)  # (1, 128, 42, 18)

        output3 = self.resNet.br_4(torch.cat([output2_0, output3], 1))  # (1, 192, 42, 18)

        l5_0 = self.resNet.level4_0(output3)
        l5_1 = self.resNet.level4_1(l5_0)
        l5_2 = self.resNet.level4_2(l5_1)
        l5_con = self.resNet.br_con_4(torch.cat([l5_0, l5_2], 1))  # (1, 256, 20, 18)

        l6_0 = self.resNet.level5_0(l5_con)
        l6_1 = self.resNet.level5_1(l6_0)
        l6_2 = self.resNet.level5_2(l6_1)
        l6_con = self.resNet.br_con_5(torch.cat([l6_0, l6_2], 1))  # (1, 256, 9, 18)

        output = self.up_1(l6_con)
        output = self.up_2(output)
        output = self.up_3(output)
        output = self.up_4(output)
        output = self.up_5(output)
        output = self.up_6(output)
        output = self.up_7(output)

        return output







