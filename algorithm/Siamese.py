import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torchvision.models as models
from torch import nn
import torch
import random

from algorithm.model import Model2
import algorithm.config as config


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_model = Model2(fts_dim=config.FTS_DIM)
        self.model = nn.Sequential(
            self.base_model.model,
            self.base_model.flatten,
            self.base_model.triplet
        )

    def forward_once(self, x):
        output = self.model(x)
        return output

    def forward(self, input1, input2, input3, input4):

        N = int(input1.shape[0])
        input1_ = torch.zeros(size=input1.shape).to(config.DEVICE)
        for i in range(N):
            if input4[i]:  # mask=1表示不同类
                input1_[i] = input3[i]
            else:  # mask=0表示同类
                input1_[i] = input2[i]

        output1 = self.forward_once(input1)
        output1_ = self.forward_once(input1_)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3, output1_


if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224))
    x = torch.cat([x, x, x, x], dim=0)
    print(x.shape)
    M = SiameseNetwork()
    y = M(x, x)
    print(y[0].shape, y[1].shape)