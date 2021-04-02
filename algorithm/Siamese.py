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

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224))
    x = torch.cat([x, x, x, x], dim=0)
    print(x.shape)
    M = SiameseNetwork()
    y = M(x, x)
    print(y[0].shape, y[1].shape)