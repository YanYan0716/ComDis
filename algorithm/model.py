import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torchvision.models as models
from torch import nn
import torch
import random

import algorithm.config as config


class Model(nn.Module):
    def __init__(self, fts_dim=256):
        super(Model, self).__init__()
        try:
            self.model = getattr(models, config.BACKBONE_ARCH)(pretrained=config.PRETRAIN_BACKARCH)
        except:
            raise ValueError('please check config.BACKBONE_ARCH ')
        self.model.fc = nn.Linear(self.model.fc.in_features, fts_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fts_dim * 3, fts_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(fts_dim*2, fts_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fts_dim, 1)
        )


    def forward(self, x, mask):
        N = int(x.shape[0] / 4)
        x = self.model(x)
        ori = x[0:N]
        pos1 = x[N:2*N]
        pos2 = x[2*N:3*N]
        neg = x[-N:]

        if random.randint(0, 1):
            out1 = torch.cat([ori, pos1, neg], dim=0)
        else:
            out1 = torch.cat([ori, pos2, neg], dim=0)

        classTensor = torch.zeros(size=(N, x.shape[-1]*3)).to(config.DEVICE)
        for i in range(N):
            if mask[i]:  # mask=1表示是同一类，[ori, pos1, pos2]
                classTensor[i] = torch.cat([ori[i], pos1[i], pos2[i]], dim=-1)
            else:  # mask=0表示是不同类，[ori, pos1, neg]
                classTensor[i] = torch.cat([ori[i], pos1[i], neg[i]], dim=-1)
        out2 = self.classifier(classTensor)
        return out1, out2


def test():
    x = torch.randn((2, 3, 224, 224))
    x = torch.cat([x, x, x, x], dim=0)
    print(x.shape)
    M = Model(fts_dim=config.FTS_DIM)
    out1, out2 = M(x, torch.tensor([1, 0]))
    print(out1.shape, out2.shape)


if __name__ == '__main__':
    test()