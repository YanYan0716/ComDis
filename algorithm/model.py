import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torchvision.models as models
from torch import nn
import torch
import random

import algorithm.config as config
from algorithm.utils import save_checkpoint


class Model(nn.Module):
    def __init__(self, fts_dim=256):
        super(Model, self).__init__()
        try:
            base = getattr(models, config.BACKBONE_ARCH)(pretrained=config.PRETRAIN_BACKARCH)
            self.model = nn.Sequential(*list(base.children())[:-1])
            self.base_output = base.fc.in_features
        except:
            raise ValueError('please check config.BACKBONE_ARCH ')
        # self.conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.triplet = nn.Linear(self.base_output, fts_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fts_dim * 3, fts_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(fts_dim*2, fts_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fts_dim, 1)
        )
        # self.class2 = nn.Sequential(  # 辅助分类函数
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, config.CLASSES_NUM)
        # )

    def forward(self, x, mask):
        N = int(x.shape[0] / 4)
        # 提特征
        # x = self.conv(x)
        x = self.model(x)
        x = self.flatten(x)
        # triplet
        out = self.triplet(x)
        ori = out[0:N]
        pos1 = out[N:2*N]
        pos2 = out[2*N:3*N]
        neg = out[-N:]
        if random.randint(0, 1):
            out1 = torch.cat([ori, pos1, neg], dim=0)
        else:
            out1 = torch.cat([ori, pos2, neg], dim=0)

        # 二分类
        classTensor = torch.zeros(size=(N, out.shape[-1]*3)).to(config.DEVICE)
        for i in range(N):
            if mask[i]:  # mask=1表示是同一类，[ori, pos1, pos2]
                classTensor[i] = torch.cat([ori[i], pos1[i], pos2[i]], dim=-1)
            else:  # mask=0表示是不同类，[ori, pos1, neg]
                classTensor[i] = torch.cat([ori[i], pos1[i], neg[i]], dim=-1)
        out2 = self.classifier(classTensor)
        # 辅助分类
        # out3 = self.class2(ori)
        return out1, out2,# out3


class Model2(nn.Module):
    def __init__(self, fts_dim=256):
        super(Model2, self).__init__()
        try:
            base = getattr(models, config.BACKBONE_ARCH)(pretrained=config.PRETRAIN_BACKARCH)
            self.model = nn.Sequential(*list(base.children())[:-1])
            self.base_output = base.fc.in_features
        except:
            raise ValueError('please check config.BACKBONE_ARCH ')
        self.flatten = nn.Flatten()
        self.triplet = nn.Sequential(
            nn.Linear(self.base_output, fts_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fts_dim*2, fts_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fts_dim * 3, fts_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fts_dim*2, fts_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fts_dim, 1)
        )

    def forward(self, x, mask):
        N = int(x.shape[0] / 4)
        # 提特征
        # x = self.conv(x)
        x = self.model(x)
        x = self.flatten(x)
        # triplet
        out = self.triplet(x)
        ori = out[0:N]
        pos1 = out[N:2*N]
        pos2 = out[2*N:3*N]
        neg = out[-N:]

        # 二分类
        classTensor = torch.zeros(size=(N, out.shape[-1]*3)).to(config.DEVICE)
        out1_2 = torch.zeros(size=(N, out.shape[-1])).to(config.DEVICE)
        for i in range(N):
            if mask[i]:  # mask=1表示不同类
                out1_2[i] = pos1[i]
                classTensor[i] = torch.cat([ori[i], pos1[i], pos2[i]], dim=-1)
            else:  # mask=0表示同类
                out1_2[i] = neg[i]
                classTensor[i] = torch.cat([ori[i], pos1[i], neg[i]], dim=-1)
        out2 = self.classifier(classTensor)
        return (ori, out1_2), out2


def test():
    x = torch.randn((2, 3, 224, 224))
    x = torch.cat([x, x, x, x], dim=0)
    print(x.shape)
    M = Model(fts_dim=config.FTS_DIM)
    print(M.flatten(M.model(x)).shape)
    # out1, out2,  = M(x, torch.tensor([1, 0]))
    #
    # print(out1.shape, out2.shape, )


if __name__ == '__main__':
    # test()
    M = Model(fts_dim=config.FTS_DIM)
    for para in M.flatten.parameters():
        para.requires_grad = False
        print(para.requires_grad)
    # nn.Conv2d()
    nn.LeakyReLU(inplace=True)