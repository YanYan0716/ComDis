'''对模型进行难例挖掘'''
import sys

sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch
from PIL import Image
import pandas as pd
import os
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data

from algorithm.trans import OriTest, Trans1, normlize
import algorithm.config as config
from algorithm.model import Model


def imageTrans(img1, img2):
    img1 = Image.open(img1).convert('RGB')
    firstImg = OriTest(img1)
    firstImg = normlize(firstImg)

    firstImg_ = Trans1(img1)
    firstImg_ = normlize(firstImg_)

    img2 = Image.open(img2).convert('RGB')
    secodeImg = OriTest(img2)
    secodeImg = normlize(secodeImg)

    # secodeImg = secodeImg * Img2Mask
    firstImg = firstImg.unsqueeze(dim=0)
    firstImg_ = firstImg_.unsqueeze(dim=0)
    secodeImg = secodeImg.unsqueeze(dim=0)
    return firstImg, firstImg_, secodeImg


def hardsample():
    net = Model(fts_dim=config.FTS_DIM).to(config.DEVICE)
    checkpoint = torch.load(config.BEST_PATH, map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    net.eval()

    hardSamples = {}
    OriImgList = pd.read_csv('train.csv')['name']
    SeaImgList = pd.read_csv('train.csv')['name']
    print(len(OriImgList))

    for i in range(len(OriImgList)):
        hard_num = []
        OriImg = OriImgList[i]
        OriImg = os.path.join('./data', OriImg)
        for j in range(len(SeaImgList)):
            if i == j:
                pass
            else:
                SeaImg = SeaImgList[j]
                SeaImg = os.path.join('./data', SeaImg)
                firstImg, firstImg_, secondImg = imageTrans(OriImg, SeaImg)
                imgs = torch.cat([firstImg, firstImg_, secondImg], dim=0)
                out1 = net.model(imgs)
                out1 = net.flatten(out1)
                out1 = net.triplet(out1)
                fts = torch.cat([out1[:1], out1[1:2], out1[-1:]], dim=-1)
                output = net.classifier(fts)
                # output_ = torch.sigmoid(output).ge(0.85).type(torch.float32).squeeze(dim=-1)
                out0 = torch.sigmoid(output).squeeze(dim=-1).detach().numpy()
                hard_num.append((SeaImgList[j], out0[0]))
                # print(j, out0, SeaImgList[j])
        hard_num.sort(key=lambda x: x[1], reverse=True)
        hardSamples[OriImgList[i]] = hard_num[:20]
        # print(hard_num[:10])
        print(hardSamples)
        np.save('./hardsam.npy', arr=hardSamples, allow_pickle=True)
        break


def hardsample2():
    net = Model(fts_dim=config.FTS_DIM).to(config.DEVICE)
    checkpoint = torch.load(config.BEST_PATH, map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    net.eval()

    hardSamples = {}
    org_fts = {}
    OriImgList = pd.read_csv('train.csv')['name']
    SeaImgList = pd.read_csv('train.csv')['name']
    print(len(OriImgList))

    for i in range(len(OriImgList)):
        hard_num = []
        OriImg = OriImgList[i]
        OriImg = os.path.join('./data', OriImg)
        firstImg, firstImg_, secondImg = imageTrans(OriImg, OriImg)
        imgs = torch.cat([firstImg, firstImg_], dim=0)
        out1 = net.model(imgs)
        out1 = net.flatten(out1)
        out1 = net.triplet(out1)
        org_ft = torch.cat([out1[:1], out1[1:2]], dim=-1)
        org_fts[OriImgList[i]] = org_ft

    np.save('./org_fts.npy', arr=org_fts, allow_pickle=True)

        # for j in range(len(SeaImgList)):
        #     if i == j:
        #         pass
        #     else:
        #         SeaImg = SeaImgList[j]
        #         SeaImg = os.path.join('./data', SeaImg)
        #         firstImg, firstImg_, secondImg = imageTrans(OriImg, SeaImg)
        #         imgs = torch.cat([firstImg, firstImg_, secondImg], dim=0)
        #         out1 = net.model(imgs)
        #         out1 = net.flatten(out1)
        #         out1 = net.triplet(out1)
        #         fts = torch.cat([out1[:1], out1[1:2], out1[-1:]], dim=-1)
        #         output = net.classifier(fts)
        #         # output_ = torch.sigmoid(output).ge(0.85).type(torch.float32).squeeze(dim=-1)
        #         out0 = torch.sigmoid(output).squeeze(dim=-1).detach().numpy()
        #         hard_num.append((SeaImgList[j], out0[0]))
        #         # print(j, out0, SeaImgList[j])
        # hard_num.sort(key=lambda x: x[1], reverse=True)
        # hardSamples[OriImgList[i]] = hard_num[:20]
        # # print(hard_num[:10])
        # print(hardSamples)
        # np.save('./hardsam.npy', arr=hardSamples, allow_pickle=True)
        # break


if __name__ == '__main__':
    hardsample2()
