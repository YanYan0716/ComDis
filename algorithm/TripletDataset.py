import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import os
from torch.utils import data
from torchvision import transforms
import pandas as pd
import torch
import random
from PIL import Image
import matplotlib.pylab as plt
import numpy as np


import algorithm.config as config
import algorithm.trans as trans


class TripletDataset(data.Dataset):
    def __init__(self, root_dir='./', img_dir='./', transform=None, train=True):
        self.RootDir = root_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.ImgList = pd.read_csv(img_dir)['name']
        else:
            self.ImgList = pd.read_csv(img_dir)['file_name']
        self.ImgsLen = self.__len__()

    def __getitem__(self, index):
        # earn images' path
        NegIndex = random.randint(0, self.ImgsLen-1)
        while index == NegIndex:
            NegIndex = random.randint(0, self.ImgsLen - 1)
        OriPath = os.path.join(self.RootDir, self.ImgList[index])
        NegPath = os.path.join(self.RootDir, self.ImgList[NegIndex])

        # img augment
        if self.train is True:
            assert self.transform is not None, 'please set transform for training ...'
            OriImg_ = Image.open(OriPath).convert('RGB')
            OriImg = self.transform['OriTrans'](OriImg_)
            PosImg1 = self.transform['PosTrans1'](OriImg_)
            PosImg2 = self.transform['PosTrans2'](OriImg_)
            NegImg_ = Image.open(NegPath).convert('RGB')
            NegImg =self.transform['NegTrans'](NegImg_)
            # mask
            mask = random.randint(0, 1)
            return OriImg, PosImg1, PosImg2, NegImg, mask
        else:
            assert self.transform is not None, 'please set transform for testing ...'
            OriImg_ = Image.open(OriPath).convert('RGB')
            OriImg = self.transform['OriTrans'](OriImg_)
            Img1 = self.transform['Trans1'](OriImg_)
            if random.randint(0, 1):  # mask=1表示是同一类，[ori, pos1, pos2]
                mask = 1
                Img2 = self.transform['Trans2'](OriImg_)
                return OriImg, Img1, Img2, mask
            else:
                mask = 0  # mask=0表示是不同类，[ori, pos1, neg]
                NegImg_ = Image.open(NegPath).convert('RGB')
                Img2 = self.transform['Trans2'](NegImg_)
                return OriImg, Img1, Img2, mask

    def __len__(self):
        return len(self.ImgList)


def transform_invert(img_tensor, data_transform):
    if 'Normalize' in str(data_transform):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), data_transform.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)# 交换矩阵通道
    img_tensor = np.array(img_tensor)*255
    if img_tensor.shape[2] == 3:
        img_tensor = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img_tensor = Image.fromarray(img_tensor.astype('uint8').squeeze())
    else:
        raise Exception('image channel is 1 or 3, please check it')
    return img_tensor


def test():
    train_trans = {
        'OriTrans': trans.OriTrain,
        'PosTrans1': trans.PosTrans1,
        'PosTrans2': trans.PosTrans2,
        'NegTrans': trans.NegTrans
    }
    DS = TripletDataset(
        root_dir=config.ROOT_PATH,
        transform=train_trans,
        img_dir=config.TRAIN_PATH,
        train=True
    )

    train_loader = data.DataLoader(
        DS,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    for i in range(2):
        for index, (OriImg, PosImg1, PosImg2, NegImg, mask) in enumerate(train_loader):
            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(transform_invert(OriImg[0], 'None'))
            plt.subplot(1, 4, 2)
            plt.imshow(transform_invert(PosImg1[0], 'None'))
            plt.subplot(1, 4, 3)
            plt.imshow(transform_invert(PosImg2[0], 'None'))
            plt.subplot(1, 4, 4)
            plt.imshow(transform_invert(NegImg[0], 'None'))
            plt.show()
            # print(type(mask[0]))
            # print(mask.shape)
            break


if __name__ == '__main__':
    test()