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
from U2net.U2model import U2NETP


class TripletDataset(data.Dataset):
    def __init__(self, root_dir='./', img_dir='./', transform=None, train=True):
        self.RootDir = root_dir
        self.transform = transform
        self.train = train
        if self.train:
            info_file = pd.read_csv(img_dir)
            self.ImgList = info_file['file_name']
            self.LabelList = info_file['label']
        else:
            info_file = pd.read_csv(img_dir)
            self.ImgList = info_file['file_name']
            self.LabelList = info_file['label']
        self.ImgsLen = self.__len__()

        # 生成图像的显著mask
        self.U2net = U2NETP(3, 1)
        self.U2net.load_state_dict(torch.load(config.U2WEITHS_DIR, map_location=config.DEVICE))
        self.U2net.eval()

    def __getitem__(self, index):
        # earn images' path
        NegIndex = random.randint(0, self.ImgsLen-1)
        OriLabel = self.LabelList[index]
        while (index == NegIndex) or (OriLabel != self.LabelList[NegIndex]):
            NegIndex = random.randint(0, self.ImgsLen - 1)
        OriPath = os.path.join(self.RootDir, self.ImgList[index])
        NegPath = os.path.join(self.RootDir, self.ImgList[NegIndex])

        # img augment
        if self.train is True:
            assert self.transform is not None, 'please set transform for training ...'
            OriImg_ = Image.open(OriPath).convert('RGB')

            OriImg = self.transform['OriTrans'](OriImg_)
            # 添加原图的mask
            if random.random() < 0.:
                OriImgMask = self.genImgMask(OriImg)
                OriImg = torch.cat([OriImg, OriImgMask], dim=0)
                # OriImg = OriImg * OriImgMask

            PosImg1 = self.transform['PosTrans1'](OriImg_)
            if random.random() < 0.:
                Pos1Mask = self.genImgMask(PosImg1)
                PosImg1 = torch.cat([PosImg1, Pos1Mask], dim=0)
                # PosImg1 = PosImg1 * Pos1Mask

            PosImg2 = self.transform['PosTrans2'](OriImg_)
            if random.random() < 0.:
                Pos2Mask = self.genImgMask(PosImg2)
                PosImg2 = torch.cat([PosImg2, Pos2Mask], dim=0)
                # PosImg2 = PosImg2 * Pos2Mask

            NegImg_ = Image.open(NegPath).convert('RGB')
            NegImg =self.transform['NegTrans'](NegImg_)
            if random.random() < 0.:
                NegMask = self.genImgMask(NegImg)
                NegImg = torch.cat([NegImg, NegMask], dim=0)
                # NegImg = NegImg* NegMask

            # mask
            mask = random.randint(0, 1)
            return OriImg, PosImg1, PosImg2, NegImg, mask, OriLabel
        else:
            assert self.transform is not None, 'please set transform for testing ...'
            OriImg_ = Image.open(OriPath).convert('RGB')

            OriImg = self.transform['OriTrans'](OriImg_)
            # OriImgMask = self.genImgMask(OriImg)
            # OriImg = torch.cat([OriImg, OriImgMask], dim=0)

            Img1 = self.transform['Trans1'](OriImg_)
            # Img1Mask = self.genImgMask(Img1)
            # Img1 = torch.cat([Img1, Img1Mask], dim=0)

            if random.randint(0, 1):  # mask=1表示是同一类，[ori, pos1, pos2]
                mask = 1
                Img2 = self.transform['Trans2'](OriImg_)
                # Img2Mask = self.genImgMask(Img2)
                # Img2 = torch.cat([Img2, Img2Mask], dim=0)
                return OriImg, Img1, Img2, mask, OriLabel
            else:
                mask = 0  # mask=0表示是不同类，[ori, pos1, neg]
                NegImg_ = Image.open(NegPath).convert('RGB')
                Img2 = self.transform['Trans2'](NegImg_)
                # Img2Mask = self.genImgMask(Img2)
                # Img2 = torch.cat([Img2, Img2Mask], dim=0)
                return OriImg, Img1, Img2, mask, OriLabel

    def __len__(self):
        return len(self.ImgList)

    def normPred(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def genImgMask(self, imgTensor):
        img = imgTensor.unsqueeze(dim=0).type(torch.FloatTensor)
        d1 = self.U2net(img)
        pred = d1[:, 0, :, :]
        pred = self.normPred(pred)
        resize_pred = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.CROP_SIZE, config.CROP_SIZE)),
            transforms.ToTensor(),
        ])
        pred = resize_pred(pred)
        return pred


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
        img_tensor = Image.fromarray(img_tensor.astype('float32').squeeze())
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
    for i in range(1):
        for index, (OriImg, PosImg1, PosImg2, NegImg, mask, label) in enumerate(train_loader):
            print(OriImg.shape, PosImg1.shape, PosImg2.shape, NegImg.shape)
            plt.figure()
            plt.subplot(2, 4, 1)
            plt.imshow(transform_invert(OriImg[0][:3], trans.OriTrain))
            # plt.subplot(2, 4, 2)
            # plt.imshow(transform_invert(OriImg[0][-1:], 'None'))
            plt.subplot(2, 4, 3)
            plt.imshow(transform_invert(PosImg1[0][:3], trans.PosTrans1))
            # plt.subplot(2, 4, 4)
            # plt.imshow(transform_invert(PosImg1[0][-1:], 'None'))
            plt.subplot(2, 4, 5)
            plt.imshow(transform_invert(PosImg2[0][:3], trans.PosTrans2))
            # plt.subplot(2, 4, 6)
            # plt.imshow(transform_invert(PosImg2[0][-1:], 'None'))
            plt.subplot(2, 4, 7)
            plt.imshow(transform_invert(NegImg[0][:3], trans.NegTrans))
            # plt.subplot(2, 4, 8)
            # plt.imshow(transform_invert(NegImg[0][-1:], 'None'))
            # img = OriImg[:, :3, :, :]*OriImg[:, -1:, :, :]
            # plt.imshow(transform_invert(OriImg[0], trans.OriTrain))
            plt.show()
            # print(label)
            # print(mask.shape)
            break


if __name__ == '__main__':
    test()