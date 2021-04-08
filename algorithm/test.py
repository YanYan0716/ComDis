import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch
import matplotlib.pyplot as plt

import algorithm.config as config
from algorithm.model import Model, Model2
from view import imageTrans, genImgMask
from algorithm.trans import OriTest, Trans1, normlize
from algorithm.TripletDataset import transform_invert


def evalution(dataLoader, model):
    total_number = 0
    correct_number1 = 0
    correct_number2 = 0
    model.eval()
    for idx, (anchor, img1, img2, mask, lable) in enumerate(dataLoader):
        total_number += anchor.shape[0]
        anchor, img1, img2, mask, lable = anchor.to(config.DEVICE), img1.to(config.DEVICE), img2.to(config.DEVICE), mask.to(
            config.DEVICE), lable.to(config.DEVICE)
        # anchor = anchor[:, :3, :, :] * anchor[:, -1:, :, :]
        # img1 = img1[:, :3, :, :] * img1[:, -1:, :, :]
        # img2 = img2[:, :3, :, :] * img2[:, -1:, :, :]

        imgs = torch.cat([anchor, img1, img2], dim=0)
        # out1 = model.conv(imgs)
        out1 = model.model(imgs)
        out1 = model.flatten(out1)
        out1 = model.triplet(out1)

        fts = torch.cat(
            [
                out1[:config.BATCH_SIZE],
                out1[config.BATCH_SIZE: config.BATCH_SIZE * 2],
                out1[-config.BATCH_SIZE:]
            ], dim=-1)
        output = model.classifier(fts)
        output = torch.sigmoid(output).ge(0.5).type(torch.float32).squeeze(dim=-1)
        result = output.eq(mask).type(torch.float32)
        correct_number1 += torch.sum(result)

        # output_ = model.class2(out1[:config.BATCH_SIZE])
        # pred = torch.max(output_.data, 1)
        # for i in range(len(lable)):
        #     if pred[1][i] == lable[i]:
        #         correct_number2 += 1

    acc1 = correct_number1 / total_number * 100
    acc2 = correct_number2 / total_number * 100
    return acc1, acc2, correct_number1, correct_number2, total_number


def evalution2(dataLoader, model):
    total_number = 0
    correct_number1 = 0
    correct_number2 = 0
    model.eval()
    for idx, (anchor, img1, img2, mask, lable) in enumerate(dataLoader):
        total_number += anchor.shape[0]
        anchor, img1, img2, mask, lable = anchor.to(config.DEVICE), img1.to(config.DEVICE), img2.to(config.DEVICE), mask.to(
            config.DEVICE), lable.to(config.DEVICE)

        imgs = torch.cat([anchor,img1, img2], dim=0)
        out1 = model.model(imgs)
        out1 = model.flatten(out1)
        out1 = model.triplet(out1)

        fts = torch.cat(
            [
                out1[:config.BATCH_SIZE],
                out1[config.BATCH_SIZE*1:config.BATCH_SIZE*2],
                out1[-config.BATCH_SIZE:]
            ], dim=-1)
        output = model.classifier(fts)
        output = torch.sigmoid(output).ge(0.5).type(torch.float32).squeeze(dim=-1)
        result = output.eq(mask).type(torch.float32)
        correct_number1 += torch.sum(result)

    acc1 = correct_number1 / total_number * 100
    acc2 = correct_number2 / total_number * 100
    return acc1, acc2, correct_number1, correct_number2, total_number


if __name__ == '__main__':
    net = Model2(fts_dim=config.FTS_DIM)
    checkpoint = torch.load(config.BEST_PATH, map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    net.eval()

    firstImg, firstImg_, secondImg = imageTrans('./test/1646233.jpg', './test/1646233.jpg')

    imgs = torch.cat([firstImg, firstImg_, secondImg], dim=0)
    with torch.no_grad():
        out1 = net.model(imgs)
        out1 = net.flatten(out1)
        out1 = net.triplet(out1)
        fts = torch.cat([out1[0,], out1[1,], out1[2,]], dim=-1).unsqueeze(dim=0)
        # print(fts.shape)
        output = net.classifier(fts)
        output_ = torch.sigmoid(output).ge(0.50).type(torch.float32).squeeze(dim=-1)
        output_ = str(output_.numpy())
        out0 = str(torch.sigmoid(output).squeeze(dim=-1).detach().numpy())
        print(out0)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(transform_invert(firstImg[0], normlize))
    plt.subplot(1, 3, 2)
    plt.imshow(transform_invert(firstImg_[0], normlize))
    plt.subplot(1, 3, 3)
    plt.imshow(transform_invert(secondImg[0], normlize))
    plt.show()