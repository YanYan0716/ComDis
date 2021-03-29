import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch

import algorithm.config as config
from algorithm.model import Model


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
        out1 = model.model(imgs)
        out1 = model.flatten(out1)

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


if __name__ == '__main__':
    net = Model(fts_dim=config.FTS_DIM)
    checkpoint = torch.load(config.BEST_PATH, map_location='cpu')
    print(checkpoint.keys())
    net.load_state_dict(checkpoint['model'])
