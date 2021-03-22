import sys

sys.path.append('./')
import torch

import algorithm.config as config


def evalution(dataLoader, model):
    total_number = 0
    correct_number = 0
    model.eval()
    for idx, (anchor, img1, img2, mask) in enumerate(dataLoader):
        total_number += anchor.shape[0]
        anchor, img1, img2, mask = anchor.to(config.DEVICE), img1.to(config.DEVICE), img2.to(config.DEVICE), mask.to(
            config.DEVICE)
        imgs = torch.cat([anchor, img1, img2], dim=0)
        out1 = model.model(imgs)

        fts = torch.cat(
            [
                out1[:config.BATCH_SIZE],
                out1[config.BATCH_SIZE: config.BATCH_SIZE * 2],
                out1[-config.BATCH_SIZE:]
            ], dim=-1)
        output = model.classifier(fts)
        output = torch.sigmoid(output)
        correct_number += torch.sum(output)
    acc = correct_number / total_number
    return acc
