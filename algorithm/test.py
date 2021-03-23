import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
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
        output = torch.sigmoid(output).ge(0.5).type(torch.float32).squeeze(dim=-1)
        result = output.eq(mask).type(torch.float32)
        correct_number += torch.sum(result)
    acc = correct_number / total_number
    return acc
