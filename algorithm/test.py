import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch

import algorithm.config as config
from algorithm.model import Model


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
    acc = correct_number / total_number * 100
    return acc, correct_number, total_number


if __name__ == '__main__':
    net = Model(fts_dim=config.FTS_DIM)
    weights_path = 'F:\\PROJECT\\ComDis\\weights\\best.pth.tar'
    checkpoint = torch.load(weights_path, map_location='cpu')
    print(checkpoint.keys())
    net.load_state_dict(checkpoint['model'])
