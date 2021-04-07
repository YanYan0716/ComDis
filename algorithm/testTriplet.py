import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

import algorithm.config as config
from view import imageTrans, genImgMask
from algorithm.trans import OriTest, Trans1, normlize
from algorithm.TripletDataset import transform_invert
from algorithm.Siamese import SiameseNetwork


if __name__ == '__main__':
    net = SiameseNetwork()
    checkpoint = torch.load(config.BEST_PATH, map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    net.eval()

    classifier = nn.Sequential(
            nn.Linear(config.FTS_DIM * 3, config.FTS_DIM * 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.FTS_DIM*2, config.FTS_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(config.FTS_DIM, 1)
        )

    checkpoint = torch.load('F:\\PROJECT\\ComDis\\weights\\bC.pth.tar', map_location=config.DEVICE)
    classifier.load_state_dict(checkpoint['model'])
    classifier.eval()

    firstImg, firstImg_, secondImg = imageTrans('./test/461490.jpg', './test/461439.jpg')
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(transform_invert(firstImg[0], normlize))
    plt.subplot(1, 3, 2)
    plt.imshow(transform_invert(firstImg_[0], normlize))
    plt.subplot(1, 3, 3)
    plt.imshow(transform_invert(secondImg[0], normlize))
    plt.show()

    with torch.no_grad():
        out1 = net.forward_once(firstImg)
        out2 = net.forward_once(firstImg_)
        out3 = net.forward_once(secondImg)

        fts = torch.cat([out1, out2, out3], dim=-1)

        output = classifier(fts)
        output_ = torch.sigmoid(output).ge(0.50).type(torch.float32).squeeze(dim=-1)
        output_ = str(output_.numpy())
        out0 = str(torch.sigmoid(output).squeeze(dim=-1).detach().numpy())
        # 1是不一致
        print(out0, output_)

        print('---------')
        dis1 = F.pairwise_distance(out1, out2)
        dis2 = F.pairwise_distance(out1, out3)
        # print(dis2)
        if (dis1*1.2)>=(dis2):
            print('same')
        else:
            print('diff')
        print(dis1, dis2)