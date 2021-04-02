import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

    firstImg, firstImg_, secondImg = imageTrans('./test/462938.jpg', './test/462983.jpg')
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
        dis1 = F.pairwise_distance(out1, out2)
        dis2 = F.pairwise_distance(out1, out3)
        print(dis2)
        # if dis1>=(dis2):
        #     print('same')
        # else:
        #     print('diff')
        # print(dis1, dis2)