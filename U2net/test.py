import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from U2net.U2model import U2NETP
from algorithm.config import DEVICE
from U2net.trans import preprocess
from algorithm.TripletDataset import transform_invert
MODEL_DIR = './U2net/weights/u2netp.pth'
IMG_DIR = '5.jpg'


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def test():
    net = U2NETP(3, 1)
    net.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
    net.eval()
    img = preprocess(IMG_DIR)
    img = img.unsqueeze(dim=0).type(torch.FloatTensor)
    # img = Variable(img)
    # d1, d2, d3, d4, d5, d6, d7 = net(img)
    d1 = net(img)
    pred = d1[:, 0, :, :]
    pred = normPRED(pred)

    predict = pred
    print('----------')
    print(predict.shape)
    res = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    predict = res(predict)
    print(predict.shape)

    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    # print(type(predict_np[0][0]*255))
    im = Image.fromarray(predict_np*255).convert('L')
    # image = io.imread(IMG_DIR)
    # imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    print(im.size)

    im.save('test.jpg')
    print('ok')


if __name__ == '__main__':
    test()