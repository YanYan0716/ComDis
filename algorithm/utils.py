import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
from torch.optim.lr_scheduler import LambdaLR
import math
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image


import algorithm.config as config


def Cos_warmup(optimizer, epoch_warmup, epoch_training, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_epoch):
        if current_epoch < epoch_warmup:
            return float(current_epoch)/float(max(1, epoch_warmup))

        process = float(current_epoch-epoch_warmup)/\
                  float(max(1, epoch_training-epoch_warmup))
        return max(0.0, 0.5*(1.0+math.cos(math.pi*float(num_cycles)*2.0*process)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(state, savepath):
    if os.path.exists(savepath) is False:
        os.mkdir(savepath)
    save_dir = os.path.join(savepath, 'best.pth.tar')
    try:
        if int(torch.__version__.strip('.')[2]) > 4:
            print(torch.__version__, 'False')
            torch.save(state, save_dir, _use_new_zipfile_serialization=False)
        else:
            print(torch.__version__, 'True')
            torch.save(state, save_dir)
    except:
        print('save failed !!!, please check again !!!')


def testCos_warmup():
    weights = torch.randn((1), requires_grad=True)
    target = torch.zeros((1))
    optimizer = torch.optim.SGD([weights], lr=0.1, momentum=0.9)
    cosWarmUp = Cos_warmup(
        optimizer,
        epoch_warmup=config.WARMUP_EPOCH,
        epoch_training=config.TOTAL_EPOCH
    )
    lr_list, epoch_list = list(), list()

    for epoch in range(0, config.TOTAL_EPOCH):
        lr_list.append(cosWarmUp.get_lr())
        epoch_list.append(epoch)
        for i in range(10):
            loss = torch.pow((weights-target), 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        cosWarmUp.step()
    plt.plot(epoch_list, lr_list, label='step LR Scheduler')
    plt.xlabel('epoch')
    plt.ylabel('Learning rate')
    plt.legend()
    plt.show()


def EarnName():
    path = "./Dataset"
    original_images = []
    pict_name = open('name.txt', 'w+')
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            original_images.append(root + "/" + filename)
    original_images = sorted(original_images)
    print('num: {}'.format(len(original_images)))
    for filename in (original_images):
        filename = filename.replace('\\', '/')
        # print(filename)
        pict_name.write(filename + '\n')
    pict_name.close()


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def Resize(imgPath):
    img = Image.open(imgPath).convert('RGB')
    img = img.resize((256, 256), resample=Image.BILINEAR)
    img.save(imgPath)


if __name__ == '__main__':
    # testCos_warmup()
    # EarnName()

    img_path = open('name.txt', 'r').readlines()
    for i in range(len(img_path)):
        imgpath = img_path[i].split()[0]
        try:
            Resize(imgpath)
        except:
            print(imgpath)
            break
    # Resize('./data/1642440.jpg')
    print('ok')