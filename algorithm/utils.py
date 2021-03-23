import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
from torch.optim.lr_scheduler import LambdaLR
import math
import torch
import matplotlib.pyplot as plt


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
    try:
        torch.save(state, savepath)
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

    for epoch in range(config.TOTAL_EPOCH):
        lr_list.append(cosWarmUp.get_lr())
        epoch_list.append(epoch)
        for i in range(10):
            loss = torch.pow((weights-target), 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        cosWarmUp.step()
    # plt.plot(epoch_list, lr_list, label='step LR Scheduler')
    # plt.xlabel('epoch')
    # plt.ylabel('Learning rate')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    testCos_warmup()