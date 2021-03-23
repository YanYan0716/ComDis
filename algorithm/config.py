import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# dataset
# ROOT_PATH = 'F:\\PROJECT\\ComDis\\algorithm\\data'
# TRAIN_PATH = 'F:\\PROJECT\\ComDis\\algorithm\\data\\train.csv'
ROOT_PATH = ''
TRAIN_PATH = '/content/cifar/train.csv'
IMG_SIZE = 256
CROP_SIZE = 224
BATCH_SIZE = 32

# model
BACKBONE_ARCH = 'resnet18'
PRETRAIN_BACKARCH = True
FTS_DIM = 256
CONTINUE = False

# optimizer
WARMUP_EPOCH = 50
LR = 0.1
MOMENTUM = 0.9

# training
TOTAL_EPOCH = 300
LOG_BATCHSIZE = 20
EVAL = 3
SAVE_PATH = './weights/Best.pth'
# eval
TEST_PATH = '/content/cifar/test.csv'