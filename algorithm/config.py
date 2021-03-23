import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# dataset
# ROOT_PATH = 'F:\\PROJECT\\ComDis\\algorithm\\data'
# TRAIN_PATH = 'F:\\PROJECT\\ComDis\\algorithm\\data\\train.csv'
# ROOT_PATH = ''  # google
# TRAIN_PATH = '/content/cifar/train.csv'
ROOT_PATH = ''  # kaggle
TRAIN_PATH = '../input/cifar10/cifar/train.csv'
IMG_SIZE = 256
CROP_SIZE = 224
BATCH_SIZE = 64

# model
BACKBONE_ARCH = 'resnet18'
PRETRAIN_BACKARCH = True
FTS_DIM = 256
CONTINUE = False
ALPHA = 2
BETAL = 1

# optimizer
WARMUP_EPOCH = 0
LR = 0.001
MOMENTUM = 0.9

# training
TOTAL_EPOCH = 300
LOG_BATCHSIZE = 50
EVAL = 1
SAVE_PATH = './weights'
# eval
# TEST_PATH = '/content/cifar/test.csv'  # google
TEST_PATH = '../input/cifar10/cifar/test.csv'