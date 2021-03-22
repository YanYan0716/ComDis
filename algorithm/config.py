import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# dataset
ROOT_PATH = 'F:\\PROJECT\\ComDis\\algorithm\\data'
TRAIN_PATH = 'F:\\PROJECT\\ComDis\\algorithm\\data\\train.csv'
IMG_SIZE = 256
CROP_SIZE = 224
BATCH_SIZE = 2

# model
BACKBONE_ARCH = 'resnet18'
PRETRAIN_BACKARCH = True
FTS_DIM = 256
CONTINUE = False

# optimizer
WARMUP_EPOCH = 10
LR = 0.1
MOMENTUM = 0.9

# training
TOTAL_EPOCH = 30
LOG_BATCHSIZE = 20
EVAL = 3
SAVE_PATH = './weights/Best.pth'
# eval
TEST_PATH = 'F:\\PROJECT\\ComDis\\algorithm\\data\\train.csv'