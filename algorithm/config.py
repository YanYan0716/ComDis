import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# dataset
# ROOT_PATH = 'E:\\Algorithm\\ComDis\\algorithm\\data'
# TRAIN_PATH = '.\\algorithm\\data\\train.csv'
# ROOT_PATH = ''  # google
# TRAIN_PATH = '/content/cifar/train.csv'
ROOT_PATH = '../input/cub-200-2011/CUB_200_2011/images'  # kaggle
# ROOT_PATH = ''
# TRAIN_PATH = '../input/cifar10/cifar/label4000.csv'
TRAIN_PATH = '../input/cub-200-2011/CUB_200_2011/train.csv'
IMG_SIZE = 256
CROP_SIZE = 224
BATCH_SIZE = 32
U2WEITHS_DIR = './ComDis/U2net/weights/u2netp.pth'

# model
BACKBONE_ARCH = 'resnet34'
PRETRAIN_BACKARCH = True
FTS_DIM = 256
ALPHA = 2
BETAL = 1
GAMMA = 1
CLASSES_NUM = 10

# optimizer
WARMUP_EPOCH = 0
LR = 0.001
MOMENTUM = 0.9

# training
TOTAL_EPOCH = 300
START_EPOCH = 0
LOG_BATCHSIZE = 20
EVAL = 1
SAVE_PATH = './weights'
CONTINUE = False
CONTINUE_PATH = '../input/bestww/best.pth.tar'


# eval
# TEST_PATH = '/content/cifar/test.csv'  # google
# TEST_PATH = '.\\algorithm\\data\\train.csv'
TEST_PATH = '../input/cub-200-2011/CUB_200_2011/test.csv'
BEST_PATH = 'E:\\Algorithm\\ComDis\\weights\\best.pth.tar'