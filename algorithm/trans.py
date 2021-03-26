import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torchvision.transforms as transforms


import algorithm.config as config
from algorithm.AutoAugment import AutoAugment
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# train
OriTrain = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

PosTrans1 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

PosTrans2 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

NegTrans = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# test
OriTest = transforms.Compose([
    transforms.Resize((config.CROP_SIZE, config.CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

Trans1 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

Trans2 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])