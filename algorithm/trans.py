import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torchvision.transforms as transforms


import algorithm.config as config
from algorithm.AutoAugment import AutoAugment
from algorithm.swap import Randomswap
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# train
OriTrain = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    # Randomswap([5, 5]),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

PosTrans1 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    # Randomswap([5, 5]),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

PosTrans2 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    # Randomswap([5, 5]),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

NegTrans = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomCrop(size=config.CROP_SIZE),
    AutoAugment(),
    # Randomswap([5, 5]),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# test
OriTest = transforms.Compose([
    # Randomswap([7, 7]),
    transforms.Resize((config.CROP_SIZE, config.CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

Trans1 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    # transforms.RandomCrop(size=config.CROP_SIZE),
    # AutoAugment(),
    # transforms.RandomChoice([
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.Compose([
    #             transforms.RandomRotation(degrees=10, expand=True),
    #             # transforms.Resize(config.IMG_SIZE),
    #             transforms.CenterCrop(size=config.CROP_SIZE),
    #         ]),
    #         transforms.RandomCrop((config.CROP_SIZE, config.CROP_SIZE)),
    #         transforms.RandomOrder([
    #             transforms.ColorJitter(brightness=0.1),
    #             transforms.ColorJitter(saturation=0.2),
    #             transforms.ColorJitter(contrast=0.2),
    #         ]),
    #         transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    #     ]),
    transforms.RandomCrop((config.CROP_SIZE, config.CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

Trans2 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    # transforms.RandomCrop(size=config.CROP_SIZE),
    # AutoAugment(),
    # transforms.RandomChoice([
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.Compose([
    #             transforms.RandomRotation(degrees=10, expand=True),
    #             # transforms.Resize(config.IMG_SIZE),
    #             transforms.CenterCrop(size=config.CROP_SIZE),
    #         ]),
    #         transforms.RandomCrop((config.CROP_SIZE, config.CROP_SIZE)),
    #         transforms.RandomOrder([
    #             transforms.ColorJitter(brightness=0.1),
    #             transforms.ColorJitter(saturation=0.2),
    #             transforms.ColorJitter(contrast=0.2),
    #         ]),
    #         # transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    #     ]),
    transforms.Resize(size=config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

normlize = transforms.Compose([
    transforms.Normalize(norm_mean, norm_std)
])