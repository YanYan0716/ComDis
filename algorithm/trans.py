import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torchvision.transforms as transforms


import algorithm.config as config
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# train
OriTrain = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=True),
        transforms.RandomOrder([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(contrast=0.5),
        ]),
        transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    ]),
    transforms.RandomCrop(size=config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2),
    transforms.Normalize(norm_mean, norm_std)
])

PosTrans1 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=True),
        transforms.RandomOrder([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(contrast=0.5),
        ]),
        transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    ]),
    transforms.RandomCrop(size=config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2),
    transforms.Normalize(norm_mean, norm_std)
])

PosTrans2 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=True),
        transforms.RandomOrder([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(contrast=0.5),
        ]),
        transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    ]),
    transforms.RandomCrop(size=config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2),
    transforms.Normalize(norm_mean, norm_std)
])

NegTrans = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=True),
        transforms.RandomOrder([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(contrast=0.5),
        ]),
        transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    ]),
    transforms.RandomCrop(size=config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2),
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
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=True),
        transforms.RandomOrder([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(contrast=0.5),
        ]),
        transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    ]),
    transforms.RandomCrop(size=config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2),
    transforms.Normalize(norm_mean, norm_std)
])

Trans2 = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, expand=True),
        transforms.RandomOrder([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(contrast=0.5),
        ]),
        transforms.RandomAffine(degrees=10, translate=(0.01, 0.1), scale=(0.9, 1.1)),
    ]),
    transforms.RandomCrop(size=config.CROP_SIZE),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.RandomErasing()], p=0.2),
    transforms.Normalize(norm_mean, norm_std)
])