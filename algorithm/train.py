import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


import algorithm.config as config
from algorithm.TripletDataset import TripletDataset
from algorithm.model import Model
from algorithm.utils import Cos_warmup, save_checkpoint
from algorithm.test import evalution


def train(dataLoader, model, optim, Triplet_loss, Classifier_loss, lrSche, testDS=None):
    print('training ...')
    BAcc = 0
    for epoch in range(config.TOTAL_EPOCH):
        model.train()
        avgLoss = 0
        for idx, (anchor, pos1, pos2, neg, mask) in enumerate(dataLoader):
            anchor, pos1, pos2, neg = anchor.to(config.DEVICE), pos1.to(config.DEVICE), pos2.to(config.DEVICE), neg.to(config.DEVICE)
            imgs = torch.cat([anchor, pos1, pos2, neg], dim=0)

            optim.zero_grad()
            out1, out2 = model(imgs, mask)
            anchorFts = out1[ :config.BATCH_SIZE]
            posFts = out1[config.BATCH_SIZE : config.BATCH_SIZE*2]
            negFts = out1[-config.BATCH_SIZE: ]
            loss1 = Triplet_loss(anchorFts, posFts, negFts)
            loss2 = Classifier_loss(out2, mask)
            loss = loss1+loss2
            avgLoss += loss
            loss.backward()
            optim.step()

            if idx % config.LOG_BATCHSIZE == 0:
                avgLoss = avgLoss / config.LOG_BATCHSIZE*anchorFts.size(0)
                print(f'[epoch:%3d/' % (epoch) + 'EPOCH: %3d]:' % config.TOTAL_EPOCH + ' [LOSS: %.4f]' % avgLoss)
                avgLoss = 0
        lrSche.step()
        if epoch % config.EVAL == 0:
            acc = evalution(testDS, model)
            if BAcc < acc:
                BAcc = acc
                state = {
                    'epoch': epoch,
                    'model': model.state_dict()
                }
                save_checkpoint(state=state, savepath=config.SAVE_PATH)


def main():
    # dataset
    train_trans = {
        'OriTrans': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ]),
        'PosTrans1': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ]),
        'PosTrans2': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ]),
        'NegTrans': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ]),
    }
    DS = TripletDataset(
        root_dir=config.ROOT_PATH,
        transform=train_trans,
        img_dir=config.TRAIN_PATH,
        train=True
    )

    train_loader = data.DataLoader(DS, batch_size=config.BATCH_SIZE, shuffle=True)

    test_trans = {
        'OriTrans': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ]),
        'Trans1': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ]),
        'Trans2': transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor()
        ]),
    }
    DS_eval = TripletDataset(
        root_dir=config.ROOT_PATH,
        transform=test_trans,
        img_dir=config.TRAIN_PATH,
        train=False
    )

    test_loader = data.DataLoader(DS_eval, batch_size=config.BATCH_SIZE, shuffle=True)

    # model
    net = Model(fts_dim=config.FTS_DIM).to(config.DEVICE)
    # loss
    Triplet_loss = torch.nn.TripletMarginLoss(margin=0.8, p=2)
    Classifier_loss = torch.nn.BCEWithLogitsLoss()
    # optimizer
    optim = torch.optim.SGD(lr=config.LR, momentum=config.MOMENTUM)
    cosWarmUp = Cos_warmup(
        optim,
        epoch_warmup=config.WARMUP_EPOCH,
        epoch_training=config.TOTAL_EPOCH
    )

    train(
        dataLoader=train_loader,
        model=net,
        Triplet_loss=Triplet_loss,
        Classifier_loss=Classifier_loss,
        optim=optim,
        lrSche=cosWarmUp,
        testDS=test_loader,
    )

if __name__ == '__main__':
    main()