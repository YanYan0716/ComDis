import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


import algorithm.config as config
from algorithm.TripletDataset import TripletDataset
from algorithm.Siamese import SiameseNetwork
from algorithm.utils import Cos_warmup, save_checkpoint
import algorithm.trans as trans
from algorithm.contrastive import ContrastiveLoss


def train(dataLoader, model, optim, Con_loss,  lrSche,):
    print(f'train 。。。 alpha: {config.ALPHA}, betal: {config.BETAL}, gamma: {config.GAMMA}, lr: {config.LR}, classes: {config.CLASSES_NUM}')
    MinLoss = 1000
    for epoch in range(config.START_EPOCH, config.TOTAL_EPOCH):
        model.train()
        epochLoss = 0
        avgLoss = 0
        tLoss = 0
        length = 0
        for idx, (anchor, pos1, pos2, neg, mask, label) in enumerate(dataLoader):
            length = idx
            anchor, pos1, pos2, neg = anchor.to(config.DEVICE), pos1.to(config.DEVICE), pos2.to(config.DEVICE), neg.to(config.DEVICE)
            mask, label = mask.type(torch.float32).to(config.DEVICE), label.to(config.DEVICE)
            imgs = torch.cat([anchor, pos1, pos2, neg], dim=0)

            optim.zero_grad()
            out1, out2, = model(imgs, mask)

            Fts1 = out1[0]
            Fts2 = out1[1]
            loss = Con_loss(Fts1, Fts2, mask) * config.ALPHA

            avgLoss += loss
            epochLoss += loss
            tLoss += loss
            loss.backward()
            optim.step()

            if idx % config.LOG_BATCHSIZE == 0:
                avgLoss = avgLoss / config.LOG_BATCHSIZE
                tLoss = tLoss / config.LOG_BATCHSIZE
                print(f'[epoch:%3d/' % (epoch) + 'EPOCH: %3d]' % config.TOTAL_EPOCH + '%4d:' % idx
                      + ' [LOSS: %.4f]' % avgLoss + '[Con Loss: %.4f' % tLoss + ']')
                avgLoss = 0
                tLoss = 0
        lrSche.step()
        epochLoss = epochLoss / length
        print(f'MinLoss: {MinLoss}, epochLoss: {epochLoss}')
        if MinLoss > epochLoss:
            MinLoss = epochLoss
            state = {
                'epoch': epoch,
                'model': model.state_dict()
            }
            save_checkpoint(state=state, savepath=config.SAVE_PATH)
            print(f'saving model to {config.SAVE_PATH} ..........................')


def main():
    # dataset
    train_trans = {
        'OriTrans': trans.OriTrain,
        'PosTrans1': trans.PosTrans1,
        'PosTrans2': trans.PosTrans2,
        'NegTrans': trans.NegTrans,
    }
    DS = TripletDataset(
        root_dir=config.ROOT_PATH,
        transform=train_trans,
        img_dir=config.TRAIN_PATH,
        train=True
    )

    train_loader = data.DataLoader(DS, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    # model
    net = SiameseNetwork().to(config.DEVICE)
    if config.CONTINUE:
        print('continue train ...')
        checkpoint = torch.load(config.CONTINUE_PATH, map_location=config.DEVICE)
        net.load_state_dict(checkpoint['model'])
    # loss
    Triplet_loss = ContrastiveLoss()#  torch.nn.TripletMarginLoss(margin=0.8, p=2)

    # optimizer
    optim = torch.optim.Adam(
        params=net.parameters(),
        lr=config.LR,
        # momentum=config.MOMENTUM
    )
    cosWarmUp = Cos_warmup(
        optim,
        epoch_warmup=config.WARMUP_EPOCH,
        epoch_training=config.TOTAL_EPOCH
    )

    train(
        dataLoader=train_loader,
        model=net,
        Con_loss=Triplet_loss,
        optim=optim,
        lrSche=cosWarmUp,
    )


if __name__ == '__main__':
    main()