import sys
sys.path.append('/content/ComDis')
sys.path.append('./')
sys.path.append('./ComDis')
import torch
import torchvision.transforms as transforms
import torch.utils.data as data


import algorithm.config as config
from algorithm.TripletDataset import TripletDataset
from algorithm.model import Model
from algorithm.utils import Cos_warmup, save_checkpoint
from algorithm.test import evalution
import algorithm.trans as trans


def train(dataLoader, model, optim, Triplet_loss, Classifier_loss, class2_loss,  lrSche, testDS=None):
    print(f'train  alpha: {config.ALPHA}, betal: {config.BETAL}, lr: {config.LR}')
    BAcc = 0
    for epoch in range(config.TOTAL_EPOCH):
        model.train()
        avgLoss = 0
        tLoss = 0
        cLoss = 0
        c2Loss = 0
        for idx, (anchor, pos1, pos2, neg, mask, label) in enumerate(dataLoader):
            anchor, pos1, pos2, neg = anchor.to(config.DEVICE), pos1.to(config.DEVICE), pos2.to(config.DEVICE), neg.to(config.DEVICE)
            mask, label = mask.to(config.DEVICE), label.to(config.DEVICE)
            imgs = torch.cat([anchor, pos1, pos2, neg], dim=0)

            optim.zero_grad()
            out1, out2, ou3 = model(imgs, mask)

            anchorFts = out1[ :config.BATCH_SIZE]
            posFts = out1[config.BATCH_SIZE : config.BATCH_SIZE*2]
            negFts = out1[-config.BATCH_SIZE: ]
            loss1 = Triplet_loss(anchorFts, posFts, negFts) * config.ALPHA
            out2 = out2.type(torch.float32)
            mask = mask.type(torch.float32)
            loss2 = Classifier_loss(out2.squeeze(dim=-1), mask)*config.BETAL
            loss3 = class2_loss(ou3, label)*config.GAMMA
            loss = loss1+loss2+loss3

            avgLoss += loss
            tLoss += loss1
            cLoss += loss2
            c2Loss += loss3
            loss.backward()
            optim.step()

            if idx % config.LOG_BATCHSIZE == 0:
                avgLoss = avgLoss / config.LOG_BATCHSIZE
                tLoss = tLoss / config.LOG_BATCHSIZE
                cLoss = cLoss / config.LOG_BATCHSIZE
                c2Loss = c2Loss / config.LOG_BATCHSIZE
                print(f'[epoch:%3d/' % (epoch) + 'EPOCH: %3d]' % config.TOTAL_EPOCH + '%4d:' % idx
                      + ' [LOSS: %.4f]' % avgLoss + '[Trip Loss: %.4f' % tLoss + '/ Class Loss: %.4f]' % cLoss
                      + ' / c2lass Loss: %.4f]' % c2Loss)
                avgLoss = 0
                tLoss = 0
                cLoss = 0
                c2Loss = 0
        lrSche.step()
        if epoch % config.EVAL == 0:
            acc, corr_num, total_num = evalution(testDS, model)
            if BAcc < acc:
                BAcc = acc
                state = {
                    'epoch': epoch,
                    'model': model.state_dict()
                }
                save_checkpoint(state=state, savepath=config.SAVE_PATH)
                print(f'saving model to {config.SAVE_PATH} ..........................')
            print(f'eval ...\t [acc: %.2f' % acc + '/ BAcc: %.4f]' % BAcc
                  + '[corr_num: %5d' % corr_num + '/ total num: %6d]' % total_num)


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

    test_trans = {
        'OriTrans': trans.OriTest,
        'Trans1': trans.Trans1,
        'Trans2': trans.Trans2,
    }
    DS_eval = TripletDataset(
        root_dir=config.ROOT_PATH,
        transform=test_trans,
        img_dir=config.TEST_PATH,
        train=False
    )

    test_loader = data.DataLoader(DS_eval, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    # model
    net = Model(fts_dim=config.FTS_DIM).to(config.DEVICE)
    # loss
    Triplet_loss = torch.nn.TripletMarginLoss(margin=0.8, p=2)
    Classifier_loss = torch.nn.BCEWithLogitsLoss()
    class2_loss = torch.nn.CrossEntropyLoss()
    # optimizer
    cls_params = list(map(id, net.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in cls_params, net.parameters())
    optim = torch.optim.SGD(
        params=[
            {'params': base_params},
            {'params': net.classifier.parameters(), 'lr': config.LR*10}
        ],
        lr=config.LR,
        momentum=config.MOMENTUM
    )
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
        class2_loss=class2_loss,
        optim=optim,
        lrSche=cosWarmUp,
        testDS=test_loader,
    )


if __name__ == '__main__':
    main()