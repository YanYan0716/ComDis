import sys
sys.path.append('../')
from flask import request, flash, redirect, url_for, render_template, Response
import os
from werkzeug.utils import secure_filename
from torchvision import transforms
from flask_bootstrap import Bootstrap
from flask import Flask
import torch
from PIL import Image
import matplotlib.pyplot as plt


import algorithm.config as config
from algorithm.TripletDataset import transform_invert
from algorithm.model import Model
from algorithm.trans import OriTest, Trans1
from U2net.U2model import U2NETP


"""https://www.pianshen.com/article/3951322566/"""
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
bootstrap = Bootstrap(app)


def normPred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def genImgMask(imgTensor):
    img = imgTensor.unsqueeze(dim=0).type(torch.FloatTensor)
    U2net = U2NETP(3, 1)
    U2net.load_state_dict(torch.load(config.U2WEITHS_DIR, map_location=config.DEVICE))
    U2net.eval()

    d1 = U2net(img)
    pred = d1[:, 0, :, :]
    pred = normPred(pred)
    resize_pred = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.CROP_SIZE, config.CROP_SIZE)),
        transforms.ToTensor(),
    ])
    pred = resize_pred(pred)
    return pred


def imageTrans(img1, img2):
    img1 = Image.open(img1).convert('RGB')
    firstImg = OriTest(img1)
    # Img1Mask = genImgMask(firstImg)
    # firstImg = torch.cat([firstImg, Img1Mask], dim=0).unsqueeze(dim=0)
    # firstImg = firstImg * Img1Mask

    firstImg_ = Trans1(img1)
    # Img1Mask_ = genImgMask(firstImg_)
    # firstImg_ = torch.cat([firstImg_, Img1Mask_], dim=0).unsqueeze(dim=0)
    # firstImg_ = firstImg_ * Img1Mask

    img2 = Image.open(img2).convert('RGB')
    secodeImg = OriTest(img2)
    # Img2Mask = genImgMask(secodeImg)
    # secodeImg = secodeImg * Img2Mask

    firstImg = firstImg.unsqueeze(dim=0)
    firstImg_ = firstImg_.unsqueeze(dim=0)
    secodeImg = secodeImg.unsqueeze(dim=0)
    return firstImg, firstImg_, secodeImg


@app.route('/', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        files = request.files.getlist('selectfile')
        imgfile = []
        uploadpaths = []
        types = ['jpg', 'png']
        basepath = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(basepath):
            os.mkdir(basepath)

        if files:
            for img in files:
                imgname = secure_filename(img.filename)
                if imgname.split('.')[-1] in types:
                    imgfile.append(imgname)
                    uploadpath = os.path.join(basepath, imgname)
                    uploadpaths.append(uploadpath)
                    img.save(uploadpath)  # 将上传的图片保存
                else:
                    flash('Unknown Types!', 'danger')
            flash('upload successfull', 'success')
            """*************加入神经网络进行判断**********************************"""
            net = Model(fts_dim=config.FTS_DIM)
            checkpoint = torch.load(config.BEST_PATH, map_location='cpu')
            net.load_state_dict(checkpoint['model'])
            net.eval()
            ### eval
            firstImg, firstImg_, secondImg = imageTrans(uploadpaths[0], uploadpaths[1])

            # firstImg = firstImg[:, :3, :, :] * firstImg[:, -1:, :, :]
            # firstImg_ = firstImg_[:, :3, :, :] * firstImg_[:, -1:, :, :]
            # secondImg = secondImg[:, :3, :, :] * secondImg[:, -1:, :, :]

            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(transform_invert(firstImg[0], OriTest))
            plt.subplot(1, 4, 2)
            plt.imshow(transform_invert(firstImg_[0], Trans1))
            plt.subplot(1, 4, 3)
            plt.imshow(transform_invert(secondImg[0], OriTest))
            plt.show()

            imgs = torch.cat([firstImg, firstImg_, secondImg], dim=0)
            out1 = net.model(imgs)
            out1 = net.flatten(out1)
            out1 = net.triplet(out1)
            fts = torch.cat([out1[:1], out1[1:2], out1[-1:]], dim=-1)
            output = net.classifier(fts)
            output_ = torch.sigmoid(output).ge(0.51).type(torch.float32).squeeze(dim=-1)
            output_ = str(output_.numpy())
            out0 = str(torch.sigmoid(output).squeeze(dim=-1).detach().numpy())
            ooo=[output_, out0]
            # return render_template('base.html', imglist=imgfile)
            return Response(str(ooo))
        else:
            flash('no File Selected', 'danger')
    return render_template('base.html')


if __name__ == '__main__':
    img1='static\Black_Footed_Albatross_0005_796090.jpg'
    imageTrans(img1, img1)
