from flask import request, flash, redirect, url_for, render_template, Response
import os
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap

from flask import Flask
'''
https://www.pianshen.com/article/3951322566/
'''
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
bootstrap = Bootstrap(app)


@app.route('/', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        files = request.files.getlist('selectfile')
        imgfile = []
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
                    img.save(uploadpath)  # 将上传的图片保存
                else:
                    flash('Unknown Types!', 'danger')
            flash('upload successfull', 'success')
            # return render_template('base.html', imglist=imgfile)
            return Response('123123')
        else:
            flash('no File Selected', 'danger')
    return render_template('base.html')


# @app.route('/', methods=['POST', 'GET'])
# def ComDis():
#     print('ok ----------------')
#     return render_template('base.html')