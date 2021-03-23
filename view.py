from flask import request, flash, redirect, url_for, render_template
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
        f = request.files.get('selectfile')
        basepath = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(basepath):
            os.mkdir(basepath)
        if f:
            filename = secure_filename(f.filename)
            types = ['jpg', 'png']
            if filename.split('.')[-1] in types:
                uploadpath = os.path.join(basepath, filename)
                f.save(uploadpath)
                flash('upload successfull ! ', 'success')
                return render_template('base.html', imagename=filename)
            else:
                flash('Unknown Types!', 'danger')
        else:
            flash('no File Selected', 'danger')
    return render_template('base.html')