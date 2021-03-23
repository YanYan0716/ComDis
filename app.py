from flask import Flask
from flask import render_template, request, flash, redirect, url_for
from flask_bootstrap import Bootstrap
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/hello')
def hello(name='yanqian'):
    return render_template('helloWorld.html', name=name)


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
