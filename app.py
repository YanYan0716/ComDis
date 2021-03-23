from flask import Flask
from flask import render_template


app = Flask(__name__)


@app.route('/hello')
def hello(name='yanqian'):
    return render_template('helloWorld.html', name=name)


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True)
