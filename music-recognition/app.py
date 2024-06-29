from flask import Flask
from pyqt import pyqt

app = Flask(__name__)

@app.route('/music_recognition')
def hello_world():
    pyqt()
    return 'flask_test is running!!!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
