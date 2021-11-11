from flask import Flask

from .recognition.recognition import recognition_bp


app = Flask(__name__)

app.register_blueprint(recognition_bp, url_prefix='/')
