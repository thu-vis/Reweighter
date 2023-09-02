from flask import Flask
from flask_cors import CORS
from flask_session import Session
from flask_compress import Compress

def create_app():
    app = Flask(__name__)
    CORS(app, resources=r'/*')
    Compress(app)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    Session(app)
    from application.views.API import api
    app.register_blueprint(api)
    return app

def create_model_app():
    app = Flask(__name__)
    CORS(app, resources=r'/*')
    Compress(app)
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
    Session(app)
    from application.views.DEEPAPI import deepapi
    app.register_blueprint(deepapi)
    return app