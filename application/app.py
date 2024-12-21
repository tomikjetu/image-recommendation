import logging
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  

import application.routes.index.index
import application.routes.api.upload_image
import application.routes.api.get_image
import application.routes.api.get_recommendation
import application.routes.api.interaction_like

def run(): 
    print("Starting server")
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=80, debug=True)
