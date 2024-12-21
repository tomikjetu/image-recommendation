from flask import send_from_directory
from application.app import app

@app.route('/')
def serve_html():
    return send_from_directory("routes/index", path="index.html")
