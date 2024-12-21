from application.app import app
from flask import request
import os
import json

storage_dir = "application/storage"
os.makedirs(storage_dir, exist_ok=True)
images_dir = os.path.join(storage_dir, "images")
os.makedirs(images_dir, exist_ok=True)
data_file = os.path.join(storage_dir, "data.json")

def load_data():
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except:
        data = []
    return data

def save_data(data):
    with open(data_file, "w") as f:
        json.dump(data, f)

data = load_data()

@app.route('/upload_image', methods=['POST'])
def upload_image():
    image = request.files['image']
    image.save(os.path.join(images_dir, image.filename))
    data.append(image.filename)
    save_data(data)

