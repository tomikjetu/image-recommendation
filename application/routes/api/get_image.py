from flask import send_from_directory, jsonify
from application.app import app
import os

from application.storage.storage_manager import images_dir

@app.route('/image/<id>', methods=['GET'])
def image(id):
    image_path = os.path.join(images_dir, f"{id}.jpg")
    
    if os.path.exists(image_path):
        return send_from_directory("storage/images", f"{id}.jpg")
    else:
        return jsonify({"error": "Image not found."}), 404
