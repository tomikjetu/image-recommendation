from application.app import app
from embedding.post_embedding import embed_post
from image_caption.image_caption import get_caption
from flask import request
import os
import uuid
from application.storage.storage_manager import save_json, posts, p_embeddings, images_dir, posts_file, p_embedding_file
from PIL import Image

UPLOAD_ENABLED = True

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if not UPLOAD_ENABLED:
        return "Upload is disabled"
    id = uuid.uuid4().hex
    image = Image.open(request.files['image'])
    
    keyword = request.files['image'].filename.split("_")[0]
    description = f"[{keyword}] {get_caption(image)}"
    
    post =  {
        "id": id,
        "description": description
    }
    posts.append(post)

    embedding = embed_post(post)
    p_embeddings[id] = embedding

    image.save(os.path.join(images_dir, f"{id}.jpg"))
    save_json(posts_file, posts)
    save_json(p_embedding_file, p_embeddings)
    return "Image uploaded successfully"