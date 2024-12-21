import os
import json

storage_dir = "application/storage"
os.makedirs(storage_dir, exist_ok=True)

images_dir = os.path.join(storage_dir, "images")
os.makedirs(images_dir, exist_ok=True)

posts_file = os.path.join(storage_dir, "posts.json")
p_embedding_file = os.path.join(storage_dir, "p_embeddings.json")

users_file = os.path.join(storage_dir, "users.json")
u_embedding_file = os.path.join(storage_dir, "u_embeddings.json")

def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

def save_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f)

posts = load_json(posts_file, [])
p_embeddings = load_json(p_embedding_file, {})
u_embeddings = load_json(u_embedding_file, {})
users = load_json(users_file, {})
recommendation_storage = {}