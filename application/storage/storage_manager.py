import os
import json
import uuid

storage_dir = "application/storage"
os.makedirs(storage_dir, exist_ok=True)

images_dir = os.path.join(storage_dir, "images")
os.makedirs(images_dir, exist_ok=True)

posts_file = os.path.join(storage_dir, "posts.json")
p_embedding_file = os.path.join(storage_dir, "p_embeddings.json")

users_file = os.path.join(storage_dir, "users.json")
u_embedding_file = os.path.join(storage_dir, "u_embeddings.json")


posts = [] 
p_embeddings = {} 
u_embeddings = {} 
users = {}
recommendation_storage = {} 

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

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.liked_posts = []  
        self.has_seen = []  

    def like_post(self, post_id):
        if post_id not in self.liked_posts:
            self.liked_posts.append(post_id)
            
def generate_user_id():
    return str(uuid.uuid4())

def load_users():
    users_data = load_json(users_file, {})
    for user_id, user_data in users_data.items():
        user = User(user_id)
        user.liked_posts = user_data["liked_posts"]
        user.has_seen = user_data["has_seen"]
        users[user_id] = user
    print(users)

def save_users():
    users_data = {}
    for user_id, user in users.items():
        users_data[user_id] = {
            "liked_posts": user.liked_posts,
            "has_seen": user.has_seen
        }
    save_json(users_file, users_data)


posts = load_json(posts_file, [])
p_embeddings = load_json(p_embedding_file, {})
u_embeddings = load_json(u_embedding_file, {})
load_users()
recommendation_storage = {}