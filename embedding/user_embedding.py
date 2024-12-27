from models_manager import get_sentence_transformer_model
import numpy as np

from application.storage.storage_manager import users, p_embeddings

def embed_user(user_id):     
    user = users[user_id]
    liked_posts = user["liked_posts"]
    if(len(liked_posts)==0):
        return []

    liked_embeddings = [p_embeddings[post_id] for post_id in liked_posts]
    combined_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)
    return combined_embedding