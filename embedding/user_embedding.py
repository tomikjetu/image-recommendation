from models_manager import get_sentence_transformer_model
import numpy as np

from application.storage.storage_manager import users, p_embeddings, save_json, users_file, p_embedding_file

def embed_user(user_id):     
    user = users[user_id]
    liked_posts = user["liked_posts"]
    
    if(len(liked_posts)==0):
        return []
    
    if(not user["updated"] and p_embeddings[user_id] is not None):
        return p_embeddings[user_id]

    liked_embeddings = [p_embeddings[post_id] for post_id in liked_posts]
    combined_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)

    p_embeddings[user_id] = combined_embedding
    user["updated"] = False

    save_json(users_file, users)
    save_json(p_embeddings_file, p_embeddings)

    return combined_embedding