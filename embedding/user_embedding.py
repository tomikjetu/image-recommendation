from models_manager import get_sentence_transformer_model
import numpy as np

from application.storage.storage_manager import users, p_embeddings, save_json, users_file, p_embedding_file, u_embeddings, u_embedding_file

def embed_user(user_id):     
    user = users[user_id]
    liked_posts = user["liked_posts"]
    
    if(len(liked_posts)==0):
        return []
    
    if(not user["updated"] and user_id in u_embeddings):
        return np.array(u_embeddings[user_id])
    
    print("Embedding user ", user_id)
    print (liked_posts)

    liked_embeddings = [p_embeddings[post_id] for post_id in liked_posts]
    print(liked_embeddings[0][0])
    combined_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)

    u_embeddings[user_id] = combined_embedding.tolist()
    user["updated"] = False

    save_json(users_file, users)
    save_json(u_embedding_file, u_embeddings)

    return combined_embedding