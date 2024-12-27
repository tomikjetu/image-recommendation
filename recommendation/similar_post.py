import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from application.storage.storage_manager import users, p_embeddings

def similar_posts(user_id, top_n=5):
    user = users[user_id]
    liked_posts = user["liked_posts"]
    seen_posts = user["has_seen"]
    if(len(liked_posts)==0):
        return []

    liked_embeddings = [p_embeddings[post_id] for post_id in liked_posts]
    combined_embedding = np.mean(liked_embeddings, axis=0).reshape(1, -1)

    candidate_ids = [post_id for post_id in p_embeddings if post_id not in seen_posts]
    candidate_embeddings = [p_embeddings[post_id] for post_id in p_embeddings if post_id not in seen_posts]    

    similarities = cosine_similarity(combined_embedding, candidate_embeddings)[0]
    post_similarity_pairs = list(zip(candidate_ids, similarities))
    sorted_pairs = sorted(post_similarity_pairs, key=lambda x: x[1], reverse=True)

    sorted_posts = [post_id for post_id, _ in sorted_pairs]
    print(sorted_posts)
    return sorted_posts[:top_n]