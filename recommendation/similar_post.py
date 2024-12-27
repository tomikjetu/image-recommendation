import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from application.storage.storage_manager import users, p_embeddings
from embedding.user_embedding import embed_user

def similar_posts(user_id, top_n=5):
    user = users[user_id]
    seen_posts = user["has_seen"]
    u_embedding = embed_user(user_id)
    if(len(u_embedding)==0):
        return []

    candidate_ids = [post_id for post_id in p_embeddings if post_id not in seen_posts]
    candidate_embeddings = [p_embeddings[post_id] for post_id in p_embeddings if post_id not in seen_posts]    

    similarities = cosine_similarity(u_embedding, candidate_embeddings)[0]
    post_similarity_pairs = list(zip(candidate_ids, similarities))
    sorted_pairs = sorted(post_similarity_pairs, key=lambda x: x[1], reverse=True)

    sorted_posts = [post_id for post_id, _ in sorted_pairs]
    print(sorted_posts)
    return sorted_posts[:top_n]