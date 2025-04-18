import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from application.storage.storage_manager import users, p_embeddings

def similar_posts(user_id, embedding, top_n=5):
    user = users[user_id]
    seen_posts = user["has_seen"]

    candidate_ids = [post_id for post_id in p_embeddings if post_id not in seen_posts]
    candidate_embeddings = [p_embeddings[post_id] for post_id in p_embeddings if post_id not in seen_posts]    

    if(len(candidate_ids)==0):
        return []

    similarities = cosine_similarity(np.array(embedding).reshape(1, -1), np.array(candidate_embeddings))[0]
    post_similarity_pairs = list(zip(candidate_ids, similarities))
    sorted_pairs = sorted(post_similarity_pairs, key=lambda x: x[1], reverse=True)

    sorted_posts = [post_id for post_id, _ in sorted_pairs]

    # adjust this algorithm
    # use a neural network instead
    recommendation = sorted_posts[:top_n] 
    recommendation.append(sorted_posts[-1])
    return recommendation
